import tqdm
from typing import List, Tuple

import torch
from torch import nn

from .base import BaseAWQForCausalLM
from transformers.models.mllama.modeling_mllama import (
    MllamaForConditionalGeneration as OldMllamaForConditionalGeneration, 
    MllamaSelfAttentionDecoderLayer as OldMllamaSelfAttentionDecoderLayer,
    MllamaCrossAttentionDecoderLayer as OldMllamaCrossAttentionDecoderLayer,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers import AutoProcessor, AutoConfig
from awq.quantize.quantizer import AwqQuantizer

from awq.utils.utils import clear_memory, get_best_device
from awq.utils.calib_data import get_calib_dataset

from tqdm import tqdm
from collections import defaultdict
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)

class MllamaAWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = ["MllamaSelfAttentionDecoderLayer", "MllamaCrossAttentionDecoderLayer"]
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model):
        raise NotImplementedError()

    @staticmethod
    def get_model_layers(model: OldMllamaForConditionalGeneration):
        return model.language_model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldMllamaSelfAttentionDecoderLayer | OldMllamaCrossAttentionDecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldMllamaForConditionalGeneration, device: str):
        m_list = [
            model.language_model.model.rotary_emb,
            model.language_model.model.embed_tokens,
            model.vision_model,
            model.multi_modal_projector
        ]
        for m in m_list:
            m = m.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldMllamaSelfAttentionDecoderLayer | OldMllamaCrossAttentionDecoderLayer, input_feat, module_kwargs):
        layers = []
    
        # attention input
        if isinstance(module, OldMllamaSelfAttentionDecoderLayer):
            if "self_attn.q_proj" in input_feat.keys():
                layers.append(
                    dict(
                        prev_op=module.input_layernorm,
                        layers=[
                            module.self_attn.q_proj,
                            module.self_attn.k_proj,
                            module.self_attn.v_proj,
                        ],
                        inp=input_feat["self_attn.q_proj"],
                        module2inspect=module.self_attn,
                        kwargs=module_kwargs,
                    )
                )
            
            if "self_attn.o_proj" in input_feat.keys() and module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

        elif isinstance(module, OldMllamaCrossAttentionDecoderLayer):
            if "cross_attn.q_proj" in input_feat.keys():
                layers.append(
                    dict(
                        prev_op=module.input_layernorm,
                        layers=[module.cross_attn.q_proj],
                        inp=input_feat["cross_attn.q_proj"],
                        kwargs=module_kwargs,
                    )
                )
        
            if "cross_attn.o_proj" in input_feat.keys() and module.cross_attn.v_proj.weight.shape == module.cross_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.cross_attn.v_proj,
                        layers=[module.cross_attn.o_proj],
                        inp=input_feat["cross_attn.o_proj"],
                    )
                )
            
        if "mlp.gate_proj" in input_feat.keys():
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )
        if "mlp.down_proj" in input_feat.keys():
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )
            
        return layers

    @classmethod
    def from_pretrained(
        self,
        model_path,
        model_type,
        torch_dtype = torch.float16,
        trust_remote_code = True,
        safetensors = True,
        device_map = "auto",
        download_kwargs = None,
        low_cpu_mem_usage = True,
        use_cache = False,
        **model_init_kwargs,
    ):
        if "config" not in model_init_kwargs.keys():
            model_init_kwargs["config"] = AutoConfig.from_pretrained(model_path)
            model_init_kwargs["config"].text_config.use_cache = use_cache
        else:
            model_init_kwargs["config"].text_config.use_cache = use_cache
        model_init_kwargs["config"].torch_dtype = torch_dtype
        model_init_kwargs["config"].vision_config.torch_dtype = torch_dtype
        model_init_kwargs["config"].text_config.torch_dtype = torch_dtype
        
        model = super().from_pretrained(
            model_path,
            model_type,
            torch_dtype = torch.float16,
            trust_remote_code = True,
            safetensors = True,
            device_map = "auto",
            download_kwargs = None,
            low_cpu_mem_usage = True,
            **model_init_kwargs,
        )
        model.processor = AutoProcessor.from_pretrained(model_path)
        return model

    def _load_quantized_modules(
        self, model, quant_config, version, use_exllama, use_exllama_v2, use_ipex=False
    ):
        # Real quantization of weights
        assert not (
            version == "gemv" and (use_exllama or use_exllama_v2 or use_ipex)
        ), "Exllama kernels only support GEMM version."

        # Get blocks of model
        layers = self.get_model_layers(model)

        for i in tqdm(range(len(layers)), desc="Replacing layers..."):
            layer = layers[i]

            # Get every linear layer in a block
            named_linears = get_named_linears(layer)

            modules_to_not_convert = deepcopy(quant_config.modules_to_not_convert)
            if isinstance(layer, OldMllamaCrossAttentionDecoderLayer):
                if "cross_attn" in quant_config.modules_to_not_convert:
                    modules_to_not_convert.append("mlp")
                elif all(item in quant_config.modules_to_not_convert for item in ["cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.o_proj"]):
                    modules_to_not_convert.append("mlp")

            # Filter out the linear layers we don't want to include
            named_linears = exclude_layers_to_not_quantize(
                named_linears, quant_config.modules_to_not_convert
            )

            # Replace activation functions
            self._scale_activations(self, layer)

            # Replace nn.Linear with WQLinear
            for name, module in named_linears.items():
                if use_ipex:
                    q_linear_module = WQLinear_IPEX
                elif version == "marlin":
                    q_linear_module = WQLinear_Marlin
                elif use_exllama:
                    q_linear_module = WQLinear_Exllama
                elif use_exllama_v2:
                    q_linear_module = WQLinear_ExllamaV2
                elif version == "gemm":
                    q_linear_module = WQLinear_GEMM
                elif version == "gemv":
                    q_linear_module = WQLinear_GEMV
                elif version == "gemv_fast":
                    q_linear_module = WQLinear_GEMVFast


                q_linear = q_linear_module.from_linear(
                    module, quant_config.w_bit, quant_config.q_group_size, True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)

            if not use_ipex:
                torch.cuda.empty_cache()
            gc.collect()

class MllamaAwqQuantizer(AwqQuantizer):
    """
    mllama用のquantizer.
    cross attention blockに関してはmodule_to_not_convertで"cross_attn"を指定している時はMllamaCrossAttentionDecoderLayerにおいては"mlp"も量子化の対象外とする．
    """
    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        
        if isinstance(self.calib_data, BatchFeature):
            samples = self.calib_data
        else:
            messages = get_calib_dataset(
                data=self.calib_data,
                tokenizer=self.tokenizer,
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                split=self.split,
                text_column=self.text_column,
            )
            messages = [self.awq_model.processor.decode(m[0]) for m in messages]

            samples = self.awq_model.processor(
                images=None,
                text=messages,
                audio=None,
                videos=None,
                add_special_tokens=False,
                padding="max_length",
                padding_side='left',
                max_length=max_seq_len,
                return_tensors="pt"
            )

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)
            
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(**samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        return modules, layer_kwargs, inps

    def _preprocess(self):
        pass

    def _preprocess_layer_iter(self, layer_index):
        from copy import deepcopy
        common_device = next(self.modules[layer_index].parameters()).device
        if common_device is None or str(common_device) == "cpu":
            if torch.cuda.is_available():
                best_device = "cuda:" + str(layer_index % torch.cuda.device_count())
            else:
                best_device = get_best_device()

            self.modules[layer_index] = self.modules[layer_index].to(best_device)
            common_device = next(self.modules[layer_index].parameters()).device

        if self.module_kwargs.get("position_ids") is not None:
            self.module_kwargs["position_ids"] = self.module_kwargs[
                "position_ids"
            ].to(common_device)

        if self.module_kwargs.get("attention_mask") is not None:
            self.module_kwargs["attention_mask"] = self.module_kwargs[
                "attention_mask"
            ].to(common_device)

        self.inps = self.inps.to(common_device)

        self.prev_modules_to_not_convert = deepcopy(self.modules_to_not_convert)
        if isinstance(self.modules[layer_index], OldMllamaCrossAttentionDecoderLayer):
            if "cross_attn" in self.modules_to_not_convert:
                self.modules_to_not_convert.append("mlp")
            elif all(item in self.modules_to_not_convert for item in ["cross_attn.q_proj", "cross_attn.k_proj", "cross_attn.v_proj", "cross_attn.o_proj"]):
                self.modules_to_not_convert.append("mlp")

    def _postprocess_layer_iter(self, layer_index):
        self.modules_to_not_convert = self.prev_modules_to_not_convert
    
    def quantize(self):
        self._preprocess()#重要!
        for i in tqdm(range(len(self.modules)), desc="AWQ"):
            self._preprocess_layer_iter(i)#重要!

            # [STEP 1]: Get layer, extract linear modules, extract input features
            named_linears = get_named_linears(self.modules[i])

            # Filter out the linear layers we don't want to exclude
            named_linears = exclude_layers_to_not_quantize(
                named_linears, self.modules_to_not_convert
            )

            input_feat = self._get_input_feat(self.modules[i], named_linears)
            clear_memory()

            # [STEP 2]: Compute and apply scale list
            module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                self.modules[i], input_feat, self.module_kwargs
            )
            scales_list = [
                self._search_best_scale(self.modules[i], **layer)
                for layer in module_config
            ]
            apply_scale(self.modules[i], scales_list, input_feat_dict=input_feat)
            scales_list = append_str_prefix(
                scales_list, get_op_name(self.model, self.modules[i]) + "."
            )

            # [STEP 3]: Compute and apply clipping list
            if self.apply_clip:
                clip_list = self._search_best_clip(
                    self.modules[i], named_linears, input_feat
                )
                apply_clip(self.modules[i], clip_list)
                clip_list = append_str_prefix(
                    clip_list, get_op_name(self.model, self.modules[i]) + "."
                )

            # [STEP 4]: Quantize weights
            if not self.export_compatible:
                self._apply_quant(self.modules[i], named_linears)
            self._postprocess_layer_iter(i)#重要!

            clear_memory()

    def _get_input_feat(self, layer, named_linears):
        if len(named_linears)==0:
            return defaultdict(list)
        return super()._get_input_feat(layer, named_linears)
        