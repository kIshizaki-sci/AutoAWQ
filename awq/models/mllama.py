import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from transformers.models.mllama.modeling_mllama import (
    MllamaForConditionalGeneration as OldMllamaForConditionalGeneration, 
    MllamaSelfAttentionDecoderLayer as OldMllamaSelfAttentionDecoderLayer,
    MllamaCrossAttentionDecoderLayer as OldMllamaCrossAttentionDecoderLayer,
)
from transformers.models.mllama.processing_mllama import MllamaProcessor

from awq.quantize.quantizer import AwqQuantizer

class MllamaAWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = ["MllamaSelfAttentionDecoderLayer", "MllamaCrossAttentionDecoderLayer"]
    max_seq_len_key = "max_position_embeddings"
    
    def __init__(self):
        raise NotImplementedError()

    @staticmethod
    def fuse_layers(model):
        raise NotImplementedError()

    @staticmethod
    def get_model_layers(model: OldMllamaForConditionalGeneration):
        return model.language_model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldMllamaSelfAttentionDecoderLayer | OldMllamaCrossAttentionDecoderLayer):
        return dict(is_scalable=False)

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
        
            # attention out
            # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.self_attn.v_proj,
                        layers=[module.self_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )
        
            # linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )
        
            # linear 2
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )

        elif isinstance(module, OldMllamaCrossAttentionDecoderLayer):
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[module.cross_attn.q_proj],
                    inp=input_feat["self_attn.q_proj"],
                    kwargs=module_kwargs,
                )
            )
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[
                        module.cross_attn.k_proj,
                        module.cross_attn.v_proj,
                    ],
                    inp=input_feat["cross_attn.k_proj"],
                    module2inspect=module.cross_attn,
                    kwargs=module_kwargs,
                )
            )
        
            # attention out
            # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
            if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.cross_attn.v_proj,
                        layers=[module.cross_attn.o_proj],
                        inp=input_feat["self_attn.o_proj"],
                    )
                )

            # linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )
        
            # linear 2
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )
    
        return layers

    @torch.no_grad()
    def quantize(
        self,
        tokenizer = None,
        quant_config = {},
        calib_data: Annotated[
            Union[str, List[str], List[Union[Tuple[str, None], Tuple[str, numpy.ndarray(dtype=np.uint8)]] ],
            Doc(
                "The calibration dataset. Either a string pointing to Huggingface or a list of preloaded examples."
            ),
        ] = "pileval",
        split = "train",
        text_column = "text",
        duo_scaling = True,
        export_compatible = False,
        apply_clip = True,
        n_parallel_calib_samples = None,
        max_calib_samples = 128,
        max_calib_seq_len = 512,
        max_chunk_memory = 1024 * 1024 * 1024,
        quantizer_cls = AwqQuantizer,
        processor: Annotated[
            MllamaProcessor, Doc("The processor to use for quantization.")
        ] = None,
        **kwargs,
    ):
        """
        The main quantization function that you can use to quantize your model.

        Example:

        ```python
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        model_path = "..."
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }
        model.quantize(tokenizer, quant_config)
        ```
        """
        self.quant_config: AwqConfig = AwqConfig.from_dict(quant_config)

        if hasattr(self, "modules_to_not_convert"):
            self.quant_config.modules_to_not_convert = self.modules_to_not_convert

        # calib_dataの中にimageが含まれていればMllamaAWQuantizerを使う．
        if quantizer_cls in not None:
            self.quantizer = quantizer_cls(
                self,
                self.model,
                tokenizer,
                self.quant_config.w_bit,
                self.quant_config.q_group_size,
                self.quant_config.zero_point,
                self.quant_config.version,
                calib_data,
                split,
                text_column,
                duo_scaling,
                modules_to_not_convert=self.quant_config.modules_to_not_convert,
                export_compatible=export_compatible,
                apply_clip=apply_clip,
                n_parallel_calib_samples=n_parallel_calib_samples,
                max_calib_samples=max_calib_samples,
                max_calib_seq_len=max_calib_seq_len,
                max_chunk_memory=max_chunk_memory,
                **kwargs,
            )
        elif isinstance(calib_data, list) and :# 「calib_dataの要素がすべてstr」ではない．
            if processor is None:
                raise #画像データを用いてキャリブレーションを実行する際は引数processorを指定してください．
            self.quantizer = MllamaAWQuantizer(
                self,
                self.model,
                tokenizer,
                self.quant_config.w_bit,
                self.quant_config.q_group_size,
                self.quant_config.zero_point,
                self.quant_config.version,
                calib_data,
                split,
                text_column,
                duo_scaling,
                processor,
                modules_to_not_convert=self.quant_config.modules_to_not_convert,
                export_compatible=export_compatible,
                apply_clip=apply_clip,
                n_parallel_calib_samples=n_parallel_calib_samples,
                max_calib_samples=max_calib_samples,
                max_calib_seq_len=max_calib_seq_len,
                max_chunk_memory=max_chunk_memory,
                **kwargs,
            )
        else:
            self.quantizer = AwqQuantizer(
                self,
                self.model,
                tokenizer,
                self.quant_config.w_bit,
                self.quant_config.q_group_size,
                self.quant_config.zero_point,
                self.quant_config.version,
                calib_data,
                split,
                text_column,
                duo_scaling,
                modules_to_not_convert=self.quant_config.modules_to_not_convert,
                export_compatible=export_compatible,
                apply_clip=apply_clip,
                n_parallel_calib_samples=n_parallel_calib_samples,
                max_calib_samples=max_calib_samples,
                max_calib_seq_len=max_calib_seq_len,
                max_chunk_memory=max_chunk_memory,
                **kwargs,
            )
        self.quantizer.quantize()

        self.is_quantized = True

class MllamaAWQuantizer(AwqQuantizer):
        def __init__(
        self,
        awq_model,
        model,
        tokenizer,
        w_bit,
        group_size,
        zero_point,
        version,
        calib_data,
        split,
        text_column,
        duo_scaling,
        processor
        modules_to_not_convert=None,
        export_compatible=False,
        apply_clip=True,
        n_parallel_calib_samples=None,
        max_calib_samples=128,
        max_calib_seq_len=512,
        max_chunk_memory=1024 * 1024 * 1024,
    ) -> None:
        self.awq_model = awq_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.zero_point = zero_point
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.duo_scaling = duo_scaling
        self.processor = processor
        self.export_compatible = export_compatible
        self.apply_clip = apply_clip
        self.n_parallel_calib_samples = n_parallel_calib_samples
        self.max_calib_samples = max_calib_samples
        self.max_calib_seq_len = max_calib_seq_len
        self.max_chunk_memory = max_chunk_memory
        self.modules_to_not_convert = (
            modules_to_not_convert if modules_to_not_convert is not None else []
        )
        self.modules, self.module_kwargs, self.inps = self.init_quant(
            n_samples=self.max_calib_samples, max_seq_len=self.max_calib_seq_len
        )

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)
        # ここは変更すべし
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            processor=self.processor
            n_samples=n_samples,
            max_seq_len=max_seq_len,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.awq_model.move_embed(self.model, best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
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
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        return modules, layer_kwargs, inps
