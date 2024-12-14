import tqdm
from typing import List, Tuple, Dict

import torch
from torch import nn
import logging
from datasets import load_dataset

from .base import BaseAWQForCausalLM
from transformers.models.mllama.modeling_mllama import (
    MllamaForConditionalGeneration as OldMllamaForConditionalGeneration, 
    MllamaSelfAttentionDecoderLayer as OldMllamaSelfAttentionDecoderLayer,
    MllamaCrossAttentionDecoderLayer as OldMllamaCrossAttentionDecoderLayer,
)
from transformers.feature_extraction_utils import BatchFeature
from awq.quantize.quantizer import AwqQuantizer

from awq.utils.utils import clear_memory, get_best_device

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
    
    #def __init__(self):
    #    raise NotImplementedError()

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

        elif isinstance(module, OldMllamaCrossAttentionDecoderLayer):
            layers.append(
                dict(
                    prev_op=module.input_layernorm,
                    layers=[module.cross_attn.q_proj],
                    inp=input_feat["cross_attn.q_proj"],
                    kwargs=module_kwargs,
                )
            )
            """
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
            """
        
            # attention out
            # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
            if module.cross_attn.v_proj.weight.shape == module.cross_attn.o_proj.weight.shape:
                layers.append(
                    dict(
                        prev_op=module.cross_attn.v_proj,
                        layers=[module.cross_attn.o_proj],
                        inp=input_feat["cross_attn.o_proj"],
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

class MllamaAwqQuantizer(AwqQuantizer):

    def init_quant(self, n_samples=128, max_seq_len=512):
        modules = self.awq_model.get_model_layers(self.model)

        if isinstance(self.calib_data, BatchFeature):
            samples = self.calib_data
        else:
            samples = get_calib_dataset(
                data=self.calib_data,
                tokenizer=self.tokenizer,
                processor=self.processor,
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                split=self.split,
                text_column=self.text_column,
            )
            samples = torch.cat(samples, dim=0)
        # samplesに画像を使ったデータセットを追加してshuffle

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
            self.model(**samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        #print(layer_kwargs.keys(), samples.keys())
        dupl_keys = []
        for k in samples.keys():
            if k in layer_kwargs.keys():
                dupl_keys.append(k)
        for k in dupl_keys:
            samples[k] = samples[k].to("cpu")
            del samples[k]
        #layer_kwargs = self.model.prepare_inputs_for_generation(**samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        #layer_kwargs.pop("input_ids")

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

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel mean of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        org_shape = weight.shape
        # The weights are reshaped to be organised by quantization group
        weight = weight.view(-1, self.group_size)
        # Calculates the relative magnitude of the weights within each of the quantization groups,
        # and rescales each group individually so that each group has weights on a 0-1 scale.
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Resizes the rescaled weight matrix back up to its original dimensions
        w_scale = w_scale.view(org_shape)
        # Gets the average rescaled magnitude for each output channel
        w_mean = w_scale.mean(0)
        clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for sum calculation
        x_sum = torch.zeros(num_channels, dtype=torch.float32, device=inp.device)
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_sum = inp_flat[i:end].to(torch.float32).sum(dim=0)
            x_sum += chunk_sum.to(inp.device)

        x_mean = (x_sum / num_elements).to(inp.dtype)
        clear_memory(x_sum)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)

        # [STEP 4]: Compute loss
        best_scales = self._compute_best_scale(
            inp, w_mean, x_mean, module2inspect, layers, fp16_output, module_kwargs
        )

        if isinstance(prev_op, str):
            return (
                prev_op,
                tuple([get_op_name(module, m) for m in layers]),
                best_scales,
            )
        else:
            return (
                get_op_name(module, prev_op),
                tuple([get_op_name(module, m) for m in layers]),
                best_scales,
            )