import tqdm
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from awq.modules.act import ScaledActivation
from awq.utils.module import set_op_by_name

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
import torch
from torch import nn
import types

class Qwen3MoeAWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "Qwen3MoeDecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        scales = []
        if isinstance(module.mlp, Qwen3MoeSparseMoeBlock):
            for i in range(len(module.mlp.experts)):
                scales.append(
                    dict(
                        scale_name=f"mlp.experts.{i}.dummy_fn",
                        scale_layer=module.mlp.experts[i].dummy_fn,
                        scale_shape=module.mlp.experts[i].gate_proj.in_features
                    )
                )
        
        return dict(is_scalable=True, scales = scales)

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # attention input
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

        if hasattr(module.mlp, "gate"):
            # linear in
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[
                        w
                        for expert in module.mlp.experts
                        for w in [expert.gate_proj, expert.up_proj]
                    ],
                    inp=input_feat["mlp"],
                    module2inspect=module.mlp,
                )
            )

            # hange the prev_op to dummy_fn inserted in each expert and apply the scale parameter associated with dummy_fn.
            # linear out
            for i, expert in enumerate(module.mlp.experts):
                layers.append(
                    dict(
                        prev_op=expert.dummy_fn,
                        layers=[expert.gate_proj, expert.up_proj],
                        inp=input_feat["mlp"],
                        module2inspect=module.mlp,
                    )
                )
                layers.append(
                    dict(
                        prev_op=expert.up_proj,
                        layers=[expert.down_proj],
                        inp=input_feat[f"mlp.experts.{i}.down_proj"],
                    )
                )

        else:
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
        awq_model = super().from_pretrained(
            model_path,
            model_type,
            torch_dtype,
            trust_remote_code,
            safetensors,
            device_map,
            download_kwargs,
            low_cpu_mem_usage,
            use_cache,
            **model_init_kwargs,
        )
        awq_model._insert_dummy_fn()
        return awq_model

    def _load_quantized_modules(
        self, model, quant_config, version, use_exllama, use_exllama_v2, use_ipex=False
    ):
        self._insert_dummy_fn(model)
        super()._load_quantized_modules(self, model, quant_config, version, use_exllama, use_exllama_v2, use_ipex)

    @staticmethod
    def _scale_activations(self, layer):
        scale_dict = self.get_act_for_scaling(layer)
        scales = scale_dict.get('scales')
        if scale_dict["is_scalable"] and scales is None:
            scales = [scale_dict]
        if scale_dict["is_scalable"]:
            for scale in scales:
                if not isinstance(scale["scale_layer"], ScaledActivation):
                    param = next(layer.parameters())
    
                    # get activation scale
                    scale_like = torch.ones(
                        scale["scale_shape"], dtype=param.dtype, device=param.device
                    )
    
                    # scale activation
                    scaled_act = ScaledActivation(scale["scale_layer"], scale_like)
                    set_op_by_name(layer, scale["scale_name"], scaled_act)

    def _insert_dummy_fn(self):
        def forward(self, x):
            x = self.dummy_fn(x)
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj

        if hasattr(self.model, "model"):
            layers = self.model.model.layers
        else:
            layers = self.model.layers
        for layer in layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                for expert in layer.mlp.experts:
                    expert.dummy_fn = nn.Identity()
                    expert.forward = types.MethodType(forward, expert)


    
