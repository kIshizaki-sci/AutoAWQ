import copy
import tqdm
import torch
from typing import List, Tuple
from .base import BaseAWQForCausalLM
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3DecoderLayer as OldGemma3DecoderLayer,
    Gemma3ForConditionalGeneration as OldGemma3ForConditionalGeneration,
)
class Gemma3ForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = "Gemma3DecoderLayer"
    max_new_tokens_key = "max_position_embeddings"

    @staticmethod
    def fuse_layers(model):
        raise NotImplementedError()

    @staticmethod
    def get_model_layers(model: OldGemma3ForConditionalGeneration):
        return model.language_model.model.layers

    @staticmethod
    def get_act_for_scaling(module: OldGemma3DecoderLayer):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: OldGemma3ForConditionalGeneration, device: str):
        m_list = [
            model.vision_tower,
            model.multi_modal_projector,
            model.language_model.lm_head,
            model.language_model.model.embed_tokens,
            model.language_model.model.norm,
            model.language_model.model.rotary_emb,
            model.language_model.model.rotary_emb_local,
        ]
        for m in m_list:
            m = m.to(device)

    @staticmethod
    def get_layers_for_scaling(module: OldGemma3DecoderLayer, input_feat, module_kwargs):
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

        layers.append(
            dict(
                prev_op=module.pre_feedforward_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )

        layers.append(
            dict(
                prev_op=module.mlp.up_proj,# or module.mlp.act_fn
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )

        return layers
    