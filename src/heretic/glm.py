"""Module views of GLM's packed experts for hooks and ordinary PEFT LoRA."""

import torch
import torch.nn.functional as F
from torch import nn

from .config import QuantizationMethod

GLM_PACKED_MODEL_TYPES = frozenset({"glm4_moe_lite", "glm_moe_dsa"})


def validate_glm_quantization(model_type, quantization, checkpoint_quantization=None):
    if model_type not in GLM_PACKED_MODEL_TYPES:
        return
    if quantization != QuantizationMethod.NONE or checkpoint_quantization:
        raise ValueError(
            "GLM packed expert support requires unquantized weights and "
            "quantization='none'. For GLM-5.3, use zai-org/GLM-5.3-BF16; "
            "setting dtype to bfloat16 does not dequantize an FP8 checkpoint."
        )


class GlmExpert(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.down_proj = nn.Linear(
            weight.shape[1],
            weight.shape[0],
            bias=False,
            device=weight.device,
            dtype=weight.dtype,
        )
        # Share the packed storage, without copying the checkpoint.
        self.down_proj.weight = nn.Parameter(weight, requires_grad=False)


class GlmExpertViews(nn.Module):
    def __init__(self, packed):
        super().__init__()
        self.packed = packed
        self.experts = nn.ModuleList([GlmExpert(w) for w in packed.down_proj])

    def forward(self, hidden_states, top_k_index, top_k_weights):
        result = torch.zeros_like(hidden_states)
        for index, expert in enumerate(self.experts):
            tokens, slots = torch.where(top_k_index == index)
            if tokens.numel() == 0:
                continue
            gate, up = F.linear(
                hidden_states[tokens], self.packed.gate_up_proj[index]
            ).chunk(2, dim=-1)
            output = expert.down_proj(self.packed.act_fn(gate) * up)
            output = output * top_k_weights[tokens, slots, None]
            result.index_add_(0, tokens, output.to(result.dtype))
        return result

    def repack(self):
        # PEFT merge has already added each adapter to its projection.
        for index, expert in enumerate(self.experts):
            self.packed.down_proj.data[index].copy_(expert.down_proj.weight)
        return self.packed
