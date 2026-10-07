"""Module views of GLM's packed experts for hooks and ordinary PEFT LoRA."""

import torch
import torch.nn.functional as F
from torch import nn


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
