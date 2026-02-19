"""
Adapter modules for parameter-efficient fine-tuning.

Implements three adapter variants:
  - Adapter:          Single bottleneck adapter (AdaptFormer-style).
  - ModalityAdapter:  Dual-path adapter (2D / 3D), used in V1 mode.
  - MoEAdapter:       Three-expert mixture (A / B / C), used in V2 mode.

Reference:
    Chen et al., "AdaptFormer: Adapting Vision Transformers for Scalable
    Visual Recognition", NeurIPS 2022.  https://arxiv.org/abs/2205.13535
"""

import math
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Single bottleneck adapter: LN -> Down -> ReLU -> Up -> Scale."""

    def __init__(self, d_model=768, bottleneck=64, dropout=0.0,
                 init_option="lora", adapter_scalar=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, bottleneck)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = adapter_scalar
        self._init_weights(init_option)

    def _init_weights(self, init_option):
        if init_option == "lora":
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_proj.bias)
            # Zero init on up_proj ensures adapter output is zero at start.
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)
        elif init_option == "bert":
            nn.init.normal_(self.down_proj.weight, std=0.02)
            nn.init.normal_(self.up_proj.weight, std=0.02)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
        else:
            raise ValueError(f"Unknown init_option: {init_option}")

    def forward(self, x):
        """x: [B, N, D] -> adapter output [B, N, D] (without residual)."""
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x * self.scale


class ModalityAdapter(nn.Module):
    """V1: Dual-path adapter with separate 2D and 3D branches."""

    def __init__(self, d_model=768, bottleneck=64, dropout=0.0,
                 init_option="lora", adapter_scalar=0.1):
        super().__init__()
        kw = dict(d_model=d_model, bottleneck=bottleneck, dropout=dropout,
                  init_option=init_option, adapter_scalar=adapter_scalar)
        self.adapter_2d = Adapter(**kw)
        self.adapter_3d = Adapter(**kw)

    def forward(self, x, is_3d=False):
        return self.adapter_3d(x) if is_3d else self.adapter_2d(x)


class MoEAdapter(nn.Module):
    """
    V2: Three-expert Mixture-of-Experts adapter with hard routing.

    Experts:
      A — Bio-Medical (RGB, microscopic texture)
      B — Radiology   (grayscale, macroscopic geometry)
      C — Volumetric  (3D voxels, spatial structure)
    """

    def __init__(self, d_model=768, bottleneck_a=64, bottleneck_b=96,
                 bottleneck_c=192, dropout=0.0, init_option="lora",
                 adapter_scalar=0.1):
        super().__init__()
        kw = dict(d_model=d_model, dropout=dropout,
                  init_option=init_option, adapter_scalar=adapter_scalar)
        self.expert_a = Adapter(bottleneck=bottleneck_a, **kw)
        self.expert_b = Adapter(bottleneck=bottleneck_b, **kw)
        self.expert_c = Adapter(bottleneck=bottleneck_c, **kw)

    def forward(self, x, expert_id="A"):
        if expert_id == "A":
            return self.expert_a(x)
        elif expert_id == "B":
            return self.expert_b(x)
        elif expert_id == "C":
            return self.expert_c(x)
        raise ValueError(f"Unknown expert_id: {expert_id}")

    def get_expert(self, expert_id):
        return {"A": self.expert_a, "B": self.expert_b,
                "C": self.expert_c}[expert_id]