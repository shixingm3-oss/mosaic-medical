"""
2D / 3D Patch Embedding (MedCoSS-style unified tokenizer).

Maps raw images into a shared token space so that a single Transformer
backbone can process both 2D and 3D medical inputs:
  - 2D: [B, C, H, W]       -> [B, N, D]   (N = 196 for 224x224, P=16)
  - 3D: [B, C, D, H, W]    -> [B, N, D]   (N = 512 for 64^3,   P=8)
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbed2D(nn.Module):
    """2D patch embedding via Conv2d."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                       # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)       # [B, N, D]
        return self.norm(x)


class PatchEmbed3D(nn.Module):
    """3D patch embedding via Conv3d."""

    def __init__(self, img_size=64, patch_size=8, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 3
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                       # [B, D, D', H', W']
        x = x.flatten(2).transpose(1, 2)       # [B, N, D]
        return self.norm(x)


class UnifiedPatchEmbed(nn.Module):
    """Unified tokenizer that dispatches 4D -> 2D and 5D -> 3D automatically."""

    def __init__(self, img_size_2d=224, patch_size_2d=16, in_chans_2d=3,
                 img_size_3d=64, patch_size_3d=8, in_chans_3d=1,
                 embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed_2d = PatchEmbed2D(img_size_2d, patch_size_2d,
                                           in_chans_2d, embed_dim)
        self.patch_embed_3d = PatchEmbed3D(img_size_3d, patch_size_3d,
                                           in_chans_3d, embed_dim)
        self.num_patches_2d = self.patch_embed_2d.num_patches
        self.num_patches_3d = self.patch_embed_3d.num_patches

    def forward(self, x) -> Tuple[torch.Tensor, bool]:
        """Returns (tokens [B, N, D], is_3d)."""
        if x.dim() == 4:
            return self.patch_embed_2d(x), False
        elif x.dim() == 5:
            return self.patch_embed_3d(x), True
        raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")