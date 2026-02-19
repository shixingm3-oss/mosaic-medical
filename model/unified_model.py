"""
Unified 2D / 3D medical image classification model.

Architecture:
  1. UnifiedPatchEmbed — tokenizes 2D or 3D inputs into [B, N, D]
  2. TransformerEncoder — shared backbone with MoE adapters
  3. Per-task classification heads

Includes:
  - TeacherModel:  EMA copy of UnifiedModel with expert-aware updates.
  - create_model_and_teacher():  convenience factory.
  - load_pretrained_vit_from_npz():  loads ViT-B/16 ImageNet weights.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .patch_embed import UnifiedPatchEmbed
from .transformer_block import TransformerEncoder


class UnifiedModel(nn.Module):
    """Unified 2D/3D classification model with MoE adapters."""

    def __init__(self, num_classes_list, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, drop_rate=0.0,
                 attn_drop_rate=0.0, use_adapter=True,
                 adapter_mode="v2_moe", adapter_bottleneck=64,
                 adapter_bottleneck_a=64, adapter_bottleneck_b=96,
                 adapter_bottleneck_c=192, adapter_dropout=0.0,
                 adapter_scalar=0.1, img_size_2d=224, patch_size_2d=16,
                 in_chans_2d=3, img_size_3d=64, patch_size_3d=8,
                 in_chans_3d=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_adapter = use_adapter
        self.adapter_mode = adapter_mode

        # Tokenizer
        self.patch_embed = UnifiedPatchEmbed(
            img_size_2d, patch_size_2d, in_chans_2d,
            img_size_3d, patch_size_3d, in_chans_3d, embed_dim)
        n2d = self.patch_embed.num_patches_2d
        n3d = self.patch_embed.num_patches_3d

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_2d = nn.Parameter(torch.zeros(1, n2d + 1, embed_dim))
        self.pos_embed_3d = nn.Parameter(torch.zeros(1, n3d + 1, embed_dim))

        # Encoder
        self.encoder = TransformerEncoder(
            depth=depth, dim=embed_dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate,
            use_adapter=use_adapter, adapter_mode=adapter_mode,
            adapter_bottleneck=adapter_bottleneck,
            adapter_bottleneck_a=adapter_bottleneck_a,
            adapter_bottleneck_b=adapter_bottleneck_b,
            adapter_bottleneck_c=adapter_bottleneck_c,
            adapter_dropout=adapter_dropout, adapter_scalar=adapter_scalar)

        # Per-task heads
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, nc) for nc in num_classes_list
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed_2d, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_3d, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for head in self.heads:
            nn.init.trunc_normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward_features(self, x, expert_id=None):
        """Extract CLS feature.  Returns (features [B, D], is_3d)."""
        tokens, is_3d = self.patch_embed(x)
        B = tokens.shape[0]
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        tokens = tokens + (self.pos_embed_3d if is_3d else self.pos_embed_2d)
        tokens = self.encoder(tokens, is_3d=is_3d, expert_id=expert_id)
        return tokens[:, 0], is_3d

    def forward(self, x, task_id=0, return_features=False, expert_id=None):
        """Returns (features [B, D], logits [B, C])."""
        features, _ = self.forward_features(x, expert_id)
        if return_features:
            return features
        return features, self.heads[task_id](features)

    def freeze_backbone(self):
        """Freeze everything except adapters and classification heads."""
        for name, p in self.named_parameters():
            if "adapter" not in name and "expert" not in name and "heads" not in name:
                p.requires_grad = False


# ---------------------------------------------------------------------------
# Teacher (EMA)
# ---------------------------------------------------------------------------

class TeacherModel(nn.Module):
    """EMA teacher with expert-aware parameter updates."""

    def __init__(self, student: UnifiedModel):
        super().__init__()
        self.model = copy.deepcopy(student)
        self.adapter_mode = student.adapter_mode
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, task_id=0, return_features=False, expert_id=None):
        return self.model(x, task_id=task_id,
                          return_features=return_features,
                          expert_id=expert_id)

    @torch.no_grad()
    def ema_update(self, student, momentum=0.999, is_3d=None, expert_id=None):
        """
        Expert-aware EMA update.

        In v2_moe mode, only the active expert's adapter parameters are
        updated; other experts remain untouched.  Shared backbone parameters
        are always updated.
        """
        for (n_t, p_t), (_, p_s) in zip(
                self.model.named_parameters(), student.named_parameters()):

            if self.adapter_mode == "v2_moe" and expert_id and student.use_adapter:
                # Skip inactive experts
                if "expert_a" in n_t and expert_id != "A":
                    continue
                if "expert_b" in n_t and expert_id != "B":
                    continue
                if "expert_c" in n_t and expert_id != "C":
                    continue
            elif self.adapter_mode == "v1" and is_3d is not None and student.use_adapter:
                if "adapter_2d" in n_t and is_3d:
                    continue
                if "adapter_3d" in n_t and not is_3d:
                    continue

            p_t.data.mul_(momentum).add_((1 - momentum) * p_s.data)


# ---------------------------------------------------------------------------
# Pre-trained weight loading (ViT-B/16 ImageNet, JAX .npz -> PyTorch)
# ---------------------------------------------------------------------------

def load_pretrained_vit_from_npz(model: UnifiedModel, npz_path: str):
    """Load ViT-B/16 ImageNet weights from a JAX .npz checkpoint."""
    print(f"Loading ViT pretrained weights from: {npz_path}")
    w = np.load(npz_path)
    loaded = 0

    with torch.no_grad():
        # Patch embedding (2D only; 3D is randomly initialised)
        if "embedding/kernel" in w:
            k = np.transpose(w["embedding/kernel"], (3, 2, 0, 1))
            model.patch_embed.patch_embed_2d.proj.weight.data = torch.from_numpy(k).float()
            loaded += 1
        if "embedding/bias" in w:
            model.patch_embed.patch_embed_2d.proj.bias.data = torch.from_numpy(w["embedding/bias"]).float()
            loaded += 1
        if "cls" in w:
            model.cls_token.data = torch.from_numpy(w["cls"]).float()
            loaded += 1

        # 2D position embedding
        key = "Transformer/posembed_input/pos_embedding"
        if key in w and w[key].shape[1] == model.pos_embed_2d.shape[1]:
            model.pos_embed_2d.data = torch.from_numpy(w[key]).float()
            loaded += 1

        # Transformer blocks
        for idx in range(model.encoder.depth):
            pfx = f"Transformer/encoderblock_{idx}"
            blk = model.encoder.blocks[idx]

            # LayerNorm 1 & 2
            for ln_idx, ln in [(0, blk.norm1), (2, blk.norm2)]:
                s = f"{pfx}/LayerNorm_{ln_idx}/scale"
                b = f"{pfx}/LayerNorm_{ln_idx}/bias"
                if s in w:
                    ln.weight.data = torch.from_numpy(w[s]).float()
                    ln.bias.data = torch.from_numpy(w[b]).float()
                    loaded += 2

            # QKV
            q_key = f"{pfx}/MultiHeadDotProductAttention_1/query/kernel"
            if q_key in w:
                qw = w[q_key].reshape(w[q_key].shape[0], -1).T
                kw = w[f"{pfx}/MultiHeadDotProductAttention_1/key/kernel"]
                kw = kw.reshape(kw.shape[0], -1).T
                vw = w[f"{pfx}/MultiHeadDotProductAttention_1/value/kernel"]
                vw = vw.reshape(vw.shape[0], -1).T
                blk.attn.qkv.weight.data = torch.from_numpy(
                    np.concatenate([qw, kw, vw], 0)).float()
                qb = w[f"{pfx}/MultiHeadDotProductAttention_1/query/bias"].reshape(-1)
                kb = w[f"{pfx}/MultiHeadDotProductAttention_1/key/bias"].reshape(-1)
                vb = w[f"{pfx}/MultiHeadDotProductAttention_1/value/bias"].reshape(-1)
                blk.attn.qkv.bias.data = torch.from_numpy(
                    np.concatenate([qb, kb, vb], 0)).float()
                loaded += 2

            # Output projection
            o_key = f"{pfx}/MultiHeadDotProductAttention_1/out/kernel"
            if o_key in w:
                ow = w[o_key].reshape(-1, w[o_key].shape[-1]).T
                blk.attn.proj.weight.data = torch.from_numpy(ow).float()
                blk.attn.proj.bias.data = torch.from_numpy(
                    w[f"{pfx}/MultiHeadDotProductAttention_1/out/bias"]).float()
                loaded += 2

            # FFN
            for fc_idx, fc in [(0, blk.ffn.fc1), (1, blk.ffn.fc2)]:
                k = f"{pfx}/MlpBlock_3/Dense_{fc_idx}/kernel"
                if k in w:
                    fc.weight.data = torch.from_numpy(w[k].T).float()
                    fc.bias.data = torch.from_numpy(
                        w[f"{pfx}/MlpBlock_3/Dense_{fc_idx}/bias"]).float()
                    loaded += 2

        # Final LayerNorm
        if "Transformer/encoder_norm/scale" in w:
            model.encoder.norm.weight.data = torch.from_numpy(
                w["Transformer/encoder_norm/scale"]).float()
            model.encoder.norm.bias.data = torch.from_numpy(
                w["Transformer/encoder_norm/bias"]).float()
            loaded += 2

    print(f"  Loaded {loaded} tensors.  "
          "3D patch embed, adapters, and heads are randomly initialised.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model_and_teacher(
        num_classes_list, use_adapter=True, adapter_mode="v2_moe",
        adapter_bottleneck=64, adapter_bottleneck_a=64,
        adapter_bottleneck_b=96, adapter_bottleneck_c=192,
        adapter_scalar=0.1, pretrained_path=None,
        embed_dim=768, depth=12, num_heads=12):
    """Create student + teacher pair, optionally loading pretrained weights."""
    student = UnifiedModel(
        num_classes_list=num_classes_list, use_adapter=use_adapter,
        adapter_mode=adapter_mode, adapter_bottleneck=adapter_bottleneck,
        adapter_bottleneck_a=adapter_bottleneck_a,
        adapter_bottleneck_b=adapter_bottleneck_b,
        adapter_bottleneck_c=adapter_bottleneck_c,
        adapter_scalar=adapter_scalar,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads)

    if pretrained_path is not None:
        if pretrained_path.endswith(".npz"):
            load_pretrained_vit_from_npz(student, pretrained_path)
        else:
            print(f"Loading pretrained weights from: {pretrained_path}")
            sd = torch.load(pretrained_path, map_location="cpu")
            missing, unexpected = student.load_state_dict(sd, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")

    teacher = TeacherModel(student)
    return student, teacher