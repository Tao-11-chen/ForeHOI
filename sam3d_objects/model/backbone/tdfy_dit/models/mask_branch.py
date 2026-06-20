# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Per-view amodal-mask output branch (ported from ReconViaGen v6.1).

A SECOND flow-matching output stream that denoises the per-view COMPLETE (amodal)
object silhouette jointly with the SS voxel shape. Per view the noised mask
(res×res, 1ch) is patchified (patch p) into (res/p)² tokens, embedded to
model_channels, tagged with a 2D positional embedding, a per-view-slot embedding,
a modality embedding, and a per-view appearance descriptor (mean of that view's
raw-stream DINO tokens) for viewpoint grounding. The tokens are appended to the
SS shape tokens inside the DiT (extra-token path), share its self-attention with
the 4096 shape tokens (3D↔2D exchange), cross-attend to the same image cond, and
the returned tokens are decoded back to a per-view mask velocity by `from_tokens`.

out_layer is zero-init (DiT/ControlNet style) so the mask velocity is ~0 at
warm-start → the shape branch is minimally disturbed; the branch ramps in as
out_layer learns. Lives as its own module so released ckpts load strict-clean.
"""
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.transformer import AbsolutePositionEmbedder


class PerViewCrossAttn(nn.Module):
    """Per-view cross-attention — each view's mask tokens (Q) attend to that SAME
    view's raw-stream DINO patch tokens (K/V), giving every mask patch
    spatially-addressed image evidence (vs. a single mean-pooled vector per view).

    Uses F.scaled_dot_product_attention directly. The residual writers (`to_out`,
    last MLP linear) are ZERO-INIT, so at warm-start the block is an exact no-op
    and step 0 reproduces the mean-pool path bit-identically.
    """
    def __init__(self, channels: int = 1024, ctx_channels: int = 1024,
                 num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_kv = nn.Linear(ctx_channels, channels * 2, bias=True)
        self.to_out = nn.Linear(channels, channels, bias=True)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))
        nn.init.constant_(self.to_out.weight, 0)
        nn.init.constant_(self.to_out.bias, 0)
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x: (M, Tq, C) mask tokens of one view each; ctx: (M, Tk, C_ctx)
        M, Tq, C = x.shape
        q = self.to_q(self.norm1(x)).reshape(M, Tq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k, v = self.to_kv(ctx).chunk(2, dim=-1)
        k = k.reshape(M, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(M, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.permute(0, 2, 1, 3).reshape(M, Tq, C)
        x = x + self.to_out(a)
        x = x + self.mlp(self.norm2(x))
        return x


class MaskBranch(nn.Module):
    def __init__(self, model_channels: int = 1024, mask_res: int = 64,
                 patch: int = 4, max_views: int = 8,
                 ctx_dim: int = 1024, ctx_grid: int = 37):
        super().__init__()
        assert mask_res % patch == 0
        self.model_channels = model_channels
        self.mask_res = mask_res
        self.patch = patch
        self.grid = mask_res // patch
        self.n_tok = self.grid ** 2
        self.in_dim = patch * patch                      # single channel
        self.input_layer = nn.Linear(self.in_dim, model_channels)
        self.out_layer = nn.Linear(model_channels, self.in_dim)
        # 2D absolute positional embedding over the patch grid
        pe = AbsolutePositionEmbedder(model_channels, 2)
        coords = torch.stack(torch.meshgrid(
            torch.arange(self.grid), torch.arange(self.grid), indexing="ij"), dim=-1
        ).reshape(-1, 2).float()
        self.register_buffer("pos_emb", pe(coords))      # (n_tok, C)
        # per-view cross-attn ctx pos-emb: ctx patch centers mapped into the
        # mask-grid coordinate frame so Q and K share one spatial frame (both
        # crops cover the identical union bbox).
        self.ctx_grid = ctx_grid                          # 37×37 DINO patches @518²
        ctx_coords = torch.stack(torch.meshgrid(
            torch.arange(ctx_grid), torch.arange(ctx_grid), indexing="ij"), dim=-1
        ).reshape(-1, 2).float()
        ctx_coords = (ctx_coords + 0.5) * (self.grid / ctx_grid) - 0.5
        self.register_buffer("ctx_pos_emb", pe(ctx_coords))   # (ctx_grid², C)
        self.xattn = PerViewCrossAttn(model_channels, ctx_dim)
        self.view_emb = nn.Parameter(torch.randn(max_views, model_channels) * 0.02)
        self.modality_emb = nn.Parameter(torch.zeros(model_channels))
        self.view_feat_proj = nn.Linear(model_channels, model_channels)
        # Only out_layer is zero-init: mask velocity ~0 at warm-start so no garbage
        # gradient flows into the shared LoRA.
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def _patchify(self, m: torch.Tensor) -> torch.Tensor:
        # (M, 1, res, res) -> (M, n_tok, p*p)
        M = m.shape[0]; g, p = self.grid, self.patch
        return m.reshape(M, 1, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(M, g * g, p * p)

    def _unpatchify(self, t: torch.Tensor) -> torch.Tensor:
        # (M, n_tok, p*p) -> (M, 1, res, res)
        M = t.shape[0]; g, p = self.grid, self.patch
        return t.reshape(M, g, g, 1, p, p).permute(0, 3, 1, 4, 2, 5).reshape(M, 1, g * p, g * p)

    def to_tokens(self, noised_mask: torch.Tensor, raw_ctx: torch.Tensor) -> torch.Tensor:
        """noised_mask: (B, N, 1, res, res); raw_ctx: (B, N, P, ctx_dim) — the
        FULL per-view raw-stream (union-crop) DINO patch tokens (P = ctx_grid²).
        Computes the global mean (view_feat) AND cross-attends each view's mask
        tokens to its full token field. Returns (B, N·n_tok, C)."""
        B, N = noised_mask.shape[:2]
        assert raw_ctx.shape[2] == self.ctx_pos_emb.shape[0], \
            f"raw_ctx has {raw_ctx.shape[2]} tokens, expected {self.ctx_pos_emb.shape[0]} (ctx_grid²)"
        view_feat = raw_ctx.float().mean(dim=2)          # (B, N, ctx_dim) global summary
        m = noised_mask.reshape(B * N, 1, self.mask_res, self.mask_res).float()
        h = self.input_layer(self._patchify(m))          # (B*N, n_tok, C)
        h = h + self.pos_emb[None]
        h = h.reshape(B, N, self.n_tok, self.model_channels)
        # Cycle the learned per-view-slot embeddings so N is unrestricted (transformer
        # path is variable-length; only this lookup was capped). view_emb stays (8,1024)
        # to match the trained ckpt; arange(N)%8 is bit-identical to [:N] for N<=8.
        view_idx = torch.arange(N, device=h.device) % self.view_emb.shape[0]
        h = h + self.view_emb[view_idx][None, :, None, :]
        h = h + self.modality_emb.view(1, 1, 1, -1)
        h = h + self.view_feat_proj(view_feat)[:, :, None, :]
        # per-view spatial evidence path (zero-init -> no-op at warm-start)
        h = h.reshape(B * N, self.n_tok, self.model_channels)
        ctx = raw_ctx.float().reshape(B * N, -1, raw_ctx.shape[-1]) + self.ctx_pos_emb[None]
        h = self.xattn(h, ctx)
        return h.reshape(B, N * self.n_tok, self.model_channels)

    def from_tokens(self, h: torch.Tensor, n_views: int) -> torch.Tensor:
        # h: (B, N*n_tok, C) -> mask velocity (B, N, 1, res, res)
        B = h.shape[0]
        h = h.reshape(B * n_views, self.n_tok, self.model_channels).float()
        h = F.layer_norm(h, h.shape[-1:])
        return self._unpatchify(self.out_layer(h)).reshape(
            B, n_views, 1, self.mask_res, self.mask_res)
