from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from loguru import logger

from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor
from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.transformer import (
    ModulatedSparseTransformerCrossBlock,
)
from sam3d_objects.model.backbone.tdfy_dit.modules.transformer import (
    ModulatedTransformerCrossBlock,
)
from sam3d_objects.model.backbone.tdfy_dit.modules.attention.modules import (
    MultiHeadAttention,
)

import torch.nn as nn
from sam3d_objects.model.backbone.generator.classifier_free_guidance import (
    ClassifierFreeGuidance,
    PointmapCFG,
    ClassifierFreeGuidanceWithExternalUnconditionalProbability,
)


class CrossAttentionLogger:
    """
    Utility for capturing cross-attention maps during inference.

    This logger registers forward hooks on cross-attention layers, computes the
    head-averaged attention weights, and saves them per stage/layer/step/view.
    """

    def __init__(
        self,
        save_dir: Path,
        enabled_stages: Optional[Sequence[str]] = None,
        layer_indices: Optional[Sequence[int]] = None,
        reduce_heads: bool = True,
        save_coords: bool = False,  # 是否保存 SLAT 阶段的空间坐标（默认不保存）
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.enabled_stages: Set[str] = (
            set(enabled_stages) if enabled_stages else {"ss", "slat"}
        )
        self.layer_requests: Optional[List[int]] = (
            list(layer_indices) if layer_indices else None
        )
        self.reduce_heads = reduce_heads
        self.save_coords = save_coords

        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._patched_generators: Dict[str, Tuple[object, callable]] = {}
        self._stage_layers: Dict[str, int] = {}
        self._stage_targets: Dict[str, Set[int]] = {}

        self._current_stage: Optional[str] = None
        self._current_view: int = -1
        self._num_views: int = 1
        self._current_step: Dict[str, int] = {}
        self._current_time: Dict[str, float] = {}
        self._current_branch: str = "unknown"  # 'cond', 'uncond', 'pm', etc.
        self._event_counter: Counter = Counter()
        
        # 存储 SLAT 阶段的空间坐标 (coords)
        self._slat_coords: Optional[torch.Tensor] = None
        self._coords_saved: bool = False  # 避免重复保存

    # ------------------------------------------------------------------ Public API
    def attach_to_pipeline(self, pipeline) -> None:
        models = getattr(pipeline, "models", {})
        if "ss_generator" in models and "ss" in self.enabled_stages:
            self._instrument_generator(models["ss_generator"], "ss")
        if "slat_generator" in models and "slat" in self.enabled_stages:
            self._instrument_generator(models["slat_generator"], "slat")

    def start_stage(self, stage: str) -> None:
        if stage not in self.enabled_stages:
            return
        self._current_stage = stage
        self._current_view = 0
        self._current_step[stage] = -1
        self._current_time[stage] = -1.0
        # 重置该 stage 相关的 event counter，避免多次推理时 idx 累积
        keys_to_reset = [k for k in list(self._event_counter.keys()) if k[0] == stage]
        for k in keys_to_reset:
            del self._event_counter[k]

    def set_num_views(self, num_views: int) -> None:
        self._num_views = max(1, int(num_views))

    def set_view(self, view_idx: int) -> None:
        self._current_view = int(view_idx)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for stage, (generator, original_fn) in self._patched_generators.items():
            generator.generate_iter = original_fn
            logger.debug(f"[CrossAttentionLogger] Restored generator for stage {stage}")
        self._patched_generators.clear()

    # ------------------------------------------------------------------ Instrumentation
    def _instrument_generator(self, generator, stage: str) -> None:
        # Attach logger to CFG wrappers so we can distinguish cond / uncond branches
        cfg_wrapper = getattr(generator, "reverse_fn", None)
        backbone = getattr(cfg_wrapper, "backbone", None)
        if backbone is None:
            logger.warning(f"[CrossAttentionLogger] No backbone found for stage {stage}")
            return

        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            logger.warning(
                f"[CrossAttentionLogger] Backbone has no blocks for stage {stage}"
            )
            return

        num_layers = len(blocks)
        self._stage_layers[stage] = num_layers
        target_layers = self._normalize_layers(self.layer_requests, num_layers)
        self._stage_targets[stage] = target_layers
        if not target_layers:
            logger.warning(
                f"[CrossAttentionLogger] No valid layer indices for stage {stage}, "
                "nothing will be recorded."
            )

        for idx, block in enumerate(blocks):
            if idx not in target_layers:
                continue
            cross_attn = getattr(block, "cross_attn", None)
            if cross_attn is None:
                continue
            # MM-DiT (SS) 使用 ModuleDict 存多路 cross-attn，这里只关心 'shape' 对应的 cross-attn
            if isinstance(cross_attn, nn.ModuleDict):
                for latent_name, sub in cross_attn.items():
                    if not isinstance(sub, MultiHeadAttention):
                        continue
                    # 只记录 shape latent 的 cross-attention（大量 tokens，对应体素）
                    if latent_name != "shape":
                        continue
                    hook = sub.register_forward_hook(
                        self._make_hook(stage, idx, sub, latent_name=latent_name)
                    )
                    self._handles.append(hook)
                    logger.info(
                        f"[CrossAttentionLogger] Hooked stage={stage}, layer={idx}, "
                        f"latent={latent_name}, module={sub.__class__.__name__}"
                    )
            else:
                hook = cross_attn.register_forward_hook(
                    self._make_hook(stage, idx, cross_attn, latent_name=None)
                )
                self._handles.append(hook)
                logger.info(
                    f"[CrossAttentionLogger] Hooked stage={stage}, layer={idx}, "
                    f"module={cross_attn.__class__.__name__}"
                )

        if stage not in self._patched_generators:
            original_generate_iter = generator.generate_iter

            # 使用默认参数显式绑定 stage 和 generator，避免闭包捕获问题
            def wrapped_generate_iter(
                *args,
                _stage=stage,
                _gen=generator,
                _orig=original_generate_iter,
                **kwargs
            ):
                self._current_step[_stage] = -1
                self._current_time[_stage] = -1.0
                logger.info(
                    f"[CrossAttentionLogger] Start generate_iter for stage={_stage}, "
                    f"inference_steps={getattr(_gen, 'inference_steps', 'unknown')}"
                )
                for step_idx, (t, x_t, extra) in enumerate(
                    _orig(*args, **kwargs)
                ):
                    self._current_step[_stage] = step_idx
                    t_val = float(t.item() if torch.is_tensor(t) else t)
                    self._current_time[_stage] = t_val
                    yield t, x_t, extra

            generator.generate_iter = wrapped_generate_iter
            self._patched_generators[stage] = (generator, original_generate_iter)

        # Attach this logger to CFG modules so they can set branch information
        if isinstance(
            cfg_wrapper,
            (
                ClassifierFreeGuidance,
                PointmapCFG,
                ClassifierFreeGuidanceWithExternalUnconditionalProbability,
            ),
        ):
            setattr(cfg_wrapper, "_attention_logger", self)
            logger.info(
                f"[CrossAttentionLogger] Attached to CFG wrapper for stage={stage} "
                f"({cfg_wrapper.__class__.__name__})"
            )

    def _normalize_layers(
        self, requested: Optional[Sequence[int]], total: int
    ) -> Set[int]:
        if not requested:
            return {total - 1} if total > 0 else set()
        normalized: Set[int] = set()
        for idx in requested:
            idx_int = int(idx)
            if idx_int < 0:
                idx_int = total + idx_int
            if 0 <= idx_int < total:
                normalized.add(idx_int)
            else:
                logger.warning(
                    f"[CrossAttentionLogger] Layer index {idx} is invalid for total={total}"
                )
        return normalized

    # ------------------------------------------------------------------ Hook implementation
    def _make_hook(self, stage: str, layer_idx: int, module, latent_name: Optional[str] = None):
        def hook(_module, inputs, _output):
            if stage not in self.enabled_stages:
                return
            if layer_idx not in self._stage_targets.get(stage, set()):
                return
            if len(inputs) < 2:
                return
            query, context = inputs[0], inputs[1]
            if self._current_step.get(stage, -1) < 0:
                # 尚未进入任何有效的 diffusion step，跳过以避免 step-01
                return
            with torch.no_grad():
                attn = self._compute_attention(module, query, context)
            if attn is None:
                return
            self._store_attention(stage, layer_idx, attn, latent_name=latent_name)

        return hook

    def _compute_attention(self, module, query, context) -> Optional[torch.Tensor]:
        if query is None or context is None:
            return None

        # Handle multi-view condition tensor
        context = self._select_view_slice(context)

        if isinstance(query, SparseTensor):
            return self._compute_sparse_attention(module, query, context)
        if torch.is_tensor(query):
            return self._compute_dense_attention(module, query, context)
        # Unsupported input type
        return None

    def _select_view_slice(self, context):
        if torch.is_tensor(context) and context.dim() == 4:
            # context shape: [num_views, B, L, C]
            view_idx = (
                self._current_view
                if 0 <= self._current_view < context.shape[0]
                else 0
            )
            return context[view_idx]
        return context

    def _compute_dense_attention(self, module, query, context):
        if not (torch.is_tensor(query) and torch.is_tensor(context)):
            return None
        if query.dim() == 2:
            query = query.unsqueeze(0)
        if context.dim() == 2:
            context = context.unsqueeze(0)
        if query.shape[0] != context.shape[0]:
            if context.shape[0] == 1 and query.shape[0] > 1:
                context = context.expand(query.shape[0], -1, -1)
            else:
                return None

        q = module.to_q(query)
        kv = module.to_kv(context)
        num_heads = module.num_heads
        head_dim = module.channels // num_heads
        q = q.reshape(q.shape[0], q.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
        kv = kv.reshape(kv.shape[0], kv.shape[1], 2, num_heads, head_dim)
        k = kv[:, :, 0].permute(0, 2, 1, 3)
        if module.qk_rms_norm:
            q = module.q_rms_norm(q)
            k = module.k_rms_norm(k)
        attn = self._scaled_attention(q, k)
        return attn

    def _compute_sparse_attention(self, module, query_sparse, context):
        if not isinstance(query_sparse, SparseTensor):
            return None

        layouts = query_sparse.layout
        feats = query_sparse.feats
        batch = len(layouts)
        results = []
        
        # ★ 保存 downsample 后的 coords（这才是和 attention 维度一致的 coords）
        # 只在 SLAT 阶段且开启 save_coords 时保存
        if (self._current_stage == "slat" and self.save_coords and 
            not self._coords_saved and hasattr(query_sparse, 'coords')):
            self._slat_coords = query_sparse.coords.detach().cpu().clone()
            logger.info(
                f"[CrossAttentionLogger] Captured SLAT coords from SparseTensor: "
                f"shape={tuple(self._slat_coords.shape)}, "
                f"x_range=[{self._slat_coords[:, 1].min()}, {self._slat_coords[:, 1].max()}], "
                f"y_range=[{self._slat_coords[:, 2].min()}, {self._slat_coords[:, 2].max()}], "
                f"z_range=[{self._slat_coords[:, 3].min()}, {self._slat_coords[:, 3].max()}]"
            )
        
        for batch_idx in range(batch):
            slc = layouts[batch_idx]
            q_slice = feats[slc].unsqueeze(0)  # [1, L, C]
            ctx = context
            if torch.is_tensor(ctx) and ctx.shape[0] > batch:
                ctx_slice = ctx[batch_idx : batch_idx + 1]
            else:
                ctx_slice = ctx if torch.is_tensor(ctx) else ctx
            if ctx_slice is None:
                continue
            if torch.is_tensor(ctx_slice) and ctx_slice.dim() == 2:
                ctx_slice = ctx_slice.unsqueeze(0)
            dense_attn = self._compute_dense_attention(module, q_slice, ctx_slice)
            if dense_attn is not None:
                results.append(dense_attn)
        if not results:
            return None
        return torch.cat(results, dim=0)

    def _scaled_attention(self, q, k):
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        if self.reduce_heads:
            weights = weights.mean(dim=1)
        return weights

    # ------------------------------------------------------------------ Storage
    def _store_attention(
        self, stage: str, layer_idx: int, attn: torch.Tensor, latent_name: Optional[str] = None
    ) -> None:
        # 只保留 cond 分支，避免 CFG 的 uncond / pm 噪声
        if getattr(self, "_current_branch", "cond") not in ("cond", "unknown"):
            return
        attn_cpu = attn.detach().to(torch.float32).cpu()
        step = self._current_step.get(stage, -1)
        view = self._current_view
        counter_key = (stage, layer_idx, step, view)
        idx = self._event_counter[counter_key]
        self._event_counter[counter_key] += 1

        stage_dir = self.save_dir / stage / f"layer_{layer_idx:02d}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        filename = stage_dir / f"step{step:03d}_view{view:02d}_idx{idx:03d}.pt"

        payload = {
            "stage": stage,
            "layer": layer_idx,
            "step": step,
            "view": view,
            "branch": getattr(self, "_current_branch", "unknown"),
            "latent_name": latent_name,
            "timestamp": time.time(),
            "num_views": self._num_views,
            "attention": attn_cpu,
            "reduced_heads": self.reduce_heads,
        }
        
        # 对于 SLAT 阶段，附加空间坐标信息（需要开启 save_coords 参数）
        if stage == "slat" and self.save_coords and self._slat_coords is not None:
            # 验证维度一致性：coords.shape[0] 应该等于 attention 的 L_latent 维度
            L_latent = attn_cpu.shape[1]  # attention shape: [B, L_latent, L_cond]
            coords_count = self._slat_coords.shape[0]
            
            if coords_count == L_latent:
                payload["coords"] = self._slat_coords
                # 只在第一次保存时记录 coords 保存成功
                if not self._coords_saved:
                    logger.info(
                        f"[CrossAttentionLogger] Including coords in SLAT attention files: "
                        f"coords_shape={tuple(self._slat_coords.shape)}, L_latent={L_latent} ✓"
                    )
                    self._coords_saved = True
            else:
                # 维度不匹配，记录警告但不保存 coords
                if not self._coords_saved:
                    logger.warning(
                        f"[CrossAttentionLogger] Coords dimension mismatch! "
                        f"coords_count={coords_count}, L_latent={L_latent}. "
                        f"Coords will NOT be saved. This might indicate coords were not "
                        f"properly downsampled or there's a bug in the pipeline."
                    )
                    self._coords_saved = True  # 避免重复警告
        
        torch.save(payload, filename)
        logger.info(
            f"[CrossAttentionLogger] Saved attention → {filename.name} "
            f"(stage={stage}, layer={layer_idx}, step={step}, view={view}, shape={tuple(attn_cpu.shape)})"
        )

