# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Multi-view multidiffusion utilities for SAM 3D Objects
Adapted from TRELLIS implementation, adapted for SAM 3D Objects' two-stage structure
"""
from contextlib import contextmanager
from typing import Literal, Optional
import torch
from loguru import logger

# Pose 相关的 key，这些不应该被平均
POSE_KEYS = {
    'translation', 'rotation', 'scale', 'translation_scale',
    '6drotation', '6drotation_normalized',
    'quaternion',
}


@contextmanager
def inject_generator_multi_view(
    generator,
    num_views: int,
    num_steps: int,
    mode: Literal['stochastic', 'multidiffusion'] = 'multidiffusion',
    attention_logger=None,
    shape_weights: Optional[torch.Tensor] = None,
):
    """
    Inject multi-view support into generator.
    
    Args:
        generator: SAM 3D Objects generator (ss_generator or slat_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        mode: 'stochastic' or 'multidiffusion'
        attention_logger: Logger for attention capture
        shape_weights: Optional weights for shape fusion
            - If None: use simple average
            - If [num_views]: use per-view weights (same weight for all latent points)
            - If [num_views, num_latent_points]: use per-view-per-latent weights
    
    Yields:
        None (kept for API compatibility)
        
    Multi-view Iteration Strategy:
    ------------------------------
    - Shape: Weighted average (or simple average if no weights)
    - Pose: Only View 0's velocity is used (other views' pose velocity ignored)
    - Output: shape + View 0's pose
    """
    all_view_states_storage = None
    
    original_dynamics = generator._generate_dynamics
    
    if mode == 'stochastic':
        # Stochastic mode: 每一步随机选择一个视角
        if num_views > num_steps:
            logger.warning(
                f"Warning: number of views ({num_views}) is greater than number of steps ({num_steps}). "
                "This may lead to performance degradation."
            )
        
        cond_indices = (torch.arange(num_steps) % num_views).tolist()
        cond_idx_counter = [0]
        
        def _new_dynamics_stochastic(x_t, t, *args_conditionals, **kwargs_conditionals):
            """Stochastic mode: select one view per time step"""
            cond_idx = cond_indices[cond_idx_counter[0] % len(cond_indices)]
            cond_idx_counter[0] += 1
            
            cond_tokens = args_conditionals[1]
            if isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                cond_i = cond_tokens[cond_idx]
                new_args = (args_conditionals[0], cond_i)
            else:
                new_args = args_conditionals
            
            if attention_logger is not None:
                attention_logger.set_view(cond_idx)
            return original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
        
        generator._generate_dynamics = _new_dynamics_stochastic
        
    elif mode == 'multidiffusion':
        # Multidiffusion mode: 每一步融合所有视角的预测
        dt = 1.0 / num_steps
        
        def _new_dynamics_multidiffusion(x_t, t, *args_conditionals, **kwargs_conditionals):
            """
            Multidiffusion mode: fuse predictions from all views.
            
            Shape: 用平均 velocity 更新
            Pose: 
                - 默认模式: 只用 View 0 的 velocity
                - Per-view 模式: 每个视角用自己的 velocity 更新自己的 pose
            """
            nonlocal all_view_states_storage
            
            # 找到 condition tokens 在 args 中的位置
            cond_idx = 0
            if len(args_conditionals) > 0:
                if isinstance(args_conditionals[0], (int, float)) or \
                   (isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1):
                    cond_idx = 1
            
            if len(args_conditionals) <= cond_idx:
                return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)
            
            cond_tokens = args_conditionals[cond_idx]
            
            # 日志（只打印一次）
            if not hasattr(_new_dynamics_multidiffusion, '_logged_cond_shape'):
                logger.info(f"[Multidiffusion] num_views: {num_views}, cond_idx: {cond_idx}")
                if isinstance(cond_tokens, torch.Tensor):
                    logger.info(f"[Multidiffusion] Condition tokens shape: {cond_tokens.shape}")
                elif isinstance(cond_tokens, (list, tuple)):
                    logger.info(f"[Multidiffusion] Condition tokens: list/tuple, length={len(cond_tokens)}")
                _new_dynamics_multidiffusion._logged_cond_shape = True
            
            # 解析每个视角的 condition
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                logger.warning(f"Condition tokens not organized by views, using same condition for all views")
                view_conditions = [cond_tokens] * num_views
            
            # Fuse predictions from all views
            # Shape: averaged, Pose: View 0 only
            preds = []
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                if cond_idx < len(args_conditionals):
                    new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                else:
                    new_args = args_conditionals + (view_cond,)
                if attention_logger is not None:
                    attention_logger.set_view(view_idx)
                pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                preds.append(pred)
            
            # Log (only once)
            if not hasattr(_new_dynamics_multidiffusion, '_logged_shape'):
                if isinstance(x_t, dict):
                    logger.info(f"[Multidiffusion] x_t keys: {list(x_t.keys())}")
                if isinstance(preds[0], dict):
                    logger.info(f"[Multidiffusion] pred keys: {list(preds[0].keys())}")
                if shape_weights is not None:
                    logger.info(f"[Multidiffusion] Using weighted fusion: weights shape = {shape_weights.shape}")
                else:
                    logger.info(f"[Multidiffusion] Using simple average (no weights)")
                logger.info(f"[Multidiffusion] Default mode: Shape=weighted/avg, Pose=View0")
                _new_dynamics_multidiffusion._logged_shape = True
            
            # Fuse predictions
            if isinstance(preds[0], dict):
                fused_pred = {}
                for key in preds[0].keys():
                    stacked = torch.stack([p[key] for p in preds])  # [num_views, bs, num_latent, dim]
                    if key in POSE_KEYS:
                        # Pose: View 0 only
                        fused_pred[key] = preds[0][key]
                    else:
                        # Shape: weighted average or simple average
                        if shape_weights is not None:
                            # Reshape weights for broadcasting
                            # stacked: [num_views, bs, num_latent, dim]
                            # weights: [num_views] or [num_views, num_latent]
                            w = shape_weights
                            if w.dim() == 1:
                                # [num_views] -> [num_views, 1, 1, 1]
                                w = w.view(-1, 1, 1, 1)
                            elif w.dim() == 2:
                                # [num_views, num_latent] -> [num_views, 1, num_latent, 1]
                                w = w.unsqueeze(1).unsqueeze(-1)
                            w = w.to(stacked.device, stacked.dtype)
                            fused_pred[key] = (stacked * w).sum(dim=0)
                        else:
                            fused_pred[key] = stacked.mean(dim=0)
                return fused_pred
            elif isinstance(preds[0], (list, tuple)):
                # For non-dict outputs, apply simple average (weights not supported)
                fused_pred = tuple(
                    torch.stack([p[i] for p in preds]).mean(dim=0)
                    for i in range(len(preds[0]))
                )
                return fused_pred
            else:
                # For tensor output, apply simple average (weights not supported)
                fused_pred = torch.stack(preds).mean(dim=0)
                return fused_pred
        
        generator._generate_dynamics = _new_dynamics_multidiffusion
        
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    try:
        yield all_view_states_storage
    finally:
        generator._generate_dynamics = original_dynamics
