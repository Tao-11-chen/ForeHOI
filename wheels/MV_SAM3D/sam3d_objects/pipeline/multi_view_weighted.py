"""
Multi-view weighted fusion utilities.

This module provides weighted multidiffusion fusion based on attention entropy.
It extends the basic multidiffusion to support per-latent weighting.

Key Design (Two-Pass):
    1. Warmup Pass: Run step 0 with simple averaging to collect attention
    2. Compute weights from attention entropy
    3. Main Pass: Run full generation from step 0 with weighted fusion
    
This ensures ALL steps benefit from weighted fusion.
"""
from contextlib import contextmanager
from typing import Dict, List, Literal, Optional
import math
import torch
from loguru import logger

from sam3d_objects.utils.latent_weighting import (
    LatentWeightManager,
    WeightingConfig,
)


# ============================================================================
# Attention Collector for Warmup Pass
# ============================================================================

class AttentionCollector:
    """
    Collects attention weights during warmup pass for weight computation.
    
    This is used to collect attention in memory (not files) during the warmup pass,
    so we can compute weights before the main pass.
    
    Also collects the idx mapping from SparseDownsample to expand weights
    from downsampled dimension (e.g., 4369) to original dimension (e.g., 21411).
    
    IMPORTANT: Due to CFG (Classifier-Free Guidance), the cross-attention is called
    twice per view: once for cond branch and once for uncond branch. We only want
    the cond branch attention (which has meaningful attention patterns), not the
    uncond branch (which has uniform attention due to zeroed conditions).
    """
    
    def __init__(self, num_views: int, target_layer: int = 6):
        self.num_views = num_views
        self.target_layer = target_layer
        self._attentions: Dict[int, torch.Tensor] = {}
        self._current_view: int = 0
        # Track which views have already been collected (to skip uncond branch)
        self._collected_views: set = set()
        # idx mapping: maps original points to downsampled points
        # idx[i] = j means original point i maps to downsampled point j
        self._downsample_idx: Optional[torch.Tensor] = None
        # Original coords before downsampling
        self._original_coords: Optional[torch.Tensor] = None
        # Downsampled coords (where attention is computed)
        self._downsampled_coords: Optional[torch.Tensor] = None
    
    def set_view(self, view_idx: int):
        """Set current view being processed."""
        self._current_view = view_idx
    
    def collect(
        self, 
        layer_idx: int, 
        attention: torch.Tensor,
        query_sparse=None,
    ):
        """
        Collect attention for the current view.
        
        Only collects the FIRST attention for each view (cond branch).
        Skips subsequent collections for the same view (uncond branch).
        
        Args:
            layer_idx: Layer index
            attention: [B, L_latent, L_cond] attention weights
            query_sparse: SparseTensor containing spatial cache with idx mapping
        """
        if layer_idx != self.target_layer:
            return
        
        # Skip if already collected for this view (this is the uncond branch)
        if self._current_view in self._collected_views:
            logger.debug(f"[AttentionCollector] Skipping uncond branch for view {self._current_view}")
            return
        
        # Store attention (cond branch)
        self._attentions[self._current_view] = attention.detach().cpu().clone()
        self._collected_views.add(self._current_view)
        logger.info(f"[AttentionCollector] Collected COND attention for view {self._current_view}, shape={attention.shape}, min={attention.min():.4f}, max={attention.max():.4f}")
        
        # Try to extract idx mapping from SparseTensor (only need to do once)
        if self._downsample_idx is None and query_sparse is not None:
            self._extract_downsample_info(query_sparse)
    
    def _extract_downsample_info(self, query_sparse):
        """Extract downsample idx and coords from SparseTensor's spatial cache."""
        from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor
        
        if not isinstance(query_sparse, SparseTensor):
            return
        
        # The downsampled coords are the current coords of query_sparse
        self._downsampled_coords = query_sparse.coords.detach().cpu().clone()
        
        # Try to get idx from spatial cache
        # The key format is "upsample_{factor}_idx" where factor is (2, 2, 2) for 3D
        spatial_cache = query_sparse._spatial_cache
        
        if not spatial_cache:
            logger.warning("[AttentionCollector] No spatial cache found in query_sparse")
            return
        
        # Look for the idx in any scale's cache
        for scale_key, cache in spatial_cache.items():
            # Try common factor formats
            for factor_key in ["upsample_(2, 2, 2)_idx", "upsample_2_idx"]:
                if factor_key in cache:
                    self._downsample_idx = cache[factor_key].detach().cpu().clone()
                    logger.info(f"[AttentionCollector] Found downsample idx: shape={self._downsample_idx.shape}")
                    break
            
            # Also try to get original coords
            for factor_key in ["upsample_(2, 2, 2)_coords", "upsample_2_coords"]:
                if factor_key in cache:
                    self._original_coords = cache[factor_key].detach().cpu().clone()
                    logger.info(f"[AttentionCollector] Found original coords: shape={self._original_coords.shape}")
                    break
            
            if self._downsample_idx is not None:
                break
        
        if self._downsample_idx is not None and self._original_coords is not None:
            logger.info(
                f"[AttentionCollector] Downsample mapping: "
                f"original {self._original_coords.shape[0]} -> downsampled {self._downsampled_coords.shape[0]}"
            )
    
    def get_attentions(self) -> Dict[int, torch.Tensor]:
        """Get all collected attentions."""
        return self._attentions
    
    def get_downsample_idx(self) -> Optional[torch.Tensor]:
        """Get the downsample idx mapping."""
        return self._downsample_idx
    
    def get_original_coords(self) -> Optional[torch.Tensor]:
        """Get the original coords before downsampling."""
        return self._original_coords
    
    def get_downsampled_coords(self) -> Optional[torch.Tensor]:
        """Get the downsampled coords where attention is computed."""
        return self._downsampled_coords
    
    def reset(self):
        """Reset collected data."""
        self._attentions.clear()
        self._collected_views.clear()
        self._downsample_idx = None
        self._original_coords = None
        self._downsampled_coords = None


# ============================================================================
# SS (Stage 1) Attention Collector - for Dense Latent (4096 voxels)
# ============================================================================

class SSAttentionCollector:
    """
    Collects attention weights during SS (Stage 1) warmup pass for weight computation.
    
    Unlike SLAT, SS uses dense latent (4096 voxels), so no downsample mapping is needed.
    This collector specifically targets the 'shape' latent in MM-DiT architecture.
    
    Strategy: Keep the LAST step's attention (closer to t=1, more stable patterns).
    For each step, we collect cond branch attention and overwrite previous step's data.
    
    Attention format: [bs, 4096, num_cond_tokens]
    """
    
    def __init__(self, num_views: int, target_layer: int = 9):
        self.num_views = num_views
        self.target_layer = target_layer
        self._attentions: Dict[int, torch.Tensor] = {}
        self._current_view: int = 0
        self._current_step: int = 0
        # Track which views have been collected in THIS step (to skip uncond branch)
        self._step_collected_views: set = set()
    
    def set_view(self, view_idx: int):
        """Set current view being processed."""
        self._current_view = view_idx
    
    def new_step(self):
        """Called at the start of each new step to reset per-step tracking."""
        self._current_step += 1
        self._step_collected_views.clear()
    
    def collect(self, layer_idx: int, attention: torch.Tensor):
        """
        Collect attention for the current view.
        
        Only collects cond branch (first call for each view in each step).
        Overwrites previous step's attention to keep only the latest.
        
        Args:
            layer_idx: Layer index
            attention: [B, 4096, L_cond] attention weights
        """
        if layer_idx != self.target_layer:
            return
        
        # Skip if already collected for this view in THIS step (this is the uncond branch)
        if self._current_view in self._step_collected_views:
            return
        
        # Store attention (cond branch), overwriting previous step's data
        self._attentions[self._current_view] = attention.detach().cpu().clone()
        self._step_collected_views.add(self._current_view)
        logger.debug(f"[SSAttentionCollector] Step {self._current_step}: Collected attention for view {self._current_view}")
    
    def get_attentions(self) -> Dict[int, torch.Tensor]:
        """Get all collected attentions."""
        return self._attentions
    
    def reset(self):
        """Reset collected data."""
        self._attentions.clear()
        self._step_collected_views.clear()
        self._current_step = 0


def compute_ss_entropy_weights(
    attentions: Dict[int, torch.Tensor],
    alpha: float = 60.0,
    min_weight: float = 0.001,
    patch_start: int = 1,
    patch_end: int = 1370,
) -> torch.Tensor:
    """
    Compute weights for SS (Stage 1) from attention entropy.
    
    SS stage condition layout (7528 tokens total):
    - image_cropped: [0, 1370) with CLS at 0, patches at [1, 1370)
    - mask_cropped: [1370, 2740)
    - image_full: [2740, 4110)
    - mask_full: [4110, 5480)
    - pointmap: [5480, 6504)
    - rgb_pointmap: [6504, 7528)
    
    By default, uses image_cropped patch tokens (positions 1-1370, excluding CLS at 0).
    
    Args:
        attentions: Dict mapping view_idx to attention tensor [bs, 4096, num_cond_tokens]
        alpha: Gibbs temperature for softmax (higher = more contrast)
        min_weight: Minimum weight to prevent complete zeroing
        patch_start: Start index of patch tokens (default: 1, after CLS)
        patch_end: End index of patch tokens (default: 1370)
    
    Returns:
        weights: [num_views, 4096] tensor of weights, normalized to sum to 1 across views
    """
    views = sorted(attentions.keys())
    num_views = len(views)
    
    if num_views == 0:
        return None
    if num_views == 1:
        # Single view: uniform weight
        return torch.ones(1, attentions[views[0]].shape[1])
    
    # Compute patch entropy for each view
    entropies = []
    for v in views:
        attn = attentions[v]  # [bs, 4096, num_cond_tokens]
        
        # Extract patch tokens (skip CLS, use patch tokens)
        actual_end = min(patch_end, attn.shape[-1])
        patch_attn = attn[:, :, patch_start:actual_end]  # [bs, 4096, num_patches]
        
        # Normalize attention to sum to 1 over patches
        patch_sum = patch_attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        patch_attn_norm = patch_attn / patch_sum
        
        # Compute entropy: H = -sum(p * log(p))
        log_attn = torch.log(patch_attn_norm + 1e-10)
        entropy = -(patch_attn_norm * log_attn).sum(dim=-1)  # [bs, 4096]
        
        # Average over batch dimension
        entropy = entropy.mean(dim=0)  # [4096]
        
        # Normalize by max entropy (optional, for consistency)
        num_patches = patch_attn.shape[-1]
        max_entropy = math.log(num_patches)
        entropy = entropy / max_entropy
        
        entropies.append(entropy)
        logger.info(f"[SS Entropy] View {v}: entropy mean={entropy.mean():.4f}, std={entropy.std():.4f}, min={entropy.min():.4f}, max={entropy.max():.4f}")
    
    # Stack: [num_views, 4096]
    entropy_stack = torch.stack(entropies, dim=0)
    
    # Log cross-view entropy statistics
    entropy_mean_per_view = entropy_stack.mean(dim=1)  # [num_views]
    entropy_std_across_views = entropy_stack.std(dim=0).mean()  # scalar
    logger.info(f"[SS Entropy] Cross-view statistics:")
    logger.info(f"  Per-view mean entropy: {entropy_mean_per_view.tolist()}")
    logger.info(f"  Cross-view std (avg over latents): {entropy_std_across_views:.4f}")
    
    # Compute weights: softmax(-alpha * entropy)
    # Lower entropy = higher weight
    logits = -alpha * entropy_stack
    logger.info(f"[SS Weights] Logits range: min={logits.min():.2f}, max={logits.max():.2f}, spread={logits.max()-logits.min():.2f}")
    
    weights = torch.softmax(logits, dim=0)  # [num_views, 4096]
    
    # Apply min weight
    if min_weight > 0:
        weights = weights.clamp(min=min_weight)
        weights = weights / weights.sum(dim=0, keepdim=True)
    
    # Log per-view mean weights
    logger.info(f"[SS Weights] Computed weights: shape={weights.shape}, "
                f"mean per view: {[f'{weights[i].mean():.4f}' for i in range(num_views)]}")
    
    # Log per-latent best view distribution (which view wins most often)
    best_views = weights.argmax(dim=0)  # [4096]
    view_counts = [(best_views == v).sum().item() for v in range(num_views)]
    logger.info(f"[SS Weights] Best view distribution (per latent): {view_counts}")
    
    # Log weight extremes (how polarized are the weights?)
    max_weights = weights.max(dim=0)[0]  # [4096]
    logger.info(f"[SS Weights] Max weight per latent: mean={max_weights.mean():.4f}, min={max_weights.min():.4f}, max={max_weights.max():.4f}")
    
    return weights


@contextmanager
def inject_ss_generator_with_collector(
    generator,
    num_views: int,
    num_steps: int,
    attention_collector: SSAttentionCollector,
    attention_logger=None,
):
    """
    Inject multi-view support with attention collection for SS (Stage 1).
    
    This is similar to inject_generator_multi_view_with_collector but for SS generator
    which uses MM-DiT architecture and dense latent (4096 voxels).
    
    Args:
        generator: SS generator (ss_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        attention_collector: SSAttentionCollector instance
        attention_logger: Optional CrossAttentionLogger for saving attention to files
    
    Yields:
        None
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules.attention import MultiHeadAttention
    
    original_dynamics = generator._generate_dynamics
    
    # Hook into cross-attention to collect attention
    hooks = []
    cfg_wrapper = getattr(generator, "reverse_fn", None)
    backbone = getattr(cfg_wrapper, "backbone", None)
    
    if backbone is not None:
        blocks = getattr(backbone, "blocks", None)
        if blocks is not None:
            for idx, block in enumerate(blocks):
                if idx != attention_collector.target_layer:
                    continue
                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is None:
                    continue
                
                # MM-DiT: cross_attn is a ModuleDict, we only care about 'shape'
                import torch.nn as nn
                if isinstance(cross_attn, nn.ModuleDict):
                    shape_attn = cross_attn["shape"] if "shape" in cross_attn else None
                    if shape_attn is not None and isinstance(shape_attn, MultiHeadAttention):
                        def make_hook(layer_idx):
                            def hook(module, inputs, outputs):
                                if len(inputs) < 2:
                                    return
                                query, context = inputs[0], inputs[1]
                                
                                # Compute attention weights
                                with torch.no_grad():
                                    attn = _compute_ss_attention_weights(module, query, context)
                                    if attn is not None:
                                        attention_collector.collect(layer_idx, attn)
                            return hook
                        
                        handle = shape_attn.register_forward_hook(make_hook(idx))
                        hooks.append(handle)
                        logger.info(f"[SSAttentionCollector] Hooked layer {idx} for shape attention collection")
                else:
                    # Non-MM-DiT fallback
                    if isinstance(cross_attn, MultiHeadAttention):
                        def make_hook(layer_idx):
                            def hook(module, inputs, outputs):
                                if len(inputs) < 2:
                                    return
                                query, context = inputs[0], inputs[1]
                                with torch.no_grad():
                                    attn = _compute_ss_attention_weights(module, query, context)
                                    if attn is not None:
                                        attention_collector.collect(layer_idx, attn)
                            return hook
                        
                        handle = cross_attn.register_forward_hook(make_hook(idx))
                        hooks.append(handle)
                        logger.info(f"[SSAttentionCollector] Hooked layer {idx} for attention collection")
    
    # Import POSE_KEYS from multi_view_utils
    from sam3d_objects.pipeline.multi_view_utils import POSE_KEYS
    
    def _new_dynamics_with_collection(x_t, t, *args_conditionals, **kwargs_conditionals):
        """Multidiffusion with attention collection for SS."""
        # Mark new step for attention collector (so it keeps only the last step's attention)
        attention_collector.new_step()
        
        cond_idx = 0
        if len(args_conditionals) > 0:
            if isinstance(args_conditionals[0], (int, float)) or (
                isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1
            ):
                cond_idx = 1
        
        if len(args_conditionals) > cond_idx:
            cond_tokens = args_conditionals[cond_idx]
            
            # Parse view conditions
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                view_conditions = [cond_tokens] * num_views
            
            # Collect predictions from all views
            preds = []
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                if cond_idx < len(args_conditionals):
                    new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                else:
                    new_args = args_conditionals + (view_cond,)
                
                # Set current view for attention collection
                attention_collector.set_view(view_idx)
                
                if attention_logger is not None:
                    attention_logger.set_view(view_idx)
                
                pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                preds.append(pred)
            
            # Simple average for warmup pass (with POSE_KEYS handling)
            if isinstance(preds[0], dict):
                fused_pred = {}
                for key in preds[0].keys():
                    stacked = torch.stack([p[key] for p in preds])
                    if key in POSE_KEYS:
                        fused_pred[key] = preds[0][key]
                    else:
                        fused_pred[key] = stacked.mean(dim=0)
                return fused_pred
            elif isinstance(preds[0], (list, tuple)):
                fused_pred = tuple(
                    torch.stack([p[i] for p in preds]).mean(dim=0)
                    for i in range(len(preds[0]))
                )
                return fused_pred
            else:
                return torch.stack(preds).mean(dim=0)
        else:
            return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)
    
    generator._generate_dynamics = _new_dynamics_with_collection
    
    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics
        # Remove hooks
        for handle in hooks:
            handle.remove()


def _compute_ss_attention_weights(module, query, context):
    """
    Compute attention weights for SS (Stage 1).
    
    Args:
        module: MultiHeadAttention module
        query: Query tensor [B, L_latent, C]
        context: Context tensor [B, L_cond, C]
    
    Returns:
        attention: [B, L_latent, L_cond] attention weights
    """
    if query is None or context is None:
        return None
    
    # For dense tensor
    if not torch.is_tensor(query) or not torch.is_tensor(context):
        return None
    
    try:
        B, L_q, C = query.shape
        _, L_c, _ = context.shape
        
        # Get head dim and num_heads
        head_dim = module.head_dim if hasattr(module, 'head_dim') else C // module.num_heads
        num_heads = module.num_heads
        
        # Project to Q
        q = module.to_q(query)  # [B, L_q, C]
        
        # Project to K, V (they are combined in to_kv)
        # to_kv outputs [B, L_c, C * 2] which contains both K and V
        kv = module.to_kv(context)  # [B, L_c, C * 2]
        k, v = kv.chunk(2, dim=-1)  # Each is [B, L_c, C]
        
        # Reshape for multi-head
        q = q.view(B, L_q, num_heads, head_dim).transpose(1, 2)  # [B, H, L_q, D]
        k = k.view(B, L_c, num_heads, head_dim).transpose(1, 2)  # [B, H, L_c, D]
        
        # Compute attention scores
        scale = head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L_q, L_c]
        attn = torch.softmax(attn, dim=-1)
        
        # Average over heads
        attn = attn.mean(dim=1)  # [B, L_q, L_c]
        
        return attn
    except Exception as e:
        logger.warning(f"[SS Attention] Failed to compute: {e}")
        return None


@contextmanager
def inject_generator_multi_view_with_collector(
    generator,
    num_views: int,
    num_steps: int,
    attention_collector: AttentionCollector,
    attention_logger=None,
):
    """
    Inject multi-view support with attention collection.
    
    This is similar to inject_generator_multi_view but also collects attention
    weights into memory for weight computation.
    
    Args:
        generator: SAM 3D Objects generator (slat_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        attention_collector: AttentionCollector instance
        attention_logger: Optional CrossAttentionLogger for saving attention to files
    
    Yields:
        None
    """
    original_dynamics = generator._generate_dynamics
    
    # Also hook into cross-attention to collect attention
    hooks = []
    cfg_wrapper = getattr(generator, "reverse_fn", None)
    backbone = getattr(cfg_wrapper, "backbone", None)
    
    if backbone is not None:
        blocks = getattr(backbone, "blocks", None)
        if blocks is not None:
            for idx, block in enumerate(blocks):
                if idx != attention_collector.target_layer:
                    continue
                cross_attn = getattr(block, "cross_attn", None)
                if cross_attn is None:
                    continue
                
                # Create hook to collect attention and idx mapping
                def make_hook(layer_idx):
                    def hook(module, inputs, outputs):
                        if len(inputs) < 2:
                            return
                        query, context = inputs[0], inputs[1]
                        
                        # Handle multi-view context tensor
                        # context shape could be [num_views, B, L, C]
                        if torch.is_tensor(context) and context.dim() == 4:
                            view_idx = attention_collector._current_view
                            if 0 <= view_idx < context.shape[0]:
                                context = context[view_idx]
                            else:
                                context = context[0]
                        
                        # Compute attention weights
                        with torch.no_grad():
                            attn = _compute_attention_weights(module, query, context)
                            if attn is not None:
                                # Pass query_sparse to extract idx mapping
                                attention_collector.collect(layer_idx, attn, query_sparse=query)
                    return hook
                
                handle = cross_attn.register_forward_hook(make_hook(idx))
                hooks.append(handle)
                logger.info(f"[AttentionCollector] Hooked layer {idx} for attention collection")
    
    def _new_dynamics_with_collection(x_t, t, *args_conditionals, **kwargs_conditionals):
        """Multidiffusion with attention collection."""
        cond_idx = 0
        if len(args_conditionals) > 0:
            if isinstance(args_conditionals[0], (int, float)) or (
                isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1
            ):
                cond_idx = 1
        
        if len(args_conditionals) > cond_idx:
            cond_tokens = args_conditionals[cond_idx]
            
            # Parse view conditions
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                view_conditions = [cond_tokens] * num_views
            
            # Collect predictions from all views
            preds = []
            for view_idx in range(num_views):
                view_cond = view_conditions[view_idx]
                if cond_idx < len(args_conditionals):
                    new_args = args_conditionals[:cond_idx] + (view_cond,) + args_conditionals[cond_idx+1:]
                else:
                    new_args = args_conditionals + (view_cond,)
                
                # Set current view for attention collection
                attention_collector.set_view(view_idx)
                
                if attention_logger is not None:
                    attention_logger.set_view(view_idx)
                
                pred = original_dynamics(x_t, t, *new_args, **kwargs_conditionals)
                preds.append(pred)
            
            # Simple average for warmup pass
            if isinstance(preds[0], dict):
                fused_pred = {}
                for key in preds[0].keys():
                    fused_pred[key] = torch.stack([p[key] for p in preds]).mean(dim=0)
                return fused_pred
            elif isinstance(preds[0], (list, tuple)):
                fused_pred = tuple(
                    torch.stack([p[i] for p in preds]).mean(dim=0)
                    for i in range(len(preds[0]))
                )
                return fused_pred
            else:
                return torch.stack(preds).mean(dim=0)
        else:
            return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)
    
    generator._generate_dynamics = _new_dynamics_with_collection
    
    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics
        # Remove hooks
        for handle in hooks:
            handle.remove()


def _compute_attention_weights(module, query, context):
    """
    Compute attention weights from query and context.
    
    This is a simplified version of the attention computation for collection.
    """
    from sam3d_objects.model.backbone.tdfy_dit.modules.sparse.basic import SparseTensor
    
    if query is None or context is None:
        return None
    
    # Handle SparseTensor
    if isinstance(query, SparseTensor):
        layouts = query.layout
        feats = query.feats
        batch = len(layouts)
        results = []
        
        for batch_idx in range(batch):
            slc = layouts[batch_idx]
            q_slice = feats[slc].unsqueeze(0)  # [1, L, C]
            ctx = context
            if torch.is_tensor(ctx) and ctx.shape[0] > batch:
                ctx_slice = ctx[batch_idx : batch_idx + 1]
            else:
                ctx_slice = ctx
            if ctx_slice is None:
                continue
            if torch.is_tensor(ctx_slice) and ctx_slice.dim() == 2:
                ctx_slice = ctx_slice.unsqueeze(0)
            
            dense_attn = _compute_dense_attention(module, q_slice, ctx_slice)
            if dense_attn is not None:
                results.append(dense_attn)
        
        if not results:
            return None
        return torch.cat(results, dim=0)
    
    elif torch.is_tensor(query):
        return _compute_dense_attention(module, query, context)
    
    return None


def _compute_dense_attention(module, query, context):
    """Compute dense attention weights."""
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
    
    if hasattr(module, 'qk_rms_norm') and module.qk_rms_norm:
        q = module.q_rms_norm(q)
        k = module.k_rms_norm(k)
    
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    
    # Average over heads
    weights = weights.mean(dim=1)
    return weights


def weighted_fusion_sparse(
    predictions: List[torch.Tensor],
    weights: Dict[int, torch.Tensor],
    num_views: int,
) -> torch.Tensor:
    """
    Perform weighted fusion of sparse predictions.
    
    Args:
        predictions: List of [B, L_latent, C] or [L_latent, C] tensors
        weights: Dict mapping view_idx -> [L_latent] weight tensor
                 The weights should be expanded to match prediction's L_latent dimension
        num_views: Number of views
    
    Returns:
        fused: Weighted sum of predictions
    """
    if not predictions:
        raise ValueError("Empty predictions list")
    
    device = predictions[0].device
    pred_shape = predictions[0].shape
    
    # Determine the latent dimension
    # predictions can be [B, L_latent, C] or [L_latent, C]
    if len(pred_shape) == 3:
        L_pred = pred_shape[1]
    elif len(pred_shape) == 2:
        L_pred = pred_shape[0]
    else:
        L_pred = pred_shape[-2] if len(pred_shape) > 1 else pred_shape[0]
    
    # Check if weights match prediction dimension
    sample_weight = list(weights.values())[0] if weights else None
    if sample_weight is not None:
        L_weight = sample_weight.shape[0]
        if L_weight != L_pred:
            logger.warning(
                f"[weighted_fusion_sparse] Dimension mismatch: "
                f"prediction L={L_pred}, weight L={L_weight}. "
                f"This should not happen if weights were properly expanded!"
            )
            # Fallback to simple average
            return torch.stack(predictions).mean(dim=0)
    
    fused = torch.zeros_like(predictions[0])
    
    for view_idx, pred in enumerate(predictions):
        if view_idx in weights:
            w = weights[view_idx].to(device)
            
            # Expand weight to match prediction shape
            # pred: [B, L_latent, C] or [L_latent, C]
            # w: [L_latent]
            if pred.dim() == 3:
                # [B, L_latent, C] -> w needs to be [1, L_latent, 1]
                w = w.unsqueeze(0).unsqueeze(-1)
            elif pred.dim() == 2:
                # [L_latent, C] -> w needs to be [L_latent, 1]
                w = w.unsqueeze(-1)
            
            fused = fused + pred * w
        else:
            # Fallback to equal weight
            fused = fused + pred / num_views
    
    return fused


@contextmanager
def inject_weighted_multi_view_with_precomputed_weights(
    generator,
    num_views: int,
    num_steps: int,
    precomputed_weights: Optional[Dict[int, torch.Tensor]],
    attention_logger=None,
):
    """
    Inject weighted multi-view support with precomputed weights.
    
    This is used in the second pass of the two-pass approach,
    where weights have already been computed from the warmup pass.
    
    Args:
        generator: SAM 3D Objects generator (slat_generator)
        num_views: Number of views
        num_steps: Number of inference steps
        precomputed_weights: Dict mapping view_idx -> [L_latent] weight tensor
        attention_logger: Optional CrossAttentionLogger for saving attention
    
    Yields:
        None
    """
    original_dynamics = generator._generate_dynamics
    
    # Check if we have valid weights
    use_weighted = precomputed_weights is not None and len(precomputed_weights) == num_views
    
    if use_weighted:
        logger.info(f"[WeightedMultidiffusion] Using precomputed weights for {num_views} views")
    else:
        logger.warning("[WeightedMultidiffusion] No valid precomputed weights, using simple average")
    
    def _new_dynamics_with_weights(x_t, t, *args_conditionals, **kwargs_conditionals):
        """Multidiffusion with precomputed weights."""
        cond_idx = 0
        if len(args_conditionals) > 0:
            if isinstance(args_conditionals[0], (int, float)) or (
                isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1
            ):
                cond_idx = 1
        
        if len(args_conditionals) > cond_idx:
            cond_tokens = args_conditionals[cond_idx]
            
            # Log shape once
            if not hasattr(_new_dynamics_with_weights, '_logged_cond_shape'):
                logger.info(f"[WeightedMultidiffusion] args_conditionals length: {len(args_conditionals)}")
                logger.info(f"[WeightedMultidiffusion] cond_idx: {cond_idx}")
                if isinstance(cond_tokens, torch.Tensor):
                    logger.info(f"[WeightedMultidiffusion] Condition tokens shape: {cond_tokens.shape}")
                elif isinstance(cond_tokens, (list, tuple)):
                    logger.info(f"[WeightedMultidiffusion] Condition tokens type: {type(cond_tokens)}, length: {len(cond_tokens)}")
                _new_dynamics_with_weights._logged_cond_shape = True
            
            # Parse view conditions
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                logger.warning(
                    f"Condition tokens shape {cond_tokens.shape if isinstance(cond_tokens, torch.Tensor) else type(cond_tokens)} "
                    "not organized by views, using same condition for all views"
                )
                view_conditions = [cond_tokens] * num_views
            
            # Collect predictions from all views
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
            
            # Log shapes once
            if not hasattr(_new_dynamics_with_weights, '_logged_shape'):
                if isinstance(x_t, dict):
                    logger.info(f"[WeightedMultidiffusion] Latent shape (dict): {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in x_t.items()]}")
                elif isinstance(x_t, (list, tuple)):
                    logger.info(f"[WeightedMultidiffusion] Latent shape (tuple/list): {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in x_t]}")
                else:
                    logger.info(f"[WeightedMultidiffusion] Latent shape: {x_t.shape if isinstance(x_t, torch.Tensor) else type(x_t)}")
                
                if isinstance(preds[0], dict):
                    logger.info(f"[WeightedMultidiffusion] Pred shape (dict): {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in preds[0].items()]}")
                elif isinstance(preds[0], (list, tuple)):
                    logger.info(f"[WeightedMultidiffusion] Pred shape (tuple/list): {[v.shape if isinstance(v, torch.Tensor) else type(v) for v in preds[0]]}")
                else:
                    logger.info(f"[WeightedMultidiffusion] Pred shape: {preds[0].shape if isinstance(preds[0], torch.Tensor) else type(preds[0])}")
                logger.info(f"[WeightedMultidiffusion] Number of views: {num_views}, using_weights: {use_weighted}")
                _new_dynamics_with_weights._logged_shape = True
            
            # Apply fusion
            if use_weighted:
                # Weighted fusion with precomputed weights
                if isinstance(preds[0], dict):
                    fused_pred = {}
                    for key in preds[0].keys():
                        pred_list = [p[key] for p in preds]
                        fused_pred[key] = weighted_fusion_sparse(pred_list, precomputed_weights, num_views)
                    return fused_pred
                elif isinstance(preds[0], (list, tuple)):
                    fused_pred = tuple(
                        weighted_fusion_sparse([p[i] for p in preds], precomputed_weights, num_views)
                        for i in range(len(preds[0]))
                    )
                    return fused_pred
                else:
                    return weighted_fusion_sparse(preds, precomputed_weights, num_views)
            else:
                # Simple average fallback
                if isinstance(preds[0], dict):
                    fused_pred = {}
                    for key in preds[0].keys():
                        fused_pred[key] = torch.stack([p[key] for p in preds]).mean(dim=0)
                    return fused_pred
                elif isinstance(preds[0], (list, tuple)):
                    fused_pred = tuple(
                        torch.stack([p[i] for p in preds]).mean(dim=0)
                        for i in range(len(preds[0]))
                    )
                    return fused_pred
                else:
                    return torch.stack(preds).mean(dim=0)
        else:
            return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)
    
    generator._generate_dynamics = _new_dynamics_with_weights
    
    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics


# Keep the old function for backwards compatibility (deprecated)
@contextmanager
def inject_weighted_multi_view(
    generator,
    num_views: int,
    num_steps: int,
    weight_manager: LatentWeightManager,
    attention_logger=None,
):
    """
    [DEPRECATED] Use inject_weighted_multi_view_with_precomputed_weights instead.
    
    This old approach computes weights after step 0, meaning step 0 uses simple average.
    The new two-pass approach in sample_slat_multi_view_weighted is preferred.
    """
    logger.warning("[WeightedMultidiffusion] Using deprecated inject_weighted_multi_view")
    
    original_dynamics = generator._generate_dynamics
    
    # State for tracking
    state = {
        "current_step": -1,
        "weights_computed": False,
    }
    
    def _new_dynamics_weighted(x_t, t, *args_conditionals, **kwargs_conditionals):
        """Weighted multidiffusion with deferred weight computation."""
        nonlocal state
        
        cond_idx = 0
        if len(args_conditionals) > 0:
            if isinstance(args_conditionals[0], (int, float)) or (
                isinstance(args_conditionals[0], torch.Tensor) and args_conditionals[0].numel() == 1
            ):
                cond_idx = 1
        
        if len(args_conditionals) > cond_idx:
            cond_tokens = args_conditionals[cond_idx]
            
            # Parse view conditions
            if isinstance(cond_tokens, (list, tuple)):
                view_conditions = cond_tokens
            elif isinstance(cond_tokens, torch.Tensor) and cond_tokens.shape[0] == num_views:
                view_conditions = [cond_tokens[i] for i in range(num_views)]
            else:
                view_conditions = [cond_tokens] * num_views
            
            # Collect predictions from all views
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
            
            # Check if we should use weighted fusion
            use_weighted = state["weights_computed"]
            
            if use_weighted:
                weights = weight_manager.get_weights()
                if weights and len(weights) == num_views:
                    if isinstance(preds[0], dict):
                        fused_pred = {}
                        for key in preds[0].keys():
                            pred_list = [p[key] for p in preds]
                            fused_pred[key] = weighted_fusion_sparse(pred_list, weights, num_views)
                        return fused_pred
                    elif isinstance(preds[0], (list, tuple)):
                        fused_pred = tuple(
                            weighted_fusion_sparse([p[i] for p in preds], weights, num_views)
                            for i in range(len(preds[0]))
                        )
                        return fused_pred
                    else:
                        return weighted_fusion_sparse(preds, weights, num_views)
            
            # Fall back to simple average
            if isinstance(preds[0], dict):
                fused_pred = {}
                for key in preds[0].keys():
                    fused_pred[key] = torch.stack([p[key] for p in preds]).mean(dim=0)
                return fused_pred
            elif isinstance(preds[0], (list, tuple)):
                fused_pred = tuple(
                    torch.stack([p[i] for p in preds]).mean(dim=0)
                    for i in range(len(preds[0]))
                )
                return fused_pred
            else:
                return torch.stack(preds).mean(dim=0)
        else:
            return original_dynamics(x_t, t, *args_conditionals, **kwargs_conditionals)
    
    # Wrap generate_iter to track steps and compute weights after step 0
    original_generate_iter = generator.generate_iter
    
    def wrapped_generate_iter(*args, **kwargs):
        nonlocal state
        state["current_step"] = -1
        state["weights_computed"] = False
        
        for step_idx, (t, x_t, extra) in enumerate(original_generate_iter(*args, **kwargs)):
            state["current_step"] = step_idx
            
            yield t, x_t, extra
            
            # After step 0, try to compute weights from collected attention
            if step_idx == 0 and not state["weights_computed"]:
                _try_compute_weights_from_attention(weight_manager, attention_logger, num_views)
                if weight_manager.get_weights():
                    state["weights_computed"] = True
                    logger.info("[WeightedMultidiffusion] Weights computed after step 0")
    
    generator.generate_iter = wrapped_generate_iter
    generator._generate_dynamics = _new_dynamics_weighted
    
    try:
        yield
    finally:
        generator._generate_dynamics = original_dynamics
        generator.generate_iter = original_generate_iter


def _try_compute_weights_from_attention(
    weight_manager: LatentWeightManager,
    attention_logger,
    num_views: int,
):
    """
    Try to compute weights from attention files saved by attention_logger.
    
    This reads the attention files from step 0 and computes entropy-based weights.
    """
    if attention_logger is None:
        logger.warning("[WeightedMultidiffusion] No attention_logger, cannot compute weights")
        return
    
    config = weight_manager.config
    target_layer = config.attention_layer
    target_step = config.attention_step
    
    # Find attention files for step 0
    save_dir = attention_logger.save_dir
    slat_dir = save_dir / "slat"
    
    if not slat_dir.exists():
        logger.warning(f"[WeightedMultidiffusion] SLAT attention dir not found: {slat_dir}")
        return
    
    # Find the target layer directory
    layer_dirs = list(slat_dir.glob(f"layer_{target_layer:02d}"))
    if not layer_dirs:
        # Try to find any layer
        layer_dirs = list(slat_dir.glob("layer_*"))
        if layer_dirs:
            logger.warning(
                f"[WeightedMultidiffusion] Target layer {target_layer} not found, "
                f"using {layer_dirs[0].name}"
            )
    
    if not layer_dirs:
        logger.warning("[WeightedMultidiffusion] No layer directories found")
        return
    
    layer_dir = layer_dirs[0]
    
    # Load attention for each view
    for view_idx in range(num_views):
        # Find file for this view at step 0
        pattern = f"step{target_step:03d}_view{view_idx:02d}_*.pt"
        files = list(layer_dir.glob(pattern))
        
        if not files:
            logger.warning(f"[WeightedMultidiffusion] No attention file for view {view_idx}")
            continue
        
        # Load the first matching file
        attn_data = torch.load(files[0], map_location="cpu")
        attention = attn_data.get("attention")
        
        if attention is None:
            logger.warning(f"[WeightedMultidiffusion] No attention in file {files[0]}")
            continue
        
        # Add to weight manager
        weight_manager.add_view_attention(view_idx, attention, step=target_step)
        logger.debug(f"[WeightedMultidiffusion] Loaded attention for view {view_idx}: {attention.shape}")
    
    # Compute weights
    weight_manager.compute_weights()


class WeightedMultiViewFusion:
    """
    Helper class to manage weighted multi-view fusion during inference.
    
    This class coordinates:
    1. Attention collection during step 0
    2. Weight computation from attention entropy
    3. Weighted fusion application
    """
    
    def __init__(
        self,
        config: Optional[WeightingConfig] = None,
        visualize: bool = False,
        output_dir: Optional[str] = None,
    ):
        self.config = config or WeightingConfig()
        self.weight_manager = LatentWeightManager(self.config)
        self.visualize = visualize
        self.output_dir = output_dir
        
        # State
        self._attention_collected = False
        self._current_step = -1
    
    def reset(self):
        """Reset for new inference."""
        self.weight_manager.reset()
        self._attention_collected = False
        self._current_step = -1
    
    def on_attention(
        self,
        view_idx: int,
        attention: torch.Tensor,
        step: int,
        layer: int,
    ):
        """
        Callback when attention is computed.
        
        Args:
            view_idx: View index
            attention: [B, L_latent, L_cond] attention weights
            step: Current diffusion step
            layer: Layer index
        """
        # Only collect attention at the configured step and layer
        if step != self.config.attention_step:
            return
        if layer != self.config.attention_layer:
            return
        
        self.weight_manager.add_view_attention(view_idx, attention, step)
        logger.debug(f"[WeightedMultiViewFusion] Collected attention for view {view_idx}, step {step}")
    
    def compute_weights(self) -> Dict[int, torch.Tensor]:
        """Compute fusion weights from collected attention."""
        return self.weight_manager.compute_weights()
    
    def get_analysis_data(self) -> Dict:
        """Get analysis data for visualization."""
        return self.weight_manager.get_analysis_data()
    
    def save_visualization(self, coords: Optional[torch.Tensor] = None):
        """
        Save weight visualizations.
        
        Args:
            coords: [L_latent, 4] spatial coordinates (batch, x, y, z)
        """
        if not self.visualize or not self.output_dir:
            return
        
        from pathlib import Path
        import numpy as np
        
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis = self.get_analysis_data()
        weights = analysis.get("weights", {})
        entropy_per_view = analysis.get("entropy_per_view", {})
        
        if not weights:
            logger.warning("[WeightedMultiViewFusion] No weights to visualize")
            return
        
        # Save weights as .pt file
        torch.save({
            "weights": {k: v.cpu() for k, v in weights.items()},
            "entropy": {k: v.cpu() for k, v in entropy_per_view.items()},
            "config": {
                "entropy_alpha": self.config.entropy_alpha,
                "attention_layer": self.config.attention_layer,
                "attention_step": self.config.attention_step,
            },
            "coords": coords.cpu() if coords is not None else None,
        }, output_dir / "fusion_weights.pt")
        
        logger.info(f"[WeightedMultiViewFusion] Saved weights to {output_dir / 'fusion_weights.pt'}")
        
        # Generate visualizations if matplotlib available
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            self._plot_weight_distribution(weights, output_dir)
            self._plot_entropy_distribution(entropy_per_view, output_dir)
            
            if coords is not None:
                self._plot_3d_weights(weights, coords, output_dir)
                self._plot_3d_entropy(entropy_per_view, coords, output_dir)
            
        except ImportError:
            logger.warning("[WeightedMultiViewFusion] matplotlib not available, skipping plots")
    
    def _plot_weight_distribution(self, weights: Dict[int, torch.Tensor], output_dir):
        """Plot weight distribution histogram."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, len(weights), figsize=(4 * len(weights), 4))
        if len(weights) == 1:
            axes = [axes]
        
        for ax, (view_idx, w) in zip(axes, sorted(weights.items())):
            w_np = w.cpu().numpy()
            ax.hist(w_np, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Weight')
            ax.set_ylabel('Count')
            ax.set_title(f'View {view_idx}\nmean={w_np.mean():.4f}, std={w_np.std():.4f}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'weight_distribution.png', dpi=150)
        plt.close()
        logger.info(f"[WeightedMultiViewFusion] Saved weight distribution plot")
    
    def _plot_entropy_distribution(self, entropy_per_view: Dict[int, torch.Tensor], output_dir):
        """Plot entropy distribution histogram."""
        import matplotlib.pyplot as plt
        
        if not entropy_per_view:
            return
        
        fig, axes = plt.subplots(1, len(entropy_per_view), figsize=(4 * len(entropy_per_view), 4))
        if len(entropy_per_view) == 1:
            axes = [axes]
        
        for ax, (view_idx, e) in zip(axes, sorted(entropy_per_view.items())):
            e_np = e.cpu().numpy()
            ax.hist(e_np, bins=50, alpha=0.7, edgecolor='black', color='orange')
            ax.set_xlabel('Entropy')
            ax.set_ylabel('Count')
            ax.set_title(f'View {view_idx}\nmean={e_np.mean():.4f}, std={e_np.std():.4f}')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'entropy_distribution.png', dpi=150)
        plt.close()
        logger.info(f"[WeightedMultiViewFusion] Saved entropy distribution plot")
    
    def _plot_3d_weights(self, weights: Dict[int, torch.Tensor], coords: torch.Tensor, output_dir):
        """Plot 3D weight visualization."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        coords_np = coords.cpu().numpy()
        # coords: [N, 4] where columns are (batch, x, y, z)
        x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
        
        # Normalize coordinates
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        z = (z - z.min()) / (z.max() - z.min() + 1e-6)
        
        for view_idx, w in sorted(weights.items()):
            w_np = w.cpu().numpy()
            
            # Robust normalization
            vmin, vmax = np.percentile(w_np, [2, 98])
            w_norm = np.clip((w_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
            
            # Sort by depth for better visualization
            order = np.argsort(z)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                x[order], y[order], z[order],
                c=w_norm[order],
                cmap='viridis',
                s=1,
                alpha=0.6,
            )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'View {view_idx} Weight')
            
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Weight')
            
            plt.savefig(output_dir / f'weight_3d_view{view_idx:02d}.png', dpi=150)
            plt.close()
        
        logger.info(f"[WeightedMultiViewFusion] Saved 3D weight plots")
    
    def _plot_3d_entropy(self, entropy_per_view: Dict[int, torch.Tensor], coords: torch.Tensor, output_dir):
        """Plot 3D entropy visualization."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        if not entropy_per_view:
            return
        
        coords_np = coords.cpu().numpy()
        x, y, z = coords_np[:, 1], coords_np[:, 2], coords_np[:, 3]
        
        # Normalize coordinates
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        y = (y - y.min()) / (y.max() - y.min() + 1e-6)
        z = (z - z.min()) / (z.max() - z.min() + 1e-6)
        
        for view_idx, e in sorted(entropy_per_view.items()):
            e_np = e.cpu().numpy()
            
            # Robust normalization
            vmin, vmax = np.percentile(e_np, [2, 98])
            e_norm = np.clip((e_np - vmin) / (vmax - vmin + 1e-6), 0, 1)
            
            order = np.argsort(z)
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(
                x[order], y[order], z[order],
                c=e_norm[order],
                cmap='hot',
                s=1,
                alpha=0.6,
            )
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'View {view_idx} Entropy')
            
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('Entropy')
            
            plt.savefig(output_dir / f'entropy_3d_view{view_idx:02d}.png', dpi=150)
            plt.close()
        
        logger.info(f"[WeightedMultiViewFusion] Saved 3D entropy plots")

