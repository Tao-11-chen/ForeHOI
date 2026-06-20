"""
Latent-level weighting module for multi-view fusion.

This module provides an extensible architecture for computing per-latent weights
based on various confidence factors (entropy, patch mass, etc.).

Architecture:
    1. ConfidenceFactors: Collect raw confidence factors from attention data
    2. WeightComputer: Combine factors into final weights with softmax normalization
    3. LatentWeightManager: Orchestrate the entire weighting pipeline

Extensibility:
    - Add new confidence factors by extending compute_confidence_factors()
    - The final softmax normalization is done AFTER all factors are combined
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import torch
import numpy as np
from loguru import logger


# ============================================================================
# Condition Layout Definitions
# ============================================================================

SLAT_CONDITION_LAYOUT = {
    "image_cropped": (0, 1374),      # 1 CLS + 4 Register + 1369 Patch
    "mask_cropped": (1374, 2748),
    "image_full": (2748, 4122),
    "mask_full": (4122, 5496),
}

# Within each image region: [CLS(1), Register(4), Patch(1369)]
CLS_OFFSET = 0
REGISTER_START = 1
REGISTER_END = 5
PATCH_START = 5
PATCH_END = 1374  # relative to region start


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ConfidenceFactors:
    """
    Container for various confidence factors per latent per view.
    
    All factors should have shape [num_views, num_latents].
    Higher values = higher confidence.
    """
    # Core entropy-based factor (inverted: low entropy -> high confidence)
    entropy_confidence: Optional[torch.Tensor] = None
    
    # Patch mass factor: how much attention goes to patch tokens
    patch_mass: Optional[torch.Tensor] = None
    
    # Additional factors can be added here
    # e.g., depth_confidence, semantic_confidence, etc.
    
    # Raw entropy values for analysis (not used in weight computation)
    raw_entropy: Optional[torch.Tensor] = None
    
    # Metadata
    num_views: int = 0
    num_latents: int = 0


@dataclass 
class WeightingConfig:
    """Configuration for weight computation."""
    # Weight source selection
    # - "entropy": Use attention entropy only (default, current behavior)
    # - "visibility": Use self-occlusion based visibility (DDA ray tracing, requires DA3 for camera poses)
    # - "mixed": Combine entropy and visibility
    weight_source: str = "entropy"
    
    # Which factors to use (for backward compatibility, keep use_entropy)
    use_entropy: bool = True
    use_patch_mass: bool = False  # can be enabled later
    
    # Entropy-based weighting parameters
    entropy_alpha: float = 30.0  # Gibbs temperature for entropy
    
    # Visibility-based weighting parameters
    visibility_alpha: float = 30.0  # Gibbs temperature for visibility (higher = more contrast)
    
    # Mixed mode parameters
    weight_combine_mode: str = "average"  # "average" or "multiply"
    visibility_weight_ratio: float = 0.5  # For "average" mode: ratio of visibility vs entropy
    
    # Visibility computation callback
    # Signature: callback(downsampled_coords: np.ndarray, num_views: int, object_pose: dict) -> np.ndarray
    # Returns: [num_views, num_latents] visibility matrix (0=occluded, 1=visible)
    # Uses self-occlusion (DDA ray tracing) to determine visibility
    visibility_callback: Optional[Callable] = field(default=None)
    
    # Object pose will be set after Stage 1 (before Stage 2)
    # This allows the callback to access the pose computed in Stage 1
    object_pose: Optional[Dict] = field(default=None)
    
    # Patch mass parameters (when enabled)
    patch_mass_gamma: float = 1.0  # Exponent for patch mass
    
    # Global temperature for final softmax
    final_temperature: float = 1.0
    
    # Minimum weight to prevent complete zeroing
    min_weight: float = 0.01
    
    # Which attention layer to use
    attention_layer: int = 6  # Based on our analysis, layer 6 is best
    
    # Which step to use
    attention_step: int = 0  # Step 0 has highest view differentiation


# ============================================================================
# Core Computation Functions
# ============================================================================

def compute_patch_entropy(
    attention: torch.Tensor,
    region_name: str = "image_cropped",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute per-latent entropy over patch tokens in a specific region.
    
    Args:
        attention: [B, L_latent, L_cond] attention weights
        region_name: Which region to analyze
        normalize: Whether to normalize entropy to [0, 1]
    
    Returns:
        entropy: [L_latent] entropy values
    """
    if region_name not in SLAT_CONDITION_LAYOUT:
        raise ValueError(f"Unknown region: {region_name}")
    
    region_start, region_end = SLAT_CONDITION_LAYOUT[region_name]
    
    # Extract patch tokens only (exclude CLS and Register)
    patch_start = region_start + PATCH_START
    patch_end = region_start + PATCH_END
    
    logger.debug(
        f"[compute_patch_entropy] attention shape: {attention.shape}, "
        f"region: {region_name} [{region_start}:{region_end}], "
        f"patch: [{patch_start}:{patch_end}]"
    )
    
    if patch_end > attention.shape[-1]:
        logger.warning(
            f"[compute_patch_entropy] patch_end ({patch_end}) > attention L_cond ({attention.shape[-1]}), "
            f"adjusting to {region_end}"
        )
        patch_end = region_end
    
    # Get patch attention: [B, L_latent, num_patches]
    patch_attn = attention[:, :, patch_start:patch_end]
    
    logger.debug(
        f"[compute_patch_entropy] patch_attn shape: {patch_attn.shape}, "
        f"min: {patch_attn.min():.6f}, max: {patch_attn.max():.6f}, mean: {patch_attn.mean():.6f}"
    )
    
    # Normalize to sum to 1 over patches
    patch_sum = patch_attn.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    patch_attn_norm = patch_attn / patch_sum
    
    logger.debug(
        f"[compute_patch_entropy] after normalization: "
        f"min: {patch_attn_norm.min():.6f}, max: {patch_attn_norm.max():.6f}"
    )
    
    # Compute entropy: H = -sum(p * log(p))
    log_attn = torch.log(patch_attn_norm + 1e-10)
    entropy = -(patch_attn_norm * log_attn).sum(dim=-1)  # [B, L_latent]
    
    # Average over batch dimension
    entropy = entropy.mean(dim=0)  # [L_latent]
    
    if normalize:
        # Normalize by max possible entropy (uniform distribution)
        num_patches = patch_attn.shape[-1]
        max_entropy = math.log(num_patches)
        entropy = entropy / max_entropy
    
    logger.info(
        f"[compute_patch_entropy] entropy: min={entropy.min():.4f}, max={entropy.max():.4f}, "
        f"mean={entropy.mean():.4f}, std={entropy.std():.4f}"
    )
    
    return entropy


def compute_patch_mass(
    attention: torch.Tensor,
    region_name: str = "image_cropped",
) -> torch.Tensor:
    """
    Compute total attention mass on patch tokens (vs CLS/Register).
    
    Args:
        attention: [B, L_latent, L_cond] attention weights
        region_name: Which region to analyze
    
    Returns:
        mass: [L_latent] patch mass values (0-1)
    """
    if region_name not in SLAT_CONDITION_LAYOUT:
        raise ValueError(f"Unknown region: {region_name}")
    
    region_start, region_end = SLAT_CONDITION_LAYOUT[region_name]
    
    # Get region attention
    region_attn = attention[:, :, region_start:region_end]
    
    # Split into CLS+Register and Patch
    global_attn = region_attn[:, :, :PATCH_START]  # CLS + Register
    patch_attn = region_attn[:, :, PATCH_START:]   # Patches
    
    # Compute mass ratio
    total_mass = region_attn.sum(dim=-1).clamp(min=1e-10)
    patch_mass = patch_attn.sum(dim=-1) / total_mass  # [B, L_latent]
    
    # Average over batch
    return patch_mass.mean(dim=0)  # [L_latent]


def compute_confidence_factors(
    attention: torch.Tensor,
    config: WeightingConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute all confidence factors from attention weights.
    
    This is the main extensibility point - add new factors here.
    
    NOTE: For entropy, we now return the raw entropy value (not exp(-alpha * entropy)).
    The alpha scaling is applied in compute_fusion_weights to avoid numerical issues.
    
    Args:
        attention: [B, L_latent, L_cond] attention weights
        config: Weighting configuration
    
    Returns:
        factors: Dict mapping factor name to [L_latent] tensor
                 For entropy: returns raw entropy (lower = more confident)
    """
    factors = {}
    
    # 1. Entropy-based factor
    # NOTE: We return raw entropy here. The conversion to weights happens in
    # compute_fusion_weights using softmax(-alpha * entropy) directly.
    # This avoids numerical underflow from exp(-alpha * entropy) when alpha is large.
    if config.use_entropy:
        entropy = compute_patch_entropy(attention, "image_cropped", normalize=True)
        # Store raw entropy (lower entropy = higher confidence)
        # The alpha scaling will be applied in compute_fusion_weights
        factors["entropy"] = entropy
        factors["_raw_entropy"] = entropy  # For analysis (same as entropy now)
    
    # 2. Patch mass factor
    if config.use_patch_mass:
        mass = compute_patch_mass(attention, "image_cropped")
        # Apply gamma exponent
        mass_conf = mass ** config.patch_mass_gamma
        factors["patch_mass"] = mass_conf
    
    # === Add new factors here ===
    # Example:
    # if config.use_depth_confidence:
    #     depth_conf = compute_depth_confidence(attention, ...)
    #     factors["depth"] = depth_conf
    
    return factors


def combine_factors_to_confidence(
    factors: Dict[str, torch.Tensor],
    config: WeightingConfig,
) -> torch.Tensor:
    """
    Combine multiple factors into a single value for weight computation.
    
    For entropy-only mode (current default), this simply returns the entropy.
    
    When multiple factors are enabled, they are combined additively in log-space,
    which is equivalent to multiplying confidence scores.
    
    Args:
        factors: Dict of factor name -> [L_latent] tensor
                 For "entropy": raw entropy value (lower = more confident)
                 For other factors: confidence score (higher = more confident)
        config: Weighting configuration
    
    Returns:
        combined: [L_latent] combined value
                  If only entropy: returns entropy directly
                  If multiple factors: returns combined score
    """
    # Filter out analysis-only factors (those starting with _)
    active_factors = {k: v for k, v in factors.items() if not k.startswith("_")}
    
    if not active_factors:
        raise ValueError("No active confidence factors!")
    
    # If only entropy is used, return it directly
    # (the alpha scaling happens in compute_fusion_weights)
    if len(active_factors) == 1 and "entropy" in active_factors:
        return active_factors["entropy"]
    
    # For multiple factors, we need to combine them
    # Currently: multiply non-entropy factors, then... (TODO: handle mixed factors)
    # For now, just return the first factor if multiple exist
    combined = None
    for name, factor in active_factors.items():
        if combined is None:
            combined = factor.clone()
        else:
            # Note: this multiplication only makes sense if all factors are confidence scores
            # If entropy is mixed with other factors, we need a different strategy
            combined = combined * factor
    
    return combined


def compute_fusion_weights(
    confidences_per_view: Dict[int, torch.Tensor],
    config: WeightingConfig,
) -> Dict[int, torch.Tensor]:
    """
    Convert per-view entropy values to normalized fusion weights.
    
    This applies softmax(-alpha * entropy) directly to avoid numerical issues.
    Lower entropy -> higher weight.
    
    Args:
        confidences_per_view: Dict mapping view_idx -> [L_latent] entropy values
                              (NOTE: despite the name, these are now entropy values,
                               not confidence scores)
        config: Weighting configuration
    
    Returns:
        weights: Dict mapping view_idx -> [L_latent] normalized weights (sum to 1)
    """
    views = sorted(confidences_per_view.keys())
    num_views = len(views)
    
    if num_views == 0:
        return {}
    
    if num_views == 1:
        # Single view: weight = 1
        v = views[0]
        return {v: torch.ones_like(confidences_per_view[v])}
    
    # Stack entropy values: [num_views, num_latents]
    entropy_stack = torch.stack([confidences_per_view[v] for v in views], dim=0)
    
    # Apply softmax directly on -alpha * entropy
    # This is mathematically equivalent to:
    #   confidence = exp(-alpha * entropy)
    #   weights = softmax(log(confidence))
    # But avoids numerical underflow when alpha is large and entropy is high
    #
    # Lower entropy -> higher -alpha * entropy -> higher softmax weight
    # 
    # The temperature parameter controls the sharpness:
    #   - temperature < 1: sharper (winner-take-all)
    #   - temperature > 1: smoother (more uniform)
    logits = -config.entropy_alpha * entropy_stack / config.final_temperature
    weights = torch.softmax(logits, dim=0)
    
    # Apply minimum weight constraint
    if config.min_weight > 0:
        weights = weights.clamp(min=config.min_weight)
        # Re-normalize
        weights = weights / weights.sum(dim=0, keepdim=True)
    
    return {v: weights[i] for i, v in enumerate(views)}


# ============================================================================
# Manager Class
# ============================================================================

class LatentWeightManager:
    """
    Manager for computing and applying latent-level fusion weights.
    
    Supports three weight sources:
    1. "entropy": Use attention entropy (low entropy = more confident)
    2. "visibility": Use latent visibility from pointmaps (visible = more confident)
    3. "mixed": Combine both sources
    
    Usage:
        manager = LatentWeightManager(config)
        
        # During inference, collect attention for each view
        for view_idx in range(num_views):
            attention = get_attention_for_view(view_idx)
            manager.add_view_attention(view_idx, attention)
        
        # For visibility mode, set visibility matrix
        if config.weight_source in ["visibility", "mixed"]:
            manager.set_visibility_matrix(visibility_matrix)  # [num_views, num_latents]
        
        # Set downsample mapping (if available)
        manager.set_downsample_mapping(idx, original_coords, downsampled_coords)
        
        # Compute weights
        weights = manager.compute_weights()  # Returns downsampled weights
        expanded_weights = manager.get_expanded_weights()  # Returns original-dim weights
        
        # Apply weights during fusion
        fused = manager.apply_weights(predictions_per_view, expanded_weights)
    """
    
    def __init__(self, config: Optional[WeightingConfig] = None):
        self.config = config or WeightingConfig()
        self._view_attentions: Dict[int, torch.Tensor] = {}
        self._view_confidences: Dict[int, torch.Tensor] = {}  # entropy-based
        self._view_visibilities: Dict[int, torch.Tensor] = {}  # visibility-based
        self._computed_weights: Optional[Dict[int, torch.Tensor]] = None
        self._expanded_weights: Optional[Dict[int, torch.Tensor]] = None
        self._analysis_data: Dict = {}
        
        # Downsample mapping info
        self._downsample_idx: Optional[torch.Tensor] = None  # [L_original]
        self._original_coords: Optional[torch.Tensor] = None  # [L_original, 4]
        self._downsampled_coords: Optional[torch.Tensor] = None  # [L_downsampled, 4]
    
    def reset(self):
        """Reset for new inference."""
        self._view_attentions.clear()
        self._view_confidences.clear()
        self._view_visibilities.clear()
        self._computed_weights = None
        self._expanded_weights = None
        self._analysis_data.clear()
        self._downsample_idx = None
        self._original_coords = None
        self._downsampled_coords = None
    
    def set_downsample_mapping(
        self,
        idx: Optional[torch.Tensor],
        original_coords: Optional[torch.Tensor] = None,
        downsampled_coords: Optional[torch.Tensor] = None,
    ):
        """
        Set the downsample mapping from original to downsampled latent.
        
        This mapping is used to expand weights from downsampled dimension
        (where attention is computed) to original dimension (where fusion happens).
        
        Args:
            idx: [L_original] tensor, idx[i] = j means original point i maps to downsampled point j
            original_coords: [L_original, 4] original coordinates
            downsampled_coords: [L_downsampled, 4] downsampled coordinates
        """
        self._downsample_idx = idx
        self._original_coords = original_coords
        self._downsampled_coords = downsampled_coords
        
        if idx is not None:
            L_original = idx.shape[0]
            L_downsampled = idx.max().item() + 1
            logger.info(
                f"[LatentWeightManager] Downsample mapping set: "
                f"{L_original} original -> {L_downsampled} downsampled"
            )
    
    def set_visibility_matrix(
        self,
        visibility_matrix: torch.Tensor,
    ):
        """
        Set visibility matrix for visibility-based weighting.
        
        Args:
            visibility_matrix: [num_views, num_latents] tensor
                              Values in [0, 1] where 1 = visible, 0 = not visible
                              Must be computed on downsampled_coords (same dimension as attention)
        """
        num_views, num_latents = visibility_matrix.shape
        
        for view_idx in range(num_views):
            self._view_visibilities[view_idx] = visibility_matrix[view_idx].detach()
        
        # Log statistics
        logger.info(f"[LatentWeightManager] Visibility matrix set: {num_views} views x {num_latents} latents")
        for view_idx in range(num_views):
            v = visibility_matrix[view_idx]
            visible_count = (v > 0.5).sum().item()
            logger.info(
                f"  View {view_idx}: visible={visible_count}/{num_latents} ({100*visible_count/num_latents:.1f}%), "
                f"mean={v.mean():.4f}"
            )
    
    def add_view_attention(
        self, 
        view_idx: int, 
        attention: torch.Tensor,
        step: int = 0,
    ):
        """
        Add attention weights for a view.
        
        Args:
            view_idx: View index
            attention: [B, L_latent, L_cond] attention weights
            step: Diffusion step (default 0, which has best differentiation)
        """
        # Only use specified step
        if step != self.config.attention_step:
            logger.debug(f"[LatentWeightManager] Skipping step {step} (target: {self.config.attention_step})")
            return
        
        logger.info(
            f"[LatentWeightManager] add_view_attention: view={view_idx}, step={step}, "
            f"attention shape={attention.shape}, min={attention.min():.6f}, max={attention.max():.6f}"
        )
        
        self._view_attentions[view_idx] = attention.detach()
        
        # Compute confidence factors immediately
        factors = compute_confidence_factors(attention, self.config)
        confidence = combine_factors_to_confidence(factors, self.config)
        self._view_confidences[view_idx] = confidence
        
        logger.info(
            f"[LatentWeightManager] view {view_idx}: confidence (entropy) "
            f"min={confidence.min():.4f}, max={confidence.max():.4f}, mean={confidence.mean():.4f}"
        )
        
        # Store analysis data
        if "_raw_entropy" in factors:
            if "entropy_per_view" not in self._analysis_data:
                self._analysis_data["entropy_per_view"] = {}
            self._analysis_data["entropy_per_view"][view_idx] = factors["_raw_entropy"]
    
    def compute_weights(self) -> Dict[int, torch.Tensor]:
        """
        Compute final fusion weights based on weight_source config.
        
        Weight sources:
        - "entropy": w_v = softmax(-alpha_e * entropy_v) over views
        - "visibility": w_v = softmax(alpha_v * visibility_v) over views
        - "mixed": combine entropy and visibility weights
        
        Returns:
            weights: Dict mapping view_idx -> [L_latent] weights
        """
        weight_source = self.config.weight_source
        
        # Entropy-only mode
        if weight_source == "entropy":
            if not self._view_confidences:
                logger.warning("[LatentWeightManager] No view confidences (entropy) collected!")
                return {}
            
            self._computed_weights = self._compute_entropy_weights()
            
        # Visibility-only mode (pointmap-based)
        # Visibility mode (uses self-occlusion / DDA ray tracing)
        elif weight_source == "visibility":
            if not self._view_visibilities:
                logger.warning("[LatentWeightManager] No view visibilities collected!")
                return {}
            
            self._computed_weights = self._compute_visibility_weights()
            
        # Mixed mode (combines entropy and visibility)
        elif weight_source == "mixed":
            if not self._view_confidences:
                logger.warning("[LatentWeightManager] No view confidences (entropy) collected for mixed mode!")
                return {}
            if not self._view_visibilities:
                logger.warning("[LatentWeightManager] No view visibilities collected for mixed mode!")
                return {}
            
            self._computed_weights = self._compute_mixed_weights()
            
        else:
            raise ValueError(f"Unknown weight_source: {weight_source}")
        
        # Log statistics
        self._log_weight_statistics()
        
        return self._computed_weights
    
    def _compute_entropy_weights(self) -> Dict[int, torch.Tensor]:
        """
        Compute weights from entropy using softmax(-alpha * entropy).
        Lower entropy = higher weight.
        """
        views = sorted(self._view_confidences.keys())
        num_views = len(views)
        
        if num_views == 0:
            return {}
        if num_views == 1:
            v = views[0]
            return {v: torch.ones_like(self._view_confidences[v])}
        
        # Stack entropy: [num_views, num_latents]
        entropy_stack = torch.stack([self._view_confidences[v] for v in views], dim=0)
        
        # softmax(-alpha * entropy) over views
        logits = -self.config.entropy_alpha * entropy_stack / self.config.final_temperature
        weights = torch.softmax(logits, dim=0)
        
        # Apply min weight
        if self.config.min_weight > 0:
            weights = weights.clamp(min=self.config.min_weight)
            weights = weights / weights.sum(dim=0, keepdim=True)
        
        return {v: weights[i] for i, v in enumerate(views)}
    
    def _compute_visibility_weights(self) -> Dict[int, torch.Tensor]:
        """
        Compute weights from visibility using softmax(alpha * visibility).
        Higher visibility = higher weight.
        """
        views = sorted(self._view_visibilities.keys())
        num_views = len(views)
        
        if num_views == 0:
            return {}
        if num_views == 1:
            v = views[0]
            return {v: torch.ones_like(self._view_visibilities[v])}
        
        # Stack visibility: [num_views, num_latents]
        visibility_stack = torch.stack([self._view_visibilities[v] for v in views], dim=0)
        
        # softmax(alpha * visibility) over views
        # Higher visibility -> higher weight
        logits = self.config.visibility_alpha * visibility_stack / self.config.final_temperature
        weights = torch.softmax(logits, dim=0)
        
        # Apply min weight
        if self.config.min_weight > 0:
            weights = weights.clamp(min=self.config.min_weight)
            weights = weights / weights.sum(dim=0, keepdim=True)
        
        return {v: weights[i] for i, v in enumerate(views)}
    
    def _compute_mixed_weights(self) -> Dict[int, torch.Tensor]:
        """
        Compute mixed weights from entropy and visibility (Scheme A).
        
        Steps:
        1. Compute entropy weights: w_e = softmax(-alpha_e * entropy)
        2. Compute visibility weights: w_v = softmax(alpha_v * visibility)
        3. Combine:
           - "average": w = (1-r) * w_e + r * w_v
           - "multiply": w = w_e * w_v, then normalize
        """
        # Get views that have both entropy and visibility
        entropy_views = set(self._view_confidences.keys())
        visibility_views = set(self._view_visibilities.keys())
        common_views = sorted(entropy_views & visibility_views)
        
        if len(common_views) == 0:
            logger.warning("[LatentWeightManager] No common views between entropy and visibility!")
            return {}
        
        if len(common_views) == 1:
            v = common_views[0]
            return {v: torch.ones_like(self._view_confidences[v])}
        
        # Compute entropy weights
        entropy_weights = self._compute_entropy_weights()
        
        # Compute visibility weights
        visibility_weights = self._compute_visibility_weights()
        
        # Combine weights
        combine_mode = self.config.weight_combine_mode
        ratio = self.config.visibility_weight_ratio
        
        combined_weights = {}
        
        for v in common_views:
            w_e = entropy_weights[v]
            w_v = visibility_weights[v]
            
            if combine_mode == "average":
                # Weighted average: (1-r) * entropy + r * visibility
                w = (1 - ratio) * w_e + ratio * w_v
            elif combine_mode == "multiply":
                # Multiply and re-normalize (done after loop)
                w = w_e * w_v
            else:
                raise ValueError(f"Unknown weight_combine_mode: {combine_mode}")
            
            combined_weights[v] = w
        
        # For "multiply" mode, re-normalize over views
        if combine_mode == "multiply":
            views = sorted(combined_weights.keys())
            weights_stack = torch.stack([combined_weights[v] for v in views], dim=0)
            # Normalize over views
            weights_stack = weights_stack / weights_stack.sum(dim=0, keepdim=True).clamp(min=1e-10)
            combined_weights = {v: weights_stack[i] for i, v in enumerate(views)}
        
        # Apply min weight
        if self.config.min_weight > 0:
            views = sorted(combined_weights.keys())
            weights_stack = torch.stack([combined_weights[v] for v in views], dim=0)
            weights_stack = weights_stack.clamp(min=self.config.min_weight)
            weights_stack = weights_stack / weights_stack.sum(dim=0, keepdim=True)
            combined_weights = {v: weights_stack[i] for i, v in enumerate(views)}
        
        logger.info(
            f"[LatentWeightManager] Mixed weights computed: "
            f"mode={combine_mode}, ratio={ratio:.2f}"
        )
        
        return combined_weights
    
    def get_weights(self) -> Optional[Dict[int, torch.Tensor]]:
        """Get computed weights in downsampled dimension (compute if not done yet)."""
        if self._computed_weights is None:
            return self.compute_weights()
        return self._computed_weights
    
    def get_expanded_weights(self) -> Optional[Dict[int, torch.Tensor]]:
        """
        Get weights expanded to original dimension using downsample mapping.
        
        If no downsample mapping is set, returns the original weights.
        
        Returns:
            weights: Dict mapping view_idx -> [L_original] weights
        """
        if self._computed_weights is None:
            self.compute_weights()
        
        if self._computed_weights is None:
            return None
        
        # If already expanded, return cached
        if self._expanded_weights is not None:
            return self._expanded_weights
        
        # If no downsample mapping, return original weights
        if self._downsample_idx is None:
            logger.warning(
                "[LatentWeightManager] No downsample mapping, returning original weights. "
                "This may cause dimension mismatch!"
            )
            return self._computed_weights
        
        # Expand weights using idx mapping
        self._expanded_weights = {}
        idx = self._downsample_idx
        
        for view_idx, weight in self._computed_weights.items():
            # weight: [L_downsampled]
            # idx: [L_original], values in range [0, L_downsampled-1]
            # expanded_weight[i] = weight[idx[i]]
            expanded = weight[idx]
            self._expanded_weights[view_idx] = expanded
        
        # Log statistics
        L_original = idx.shape[0]
        L_downsampled = list(self._computed_weights.values())[0].shape[0]
        logger.info(
            f"[LatentWeightManager] Expanded weights: {L_downsampled} -> {L_original}"
        )
        
        return self._expanded_weights
    
    def get_original_coords(self) -> Optional[torch.Tensor]:
        """Get original coordinates before downsampling."""
        return self._original_coords
    
    def get_downsampled_coords(self) -> Optional[torch.Tensor]:
        """Get downsampled coordinates where attention is computed."""
        return self._downsampled_coords
    
    def _log_weight_statistics(self):
        """Log weight statistics for debugging."""
        if not self._computed_weights:
            return
        
        views = sorted(self._computed_weights.keys())
        weights_stack = torch.stack([self._computed_weights[v] for v in views], dim=0)
        
        logger.info(f"[LatentWeightManager] Weight statistics ({len(views)} views):")
        for i, v in enumerate(views):
            w = self._computed_weights[v]
            logger.info(
                f"  View {v}: mean={w.mean():.4f}, std={w.std():.4f}, "
                f"min={w.min():.4f}, max={w.max():.4f}"
            )
        
        # Cross-view statistics
        weight_std = weights_stack.std(dim=0)
        logger.info(
            f"  Cross-view std: mean={weight_std.mean():.4f}, max={weight_std.max():.4f}"
        )
    
    def get_analysis_data(self) -> Dict:
        """Get analysis data for visualization."""
        return {
            "config": self.config,
            "weights": self._computed_weights,
            "expanded_weights": self._expanded_weights,
            "entropy_per_view": self._analysis_data.get("entropy_per_view", {}),
            "confidences": self._view_confidences,
            "visibilities": self._view_visibilities,
            "downsample_idx": self._downsample_idx,
            "original_coords": self._original_coords,
            "downsampled_coords": self._downsampled_coords,
        }


# ============================================================================
# Fusion Functions
# ============================================================================

def weighted_fusion(
    predictions: List[torch.Tensor],
    weights: List[torch.Tensor],
) -> torch.Tensor:
    """
    Perform weighted fusion of predictions.
    
    Args:
        predictions: List of [L_latent, C] or [B, L_latent, C] tensors
        weights: List of [L_latent] weight tensors
    
    Returns:
        fused: Weighted sum of predictions
    """
    if len(predictions) != len(weights):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(weights)} weights")
    
    if not predictions:
        raise ValueError("Empty predictions list")
    
    # Ensure weights are on same device
    device = predictions[0].device
    
    # Stack and apply weights
    fused = None
    for pred, w in zip(predictions, weights):
        w = w.to(device)
        
        # Expand weight to match prediction shape
        # pred: [..., L_latent, C] or [..., L_latent]
        # w: [L_latent]
        if pred.dim() >= 2:
            # Add dimensions to broadcast
            for _ in range(pred.dim() - 1):
                w = w.unsqueeze(-1)
        
        weighted_pred = pred * w
        
        if fused is None:
            fused = weighted_pred
        else:
            fused = fused + weighted_pred
    
    return fused


def weighted_fusion_dict(
    predictions: List[Dict[str, torch.Tensor]],
    weights: List[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Perform weighted fusion on dict of predictions.
    
    Args:
        predictions: List of dicts, each mapping key -> tensor
        weights: List of [L_latent] weight tensors
    
    Returns:
        fused: Dict of weighted sums
    """
    if not predictions:
        return {}
    
    keys = predictions[0].keys()
    fused = {}
    
    for key in keys:
        preds_for_key = [p[key] for p in predictions]
        fused[key] = weighted_fusion(preds_for_key, weights)
    
    return fused

