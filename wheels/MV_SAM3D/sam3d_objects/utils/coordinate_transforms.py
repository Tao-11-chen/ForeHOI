# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Unified Coordinate Transform Utilities for MV-SAM3D

This module provides consistent coordinate transformations across the entire pipeline.
All visualization and computation functions should use these utilities.

Coordinate Systems:
===================

1. SAM3D Canonical Space (Z-up)
   - Range: [-0.5, 0.5]^3
   - Z-axis: up
   - Used in: GLB output, Stage 2 latent coordinates

2. PyTorch3D Camera Space (Y-up)
   - X-axis: left
   - Y-axis: up
   - Z-axis: forward (camera looks toward +Z)
   - Used in: SAM3D pose parameters (scale, rotation, translation)

3. OpenCV Camera Space (Y-down)
   - X-axis: right
   - Y-axis: down
   - Z-axis: forward
   - Used in: DA3 pointmaps and extrinsics

4. glTF Space (Y-up, Z-out)
   - Y-axis: up
   - Z-axis: out (camera looks toward -Z)
   - Used in: GLB file visualization

Transform Chain:
================

SAM3D Canonical (Z-up)
        ↓  z_up_to_y_up
PyTorch3D Canonical (Y-up)
        ↓  apply_pose (scale, rotate, translate)
PyTorch3D Camera Space (Y-up)
        ↓  pytorch3d_to_opencv (-x, -y, z)
OpenCV Camera Space (Y-down)
        ↓  c2w (camera extrinsics)
World Space

"""
import numpy as np
import torch
from pytorch3d.transforms import quaternion_to_matrix
from typing import Dict, Tuple, Optional, Union
from loguru import logger


# =============================================================================
# Rotation Matrices
# =============================================================================

# =============================================================================
# IMPORTANT: SAM3D Canonical Space Format
# =============================================================================
# SAM3D uses PyTorch 3D convolution format: (B, C, D, H, W)
# Where D=Depth, H=Height, W=Width corresponds to spatial axes.
#
# In SAM3D's Z-up canonical space:
# - dim0 (D) = Z axis (height/up direction in Z-up system)
# - dim1 (H) = Y axis (depth/front-back direction)
# - dim2 (W) = X axis (width/left-right direction)
#
# So the format is [Z, Y, X], NOT [X, Y, Z]!
# =============================================================================

# Z-up to Y-up for standard [X, Y, Z] input
# Maps: (x, y, z) -> (x, -z, y)
# This is a 90-degree rotation around the X-axis
Z_UP_TO_Y_UP = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0],
], dtype=np.float32)

# Y-up to Z-up (PyTorch3D to canonical)
# Inverse of Z_UP_TO_Y_UP
Y_UP_TO_Z_UP = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.float32)

# =============================================================================
# Transform: Latent [D, H, W] to Mesh [X, Y, Z] format
# =============================================================================
# From inference logs:
#   Latent: dim0=0.98 (tall), dim1=0.80 (short), dim2=0.98 (tall)
#   Mesh:   dim0=0.99 (tall), dim1=0.99 (tall), dim2=0.78 (short)
#
# Latent format: [D, H, W] where D=Height, H=Depth, W=Width  -> [0.98, 0.80, 0.98]
# Mesh format:   [X, Y, Z] where X=Width, Y=Height, Z=Depth  -> [0.99, 0.99, 0.78]
#
# To align Latent to Mesh format:
#   new_X = old_W (dim2) = Width
#   new_Y = old_D (dim0) = Height  
#   new_Z = old_H (dim1) = Depth
#
# Matrix: [D, H, W] @ M.T = [W, D, H] = [X, Y, Z]
LATENT_TO_MESH = np.array([
    [0, 0, 1],   # new_X = dim2 (Width)
    [1, 0, 0],   # new_Y = dim0 (Height)
    [0, 1, 0],   # new_Z = dim1 (Depth)
], dtype=np.float32)

# PyTorch3D to OpenCV camera space
# Maps: (x, y, z) -> (-x, -y, z)
PYTORCH3D_TO_OPENCV = np.array([
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, 1],
], dtype=np.float32)

# OpenCV to PyTorch3D camera space
# Same as PYTORCH3D_TO_OPENCV (it's its own inverse)
OPENCV_TO_PYTORCH3D = PYTORCH3D_TO_OPENCV.copy()


# =============================================================================
# Core Transform Functions
# =============================================================================

def canonical_to_pytorch3d(coords: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from SAM3D canonical space (Z-up) to PyTorch3D space (Y-up).
    
    Args:
        coords: (N, 3) coordinates in [X, Y, Z] format with Z-up
        
    Returns:
        (N, 3) coordinates in Y-up space
    """
    return coords @ Z_UP_TO_Y_UP.T


def latent_to_mesh_format(coords: np.ndarray) -> np.ndarray:
    """
    Transform Latent [D, H, W] coordinates to Mesh [X, Y, Z] format.
    
    This aligns latent coords with mesh vertices in the same canonical space.
    Both will then be in [Width, Height, Depth] format.
    """
    return coords @ LATENT_TO_MESH.T


def latent_to_canonical_scaled(
    latent_coords,  # Can be np.ndarray or torch.Tensor
    scale,  # Can be np.ndarray, torch.Tensor, float, or list
    reorder_axes: bool = False,  # Whether to reorder axes (for testing)
) -> np.ndarray:
    """
    Transform latent coords to canonical space and apply scale.
    
    This is the SIMPLE approach:
    1. Convert to numpy if needed
    2. Convert voxel indices to [-0.5, 0.5] if needed
    3. (Optional) Reorder axes if latent and mesh have different formats
    4. Apply scale only (no rotation/translation)
    
    Args:
        latent_coords: (N, 3) or (N, 4) latent coordinates
        scale: Scale factor(s)
        reorder_axes: Whether to reorder from [D,H,W] to [W,D,H]
        
    Returns:
        (N, 3) coordinates in scaled canonical space
    """
    import torch
    
    # Convert to numpy if tensor
    if isinstance(latent_coords, torch.Tensor):
        latent_coords = latent_coords.cpu().numpy()
    
    # Handle (N, 4) format
    if latent_coords.shape[1] == 4:
        coords = latent_coords[:, 1:4].copy()
    else:
        coords = latent_coords.copy()
    
    # Convert voxel indices to canonical [-0.5, 0.5]
    if coords.max() > 1.0:
        coords = (coords / 64.0) - 0.5
    
    # Clip to valid range
    coords = np.clip(coords, -0.5, 0.5)
    
    # Optionally reorder axes (for testing if formats differ)
    if reorder_axes:
        coords = latent_to_mesh_format(coords)
    
    # Convert scale to numpy
    if isinstance(scale, torch.Tensor):
        scale = scale.cpu().numpy()
    scale = np.atleast_1d(scale).flatten()
    if len(scale) == 1:
        scale = np.array([scale[0], scale[0], scale[0]])
    
    return coords * scale


def mesh_to_canonical_scaled(
    mesh_vertices,  # Can be np.ndarray or torch.Tensor
    scale,  # Can be np.ndarray, torch.Tensor, float, or list
) -> np.ndarray:
    """
    Transform mesh vertices to scaled canonical space.
    
    Mesh is already in canonical format, just apply scale.
    
    Args:
        mesh_vertices: (N, 3) mesh vertices in canonical space
        scale: Scale factor(s)
        
    Returns:
        (N, 3) coordinates in scaled canonical space
    """
    import torch
    
    # Convert to numpy if tensor
    if isinstance(mesh_vertices, torch.Tensor):
        mesh_vertices = mesh_vertices.cpu().numpy()
    
    coords = mesh_vertices.copy()
    
    # Convert scale to numpy
    if isinstance(scale, torch.Tensor):
        scale = scale.cpu().numpy()
    scale = np.atleast_1d(scale).flatten()
    if len(scale) == 1:
        scale = np.array([scale[0], scale[0], scale[0]])
    
    return coords * scale


def pytorch3d_to_canonical(coords: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from PyTorch3D space (Y-up) to SAM3D canonical space (Z-up).
    
    Args:
        coords: (N, 3) coordinates in Y-up space
        
    Returns:
        (N, 3) coordinates in Z-up space
    """
    return coords @ Y_UP_TO_Z_UP.T


def pytorch3d_to_opencv(coords: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from PyTorch3D camera space to OpenCV camera space.
    
    PyTorch3D: X-left, Y-up, Z-forward
    OpenCV: X-right, Y-down, Z-forward
    
    Args:
        coords: (N, 3) coordinates in PyTorch3D camera space
        
    Returns:
        (N, 3) coordinates in OpenCV camera space
    """
    return coords @ PYTORCH3D_TO_OPENCV.T


def opencv_to_pytorch3d(coords: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from OpenCV camera space to PyTorch3D camera space.
    
    Args:
        coords: (N, 3) coordinates in OpenCV camera space
        
    Returns:
        (N, 3) coordinates in PyTorch3D camera space
    """
    return coords @ OPENCV_TO_PYTORCH3D.T


# =============================================================================
# Pose Application
# =============================================================================

def apply_sam3d_pose(
    canonical_coords: np.ndarray,
    scale: np.ndarray,
    rotation_quat: np.ndarray,
    translation: np.ndarray,
    apply_z_to_y_up: bool = True,
    use_zyx_format: bool = True,  # NEW: SAM3D uses [Z,Y,X] format
    debug: bool = False,
) -> np.ndarray:
    """
    Apply SAM3D pose to canonical coordinates.
    
    This is the standard transform chain for SAM3D:
    1. (Optional) Z-up to Y-up rotation (handles [Z,Y,X] format)
    2. Scale
    3. Rotate (using quaternion)
    4. Translate
    
    Args:
        canonical_coords: (N, 3) coordinates in canonical space [-0.5, 0.5]^3
        scale: (3,) or (1,) scale factors
        rotation_quat: (4,) quaternion [w, x, y, z]
        translation: (3,) translation vector
        apply_z_to_y_up: Whether to apply Z-up to Y-up rotation first (default True)
        use_zyx_format: Whether input is in [Z,Y,X] format (SAM3D default) or [X,Y,Z]
        debug: Whether to print debug info
        
    Returns:
        (N, 3) coordinates in PyTorch3D camera space
    """
    # Ensure correct shapes
    scale = np.atleast_1d(scale).flatten()
    rotation_quat = np.atleast_1d(rotation_quat).flatten()
    translation = np.atleast_1d(translation).flatten()
    
    if len(scale) == 1:
        scale = np.array([scale[0], scale[0], scale[0]])
    
    if debug:
        bbox_input = canonical_coords.max(axis=0) - canonical_coords.min(axis=0)
        logger.info(f"[apply_sam3d_pose] Input bbox [dim0,dim1,dim2]: {bbox_input}")
        logger.info(f"[apply_sam3d_pose] use_zyx_format={use_zyx_format}, apply_z_to_y_up={apply_z_to_y_up}")
    
    # Step 1: Coordinate format conversion (if requested)
    if apply_z_to_y_up:
        if use_zyx_format:
            # Latent format: [D, H, W] = [Height, Depth, Width]
            # Convert to Mesh format [X, Y, Z] = [Width, Height, Depth]
            coords = latent_to_mesh_format(canonical_coords)
            if debug:
                bbox_after = coords.max(axis=0) - coords.min(axis=0)
                logger.info(f"[apply_sam3d_pose] After Latent->Mesh format: bbox={bbox_after}")
        else:
            # Mesh is already in [X, Y, Z] format, no conversion needed
            coords = canonical_coords.copy()
            if debug:
                logger.info(f"[apply_sam3d_pose] Mesh format, no conversion")
    else:
        coords = canonical_coords.copy()
    
    # Step 2: Scale
    coords = coords * scale
    if debug:
        bbox_after_scale = coords.max(axis=0) - coords.min(axis=0)
        logger.info(f"[apply_sam3d_pose] After scale ({scale}): bbox={bbox_after_scale}")
    
    # Step 3: Rotate
    quat_tensor = torch.tensor(rotation_quat, dtype=torch.float32).unsqueeze(0)
    R_obj = quaternion_to_matrix(quat_tensor).squeeze(0).numpy()
    coords = coords @ R_obj.T
    if debug:
        bbox_after_rot = coords.max(axis=0) - coords.min(axis=0)
        logger.info(f"[apply_sam3d_pose] After rotation: bbox={bbox_after_rot}")
    
    # Step 4: Translate
    coords = coords + translation
    if debug:
        bbox_final = coords.max(axis=0) - coords.min(axis=0)
        logger.info(f"[apply_sam3d_pose] Final bbox: {bbox_final}")
    
    return coords


def apply_sam3d_pose_to_mesh_vertices(
    canonical_vertices: np.ndarray,
    object_pose: Dict,
    debug: bool = False,
) -> np.ndarray:
    """
    Apply SAM3D pose to mesh vertices loaded from GLB file.
    
    IMPORTANT: Debug logs indicate mesh vertices are in [Width, Height, Depth] format (0.99, 0.99, 0.78).
    Latent coords are [Height, Depth, Width] (0.98, 0.79, 0.98).
    
    To align them and make them upright:
    1. Convert both to standard Z-up [Width, Depth, Height].
    2. Apply Z-up to Y-up transform.
    3. Apply Pose.
    
    Args:
        canonical_vertices: (N, 3) mesh vertices in canonical space
        object_pose: Dict with 'scale', 'rotation', 'translation'
        debug: Whether to print debug info
        
    Returns:
        (N, 3) vertices in PyTorch3D camera space (View 0 world frame)
    """
    if debug:
        logger.info(f"[Mesh Vertices] Canonical coords shape: {canonical_vertices.shape}")
        logger.info(f"[Mesh Vertices] Canonical coords range: "
                    f"dim0=[{canonical_vertices[:, 0].min():.4f}, {canonical_vertices[:, 0].max():.4f}], "
                    f"dim1=[{canonical_vertices[:, 1].min():.4f}, {canonical_vertices[:, 1].max():.4f}], "
                    f"dim2=[{canonical_vertices[:, 2].min():.4f}, {canonical_vertices[:, 2].max():.4f}]")
        
        # Check which dimension has the largest range in canonical space
        canonical_bbox = np.array([
            canonical_vertices[:, 0].max() - canonical_vertices[:, 0].min(),
            canonical_vertices[:, 1].max() - canonical_vertices[:, 1].min(),
            canonical_vertices[:, 2].max() - canonical_vertices[:, 2].min(),
        ])
        logger.info(f"[Mesh Vertices] Canonical bbox: dim0={canonical_bbox[0]:.4f}, dim1={canonical_bbox[1]:.4f}, dim2={canonical_bbox[2]:.4f}")
    
    scale = np.atleast_1d(object_pose.get('scale', [1, 1, 1])).flatten()[:3]
    rotation_quat = np.atleast_1d(object_pose.get('rotation', [1, 0, 0, 0])).flatten()[:4]
    translation = np.atleast_1d(object_pose.get('translation', [0, 0, 0])).flatten()[:3]
    
    result = apply_sam3d_pose(
        canonical_vertices,
        scale=scale,
        rotation_quat=rotation_quat,
        translation=translation,
        apply_z_to_y_up=True,  # Convert to Y-up using the mesh-specific logic
        use_zyx_format=False,  # Mesh is NOT [Z,Y,X] (Latent format), it is [W,H,D]
        debug=debug,
    )
    
    if debug:
        logger.info(f"[Mesh Vertices] After pose application: "
                    f"X=[{result[:, 0].min():.4f}, {result[:, 0].max():.4f}], "
                    f"Y=[{result[:, 1].min():.4f}, {result[:, 1].max():.4f}], "
                    f"Z=[{result[:, 2].min():.4f}, {result[:, 2].max():.4f}]")
    
    return result


def apply_sam3d_pose_to_latent_coords(
    latent_coords: np.ndarray,
    object_pose: Dict,
    debug: bool = False,
) -> np.ndarray:
    """
    Apply SAM3D pose to Stage 2 latent coordinates.
    
    IMPORTANT: Latent coords from torch.argwhere are in format [batch, D, H, W]
    The mesh decoder uses the SAME coordinate format - coords[:, 1:] directly.
    
    We must NOT reorder axes - keep [d0, d1, d2] format to match mesh vertices.
    The mesh vertices from FlexiCubes use the same coordinate order.
    
    Args:
        latent_coords: (N, 4) or (N, 3) latent coordinates
                       If (N, 4), format is [batch, d0, d1, d2] (from argwhere)
                       If (N, 3), format is [d0, d1, d2]
        object_pose: Dict with 'scale', 'rotation', 'translation'
        debug: Whether to print debug info
        
    Returns:
        (N, 3) coordinates in PyTorch3D camera space (View 0 world frame)
    """
    # Handle (N, 4) format where first column is batch index
    if latent_coords.shape[1] == 4:
        coords = latent_coords[:, 1:4].copy()  # [d0, d1, d2] - same as mesh
    else:
        coords = latent_coords.copy()  # Already [d0, d1, d2]
    
    if debug:
        logger.info(f"[Latent Coords] Raw coords shape: {coords.shape}")
        logger.info(f"[Latent Coords] Raw coords range: dim0=[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
                    f"dim1=[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
                    f"dim2=[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    
    # Convert voxel indices to canonical space if needed
    # Voxel coords are in [0, 64), canonical space is [-0.5, 0.5]
    if coords.max() > 1.0:
        coords = (coords / 64.0) - 0.5
    
    # Clip to valid range
    coords = np.clip(coords, -0.5, 0.5)
    
    if debug:
        logger.info(f"[Latent Coords] Canonical coords (no reorder, same as mesh): "
                    f"dim0=[{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}], "
                    f"dim1=[{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}], "
                    f"dim2=[{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}]")
        
        # Check which dimension has the largest range
        canonical_bbox = np.array([
            coords[:, 0].max() - coords[:, 0].min(),
            coords[:, 1].max() - coords[:, 1].min(),
            coords[:, 2].max() - coords[:, 2].min(),
        ])
        logger.info(f"[Latent Coords] Canonical bbox: dim0={canonical_bbox[0]:.4f}, dim1={canonical_bbox[1]:.4f}, dim2={canonical_bbox[2]:.4f}")
    
    scale = np.atleast_1d(object_pose.get('scale', [1, 1, 1])).flatten()[:3]
    rotation_quat = np.atleast_1d(object_pose.get('rotation', [1, 0, 0, 0])).flatten()[:4]
    translation = np.atleast_1d(object_pose.get('translation', [0, 0, 0])).flatten()[:3]
    
    # Apply same transform as mesh vertices
    # SAM3D canonical space uses [Z, Y, X] format (DHW from PyTorch 3D conv)
    result = apply_sam3d_pose(
        coords,
        scale=scale,
        rotation_quat=rotation_quat,
        translation=translation,
        apply_z_to_y_up=True,   # Convert to Y-up
        use_zyx_format=True,    # Input is [Z,Y,X] format
        debug=debug,
    )
    
    if debug:
        logger.info(f"[Latent Coords] After pose application: "
                    f"X=[{result[:, 0].min():.4f}, {result[:, 0].max():.4f}], "
                    f"Y=[{result[:, 1].min():.4f}, {result[:, 1].max():.4f}], "
                    f"Z=[{result[:, 2].min():.4f}, {result[:, 2].max():.4f}]")
        
        bbox_size = result.max(axis=0) - result.min(axis=0)
        logger.info(f"[Latent Coords] Bounding box size: X={bbox_size[0]:.4f}, Y={bbox_size[1]:.4f}, Z={bbox_size[2]:.4f}")
    
    return result


# =============================================================================
# Camera Pose Transforms
# =============================================================================

def convert_da3_extrinsics_to_view0_frame(
    da3_extrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert DA3 extrinsics (world-to-camera) to View 0 coordinate frame.
    
    DA3 extrinsics are w2c matrices in DA3's world coordinate system.
    This function converts them to c2w matrices in View 0's camera coordinate system.
    
    Args:
        da3_extrinsics: (N, 3, 4) or (N, 4, 4) w2c matrices
        
    Returns:
        Tuple of:
        - c2w_list: List of (4, 4) c2w matrices in View 0 frame
        - w2c_list: List of (4, 4) w2c matrices in View 0 frame
    """
    num_views = da3_extrinsics.shape[0]
    
    # Get View 0's w2c matrix
    w2c_view0 = da3_extrinsics[0]
    if w2c_view0.shape == (3, 4):
        w2c_view0_44 = np.eye(4)
        w2c_view0_44[:3, :] = w2c_view0
        w2c_view0 = w2c_view0_44
    
    c2w_list = []
    w2c_list = []
    
    for view_idx in range(num_views):
        w2c_i = da3_extrinsics[view_idx]
        if w2c_i.shape == (3, 4):
            w2c_i_44 = np.eye(4)
            w2c_i_44[:3, :] = w2c_i
            w2c_i = w2c_i_44
        
        # Camera i's c2w in DA3 world
        c2w_i_world = np.linalg.inv(w2c_i)
        
        # Transform to View 0 frame:
        # c2w_i_view0 = w2c_view0 @ c2w_i_world
        c2w_i_view0 = w2c_view0 @ c2w_i_world
        w2c_i_view0 = np.linalg.inv(c2w_i_view0)
        
        c2w_list.append(c2w_i_view0)
        w2c_list.append(w2c_i_view0)
    
    return c2w_list, w2c_list


def compute_camera_pose_from_object_poses_v2(
    pose_0: Dict,
    pose_i: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute camera i's pose relative to camera 0 from object poses.
    
    Mathematical derivation:
    -----------------------
    Given:
    - M_0: transform from object canonical to camera 0 (pose_0)
    - M_i: transform from object canonical to camera i (pose_i)
    
    We want: M_c2w_i = camera i to camera 0 (world) transform
    
    M_c2w_i = M_0 @ inv(M_i)
    
    Expanded:
    - R_c2w = R_0 @ R_i^T
    - T_c2w = T_0 - R_0 @ R_i^T @ T_i
    
    IMPORTANT: The quaternions need to be in the SAME coordinate system.
    SAM3D's rotation is defined in Y-up space (after Z-up to Y-up transform).
    
    Args:
        pose_0: View 0's pose {'rotation': [w,x,y,z], 'translation': [x,y,z]}
        pose_i: View i's pose {'rotation': [w,x,y,z], 'translation': [x,y,z]}
        
    Returns:
        Tuple of:
        - c2w: (4, 4) camera-to-world matrix
        - w2c: (4, 4) world-to-camera matrix
    """
    from scipy.spatial.transform import Rotation
    
    # Extract pose parameters
    T_0 = np.atleast_1d(pose_0['translation']).flatten()[:3]
    quat_0 = np.atleast_1d(pose_0['rotation']).flatten()[:4]  # wxyz
    
    T_i = np.atleast_1d(pose_i['translation']).flatten()[:3]
    quat_i = np.atleast_1d(pose_i['rotation']).flatten()[:4]  # wxyz
    
    # Convert quaternion from wxyz to xyzw for scipy
    quat_0_scipy = np.array([quat_0[1], quat_0[2], quat_0[3], quat_0[0]])
    quat_i_scipy = np.array([quat_i[1], quat_i[2], quat_i[3], quat_i[0]])
    
    # Get rotation matrices
    R_0 = Rotation.from_quat(quat_0_scipy).as_matrix()
    R_i = Rotation.from_quat(quat_i_scipy).as_matrix()
    
    # Compute camera-to-world transform
    R_c2w = R_0 @ R_i.T
    T_c2w = T_0 - R_c2w @ T_i
    
    # Build 4x4 matrices
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = T_c2w
    
    w2c = np.linalg.inv(c2w)
    
    return c2w, w2c


# =============================================================================
# Logging and Debug
# =============================================================================

def log_coordinate_system_info():
    """Log information about coordinate systems for debugging."""
    logger.info("=" * 60)
    logger.info("Coordinate System Information")
    logger.info("=" * 60)
    logger.info("SAM3D Canonical: Z-up, range [-0.5, 0.5]^3")
    logger.info("PyTorch3D Camera: X-left, Y-up, Z-forward")
    logger.info("OpenCV Camera: X-right, Y-down, Z-forward")
    logger.info("")
    logger.info("Transform chain: Canonical -> Z-to-Y-up -> scale -> rotate -> translate")
    logger.info("=" * 60)


def verify_pose_transform(
    canonical_coords: np.ndarray,
    object_pose: Dict,
    expected_center: Optional[np.ndarray] = None,
):
    """
    Verify that pose transform is applied correctly by checking the result center.
    
    Args:
        canonical_coords: Original canonical coordinates
        object_pose: Pose to apply
        expected_center: Expected center after transform (for verification)
    """
    transformed = apply_sam3d_pose_to_latent_coords(canonical_coords, object_pose)
    
    center = transformed.mean(axis=0)
    logger.info(f"[Verify Pose] Transformed center: {center}")
    
    if expected_center is not None:
        error = np.linalg.norm(center - expected_center)
        logger.info(f"[Verify Pose] Center error: {error:.6f}")
        if error > 0.1:
            logger.warning(f"[Verify Pose] Large center error detected!")
    
    return transformed

