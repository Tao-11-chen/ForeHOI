"""
Depth Anything 3 (DA3) Runner Script

This script runs DA3 on a folder of images and outputs:
- depth maps
- pointmaps (3D coordinates in camera space)
- camera extrinsics and intrinsics
- visualization files (optional)

The outputs can be used as input to MV-SAM3D for improved 3D reconstruction.

Usage:
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example
    
    # With custom resolution
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --process_res 756
    
    # Without visualization (faster)
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --no_vis
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

# ============================================================================
# Path setup: DA3 should be a sibling directory to MV-SAM3D
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # MV-SAM3D root
DA3_ROOT = PROJECT_ROOT.parent / "Depth-Anything-3"

if not DA3_ROOT.exists():
    raise FileNotFoundError(
        f"Depth-Anything-3 not found at {DA3_ROOT}. "
        f"Please ensure DA3 is installed as a sibling directory to MV-SAM3D:\n"
        f"  parent_dir/\n"
        f"  ├── MV-SAM3D/\n"
        f"  └── Depth-Anything-3/"
    )

sys.path.insert(0, str(DA3_ROOT / "src"))

# Now we can import DA3
from depth_anything_3.api import DepthAnything3


def depth_to_pointmap(
    depth: np.ndarray, 
    intrinsics: np.ndarray,
) -> np.ndarray:
    """
    Convert depth map to pointmap (3D coordinates in camera space).
    
    NOTE: This outputs in STANDARD CAMERA SPACE (same as MoGe raw output):
        - x: right direction
        - y: down direction  
        - z: forward direction (away from camera, positive depth)
    
    SAM3D's compute_pointmap() will apply the PyTorch3D coordinate transform
    internally, so we should NOT do the transform here.
    
    Args:
        depth: (H, W) depth map, values are distances from camera
        intrinsics: (3, 3) camera intrinsic matrix
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    
    Returns:
        pointmap: (H, W, 3) point cloud map, each pixel is (x, y, z) coordinate
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Create pixel coordinate grids
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Unproject to 3D (standard camera space)
    # z is positive (depth values are positive, pointing away from camera)
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    
    pointmap = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    return pointmap


def pointmap_to_sam3d_format(pointmap: np.ndarray) -> np.ndarray:
    """
    Convert pointmap to SAM3D expected format.
    
    Args:
        pointmap: (H, W, 3) pointmap in PyTorch3D coordinates
        
    Returns:
        pointmap_sam3d: (3, H, W) pointmap ready for SAM3D
    """
    # SAM3D expects (3, H, W) format (channel-first)
    return pointmap.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)


def run_da3_inference(
    image_dir: str,
    output_dir: str,
    model_path: Optional[str] = None,
    process_res: int = 504,
    save_visualization: bool = True,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run DA3 on a folder of images.
    
    Args:
        image_dir: Path to folder containing input images
        output_dir: Path to output directory
        model_path: Path to DA3 model checkpoint (default: auto-detect)
        process_res: Processing resolution (default: 504)
        save_visualization: Whether to save GLB and depth visualizations
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Dictionary containing:
            - depth: (N, H, W) depth maps
            - pointmaps: (N, H, W, 3) point cloud maps
            - extrinsics: (N, 3, 4) or (N, 4, 4) camera extrinsics
            - intrinsics: (N, 3, 3) camera intrinsics
            - image_files: List of input image paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect model path if not provided
    if model_path is None:
        # Check common locations
        possible_paths = [
            DA3_ROOT / "checkpoints" / "DA3NESTED-GIANT-LARGE",
            DA3_ROOT / "checkpoints" / "DA3-GIANT-LARGE",
            Path.home() / ".cache" / "huggingface" / "hub" / "models--depth-anything--DA3NESTED-GIANT-LARGE",
        ]
        for p in possible_paths:
            if p.exists():
                model_path = str(p)
                break
        
        if model_path is None:
            # Fall back to HuggingFace model ID
            model_path = "depth-anything/DA3NESTED-GIANT-LARGE"
            print(f"No local model found, will download from HuggingFace: {model_path}")
    
    print(f"Loading DA3 model from: {model_path}")
    model = DepthAnything3.from_pretrained(model_path).to(device)
    
    # Collect images
    image_dir = Path(image_dir)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(ext))
    
    # Sort with natural number ordering (consistent with inference code)
    # This ensures "2.jpg" comes before "10.jpg" for numeric filenames
    def natural_sort_key(path):
        """Sort key that handles numeric filenames correctly."""
        stem = path.stem
        try:
            return (0, int(stem), stem)  # Numeric names first, sorted numerically
        except ValueError:
            return (1, 0, stem)  # Non-numeric names after, sorted alphabetically
    
    image_files = sorted(image_files, key=natural_sort_key)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")
    
    print(f"Found {len(image_files)} images:")
    for f in image_files:
        print(f"  - {f.name}")
    
    # Build export format
    export_format = "mini_npz"
    if save_visualization:
        export_format += "-glb-depth_vis"
    
    # Run inference
    print(f"\nRunning DA3 inference (process_res={process_res})...")
    prediction = model.inference(
        image=[str(f) for f in image_files],
        process_res=process_res,
        export_dir=str(output_path),
        export_format=export_format,
        show_cameras=True,
    )
    
    # Extract results
    depth = prediction.depth           # (N, H, W)
    extrinsics = prediction.extrinsics # (N, 3, 4) or (N, 4, 4)
    intrinsics = prediction.intrinsics # (N, 3, 3)
    
    print(f"\nDA3 Output:")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"  Extrinsics shape: {extrinsics.shape}")
    print(f"  Intrinsics shape: {intrinsics.shape}")
    
    # Convert depth to pointmaps
    # Two formats:
    # 1. pointmaps: (N, H, W, 3) - standard camera space, for visualization
    # 2. pointmaps_sam3d: (N, 3, H, W) - channel-first format for SAM3D input
    # 
    # NOTE: We output in STANDARD CAMERA SPACE (z positive = away from camera)
    # SAM3D's compute_pointmap() applies the PyTorch3D transform internally
    N = depth.shape[0]
    pointmaps = []
    pointmaps_sam3d = []
    for i in range(N):
        # Convert depth to pointmap (standard camera space, no coordinate transform)
        pm = depth_to_pointmap(depth[i], intrinsics[i])
        pointmaps.append(pm)
        pointmaps_sam3d.append(pointmap_to_sam3d_format(pm))
    
    pointmaps = np.stack(pointmaps, axis=0)  # (N, H, W, 3)
    pointmaps_sam3d = np.stack(pointmaps_sam3d, axis=0)  # (N, 3, H, W)
    
    print(f"  Pointmaps shape: {pointmaps.shape} (standard camera space)")
    print(f"  Pointmaps SAM3D shape: {pointmaps_sam3d.shape} (channel-first for SAM3D)")
    print(f"  Z range: [{pointmaps[:, :, :, 2].min():.4f}, {pointmaps[:, :, :, 2].max():.4f}] (should be positive)")
    
    # Save comprehensive output
    output_file = output_path / "da3_output.npz"
    np.savez(
        output_file,
        depth=depth,                          # (N, H, W)
        pointmaps=pointmaps,                  # (N, H, W, 3) - PyTorch3D coords, for visualization
        pointmaps_sam3d=pointmaps_sam3d,      # (N, 3, H, W) - SAM3D format, ready to use
        extrinsics=extrinsics,                # (N, 3, 4) or (N, 4, 4)
        intrinsics=intrinsics,                # (N, 3, 3)
        image_files=np.array([str(f) for f in image_files]),
        process_res=process_res,
    )
    print(f"\nResults saved to: {output_file}")
    print(f"  - depth: {depth.shape}")
    print(f"  - pointmaps: {pointmaps.shape} (PyTorch3D coords, for visualization)")
    print(f"  - pointmaps_sam3d: {pointmaps_sam3d.shape} (SAM3D format, ready to use)")
    print(f"  - extrinsics: {extrinsics.shape}")
    print(f"  - intrinsics: {intrinsics.shape}")
    
    # Print summary of camera poses
    print(f"\nCamera poses (first 3 views):")
    for i in range(min(3, N)):
        ext = extrinsics[i]
        # Extract rotation and translation
        if ext.shape == (4, 4):
            R, t = ext[:3, :3], ext[:3, 3]
        else:
            R, t = ext[:, :3], ext[:, 3]
        print(f"  View {i}: t = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    
    return {
        "depth": depth,
        "pointmaps": pointmaps,
        "pointmaps_sam3d": pointmaps_sam3d,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "image_files": [str(f) for f in image_files],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Depth Anything 3 on a folder of images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example
    
    # Higher resolution
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --process_res 756
    
    # Without visualization (faster)
    python scripts/run_da3.py --image_dir ./data/example/images --output_dir ./da3_outputs/example --no_vis
        """
    )
    
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True,
        help="Path to folder containing input images"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True,
        help="Path to output directory"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to DA3 model checkpoint (default: auto-detect)"
    )
    parser.add_argument(
        "--process_res", 
        type=int, 
        default=504,
        help="Processing resolution (default: 504)"
    )
    parser.add_argument(
        "--no_vis", 
        action="store_true",
        help="Disable visualization output (GLB, depth_vis)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (default: cuda)"
    )
    
    args = parser.parse_args()
    
    run_da3_inference(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        process_res=args.process_res,
        save_visualization=not args.no_vis,
        device=args.device,
    )


if __name__ == "__main__":
    main()

