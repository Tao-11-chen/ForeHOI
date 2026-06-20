# MV-SAM3D Parameter Documentation

This document provides detailed documentation for all command-line parameters of `run_inference_weighted.py`.

## Table of Contents

- [Basic Parameters](#basic-parameters)
- [Inference Parameters](#inference-parameters)
- [Stage 1 (Shape) Weighting Parameters](#stage-1-shape-weighting-parameters)
- [Stage 2 (Texture) Weighting Parameters](#stage-2-texture-weighting-parameters)
- [Visualization Parameters](#visualization-parameters)
- [DA3 Integration Parameters](#da3-integration-parameters)
- [Usage Examples](#usage-examples)

---

## Basic Parameters

### `--input_path` (Required)
- **Type**: `str`
- **Description**: Input data path (directory or file)
- **Example**: `--input_path ./data/example`

### `--mask_prompt`
- **Type**: `str`
- **Default**: `None`
- **Description**: Mask folder name. If the input directory has multiple mask subfolders, specify which one to use.
- **Example**: `--mask_prompt stuffed_toy`

### `--image_names`
- **Type**: `str`
- **Default**: `None`
- **Description**: Specify image names to use (comma-separated). If not specified, all images in the directory will be used.
- **Example**: `--image_names 0,1,2,3,4,5,6,7`
- **Note**: Image numbering starts from 0

### `--model_tag`
- **Type**: `str`
- **Default**: `"hf"`
- **Description**: Model tag, specifies which checkpoint directory to use
- **Example**: `--model_tag hf`

---

## Inference Parameters

### `--seed`
- **Type**: `int`
- **Default**: `42`
- **Description**: Random seed for reproducibility
- **Example**: `--seed 42`

### `--stage1_steps`
- **Type**: `int`
- **Default**: `50`
- **Description**: Number of iteration steps for Stage 1 (shape generation).
- **Example**: `--stage1_steps 50`

### `--stage2_steps`
- **Type**: `int`
- **Default**: `25`
- **Description**: Number of iteration steps for Stage 2 (texture generation).
- **Example**: `--stage2_steps 25`

### `--decode_formats`
- **Type**: `str`
- **Default**: `"gaussian,mesh"`
- **Description**: Decode formats list (comma-separated). Supported formats: `gaussian`, `mesh`
- **Example**: `--decode_formats gaussian,mesh` or `--decode_formats mesh`

---

## Stage 1 (Shape) Weighting Parameters

Stage 1 generates the 3D shape structure. These parameters control entropy-based weighting for shape generation.

### `--no_stage1_weighting`
- **Type**: `flag` (boolean)
- **Default**: `False` (i.e., Stage 1 weighting is **ENABLED** by default)
- **Description**: Disable entropy-based weighting for Stage 1. Uses simple average instead.
- **Note**: Only affects shape, not pose prediction (pose always from View 0).
- **Example**: `--no_stage1_weighting`

### `--stage1_entropy_layer`
- **Type**: `int`
- **Default**: `9`
- **Description**: Attention layer for computing Stage 1 weights. Layer 9 shows highest cross-view entropy differences.
- **Example**: `--stage1_entropy_layer 9`

### `--stage1_entropy_alpha`
- **Type**: `float`
- **Default**: `30.0`
- **Description**: Gibbs temperature for Stage 1 entropy weighting.
  - **Higher values** (`> 30.0`): More selective, sharper weight distribution (aggressive)
  - **Lower values** (`< 30.0`): Smoother, more uniform weights (conservative)
  - Based on analysis: cross-view entropy std ≈ 0.10, so alpha=30 gives logit diff ≈ 3 (moderate)
- **Example**: `--stage1_entropy_alpha 30.0`

---

## Stage 2 (Texture) Weighting Parameters

Stage 2 generates texture/appearance. These parameters control weighting for texture generation.

### `--no_stage2_weighting`
- **Type**: `flag` (boolean)
- **Default**: `False` (i.e., Stage 2 weighting is **ENABLED** by default)
- **Description**: Disable weighting for Stage 2. Uses simple average instead.
- **Example**: `--no_stage2_weighting`

### `--stage2_weight_source`
- **Type**: `str`
- **Default**: `"entropy"`
- **Options**: `"entropy"`, `"visibility"`, `"mixed"`
- **Description**: Source for Stage 2 fusion weights.
  
  #### 1. `entropy` (Default)
  - **Description**: Use attention entropy only
  - **Principle**: Low entropy (focused attention) → High weight
  - **Use case**: General scenarios without DA3 depth information
  
  #### 2. `visibility`
  - **Description**: Use self-occlusion based visibility
  - **Principle**: Visible (not self-occluded) → High weight (binary 0/1)
  - **Method**: DDA ray tracing for self-occlusion detection
  - **Requirement**: **Must provide `--da3_output`**
  
  #### 3. `mixed`
  - **Description**: Combine entropy and visibility
  - **Requirement**: **Must provide `--da3_output`**

- **Example**: `--stage2_weight_source visibility`

### `--stage2_entropy_alpha`
- **Type**: `float`
- **Default**: `30.0`
- **Description**: Gibbs temperature for Stage 2 entropy weighting.
  - **Higher values** (`> 30.0`): More aggressive, sharper weight distribution
  - **Lower values** (`< 30.0`): More conservative, smoother weights
- **Example**: `--stage2_entropy_alpha 30.0`

### `--stage2_visibility_alpha`
- **Type**: `float`
- **Default**: `30.0`
- **Description**: Gibbs temperature for Stage 2 visibility weighting.
  - **Higher values** (`> 30.0`): More aggressive, sharper weight distribution
  - **Lower values** (`< 30.0`): More conservative, smoother weights
- **Only applies when**: `--stage2_weight_source` is `"visibility"` or `"mixed"`
- **Example**: `--stage2_visibility_alpha 30.0`

### `--stage2_attention_layer`
- **Type**: `int`
- **Default**: `6`
- **Description**: Attention layer for computing Stage 2 weights.
- **Example**: `--stage2_attention_layer 6`

### `--stage2_attention_step`
- **Type**: `int`
- **Default**: `0`
- **Description**: Diffusion step for computing Stage 2 weights.
- **Example**: `--stage2_attention_step 0`

### `--stage2_min_weight`
- **Type**: `float`
- **Default**: `0.001`
- **Description**: Minimum weight to prevent complete zeroing.
- **Example**: `--stage2_min_weight 0.001`

### `--self_occlusion_tolerance`
- **Type**: `float`
- **Default**: `4.0`
- **Description**: Tolerance for self-occlusion detection (voxel units). Ignores occluding voxels within this distance.
  - Higher values: More lenient, fewer false occlusions
  - Lower values: More strict
- **Only applies when**: `--stage2_weight_source` is `"visibility"` or `"mixed"`
- **Example**: `--self_occlusion_tolerance 4.0`

### `--stage2_weight_combine_mode`
- **Type**: `str`
- **Default**: `"average"`
- **Options**: `"average"`, `"multiply"`
- **Description**: How to combine entropy and visibility in mixed mode.
- **Only applies when**: `--stage2_weight_source` is `"mixed"`

### `--stage2_visibility_weight_ratio`
- **Type**: `float`
- **Default**: `0.5`
- **Range**: `0.0` - `1.0`
- **Description**: Visibility weight ratio in average mode.
  - `0.0`: Entropy only
  - `0.5`: Equal mix
  - `1.0`: Visibility only
- **Only applies when**: `--stage2_weight_source` is `"mixed"` and `--stage2_weight_combine_mode` is `"average"`

---

## Visualization Parameters

### `--visualize_weights`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Save weight and entropy visualization results.
- **Example**: `--visualize_weights`

### `--compute_latent_visibility`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Visualize latent visibility per view (green=visible, red=occluded).
- **Requirement**: **Must provide `--da3_output`**
- **Example**: `--compute_latent_visibility`

### `--save_attention`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Save all attention weights (for analysis).
- **Example**: `--save_attention`

### `--attention_layers`
- **Type**: `str`
- **Default**: `None`
- **Description**: Specify which layers' attention to save (comma-separated).
- **Example**: `--attention_layers 4,5,6`

### `--save_stage2_init`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Save Stage 2 initial latent state.
- **Example**: `--save_stage2_init`

---

## DA3 Integration Parameters

### `--da3_output`
- **Type**: `str`
- **Default**: `None`
- **Description**: Path to DA3 output npz file (generated by `scripts/run_da3.py`).
  - Required for visibility weighting
  - Required for GLB merge visualization
- **Example**: `--da3_output ./da3_outputs/example/da3_output.npz`

### `--merge_da3_glb`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Merge SAM3D GLB with DA3's scene.glb.
- **Requirement**: **Must provide `--da3_output`**
- **Output**: `result_merged_scene.glb`
- **Example**: `--merge_da3_glb`

### `--overlay_pointmap`
- **Type**: `flag`
- **Default**: `False`
- **Description**: Overlay SAM3D result on View 0 point cloud for pose verification.
- **Output**: `result_overlay.glb`
- **Example**: `--overlay_pointmap`

---

## Usage Examples

### Basic Multi-View Inference (Default: Both Stages Weighted)
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7
```

### Disable All Weighting (Simple Average)
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --no_stage1_weighting \
    --no_stage2_weighting
```

### Only Stage 2 Weighting (Disable Stage 1)
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --no_stage1_weighting
```

### Custom Weighting Parameters
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --stage1_entropy_alpha 80.0 \
    --stage2_entropy_alpha 80.0
```

### Using DA3 Point Clouds
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --da3_output ./da3_outputs/example/da3_output.npz \
    --overlay_pointmap \
    --merge_da3_glb
```

### Using Visibility Weighting for Stage 2
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --da3_output ./da3_outputs/example/da3_output.npz \
    --stage2_weight_source visibility \
    --stage2_visibility_alpha 60.0
```

### Using Mixed Weighting for Stage 2
```bash
python run_inference_weighted.py \
    --input_path ./data/example \
    --mask_prompt stuffed_toy \
    --image_names 0,1,2,3,4,5,6,7 \
    --da3_output ./da3_outputs/example/da3_output.npz \
    --stage2_weight_source mixed \
    --stage2_entropy_alpha 60.0 \
    --stage2_visibility_alpha 60.0 \
    --stage2_visibility_weight_ratio 0.5
```

---

## Output Files

### Basic Output
- `result.glb`: Main 3D reconstruction result
- `result.ply`: Point cloud format output
- `params.npz`: Parameter file (contains pose, scale, etc.)
- `inference.log`: Inference log with experiment configuration

### Overlay Visualization Output
- `result_overlay.glb`: SAM3D result overlaid on View 0 point cloud

### Merged Scene Output
- `result_merged_scene.glb`: SAM3D object merged with DA3 scene

### Weighting Visualization Output (if enabled)
- `weights/`: Contains weight distribution, entropy distribution, 3D weight visualization

---

## Parameter Dependencies

### Required Combinations
1. `--merge_da3_glb` requires `--da3_output`
2. `--stage2_weight_source visibility` or `mixed` requires `--da3_output`
3. `--compute_latent_visibility` requires `--da3_output`

### Recommended Combinations
1. High quality reconstruction: Default settings (both stages weighted)
2. Visibility weighting: `--stage2_weight_source visibility` + `--da3_output`
3. Pose verification: `--overlay_pointmap` + `--da3_output`

---

## FAQ

### Q: What's the difference between Stage 1 and Stage 2?
A:
- **Stage 1 (Shape)**: Generates 3D shape structure. Weighting affects shape quality.
- **Stage 2 (Texture)**: Generates texture/appearance. Weighting affects texture quality.

### Q: How to tune alpha values?
A:
- **Default**: 30.0 (conservative, recommended starting point)
- **Results too uniform/blurry**: Increase to 50.0 or 60.0 (more aggressive)
- **Results have artifacts**: Lower to 20.0 or 15.0 (more conservative)
- **Extreme**: 60.0+ gives winner-take-all behavior (use with caution)

### Q: Which weight source should I use for Stage 2?
A:
- **`entropy` (Default)**: Works without DA3, good for most cases
- **`visibility`**: Better for objects with obvious occlusion, requires DA3
- **`mixed`**: Combines both, requires DA3

---

