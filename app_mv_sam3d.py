"""MV-SAM3D baseline app — MV-SAM3D's OWN stage-1 + stage-2 (run_multi_view),
then DepthAnything3 + FoundationPose for a 6-DoF pose-overlay video.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
# Order matters: MV-SAM3D fork must win `import sam3d_objects`.
sys.path.insert(0, _HERE)                                        # ui_common, pose_est, `wheels` namespace
sys.path.insert(0, os.path.join(_HERE, "wheels/MV_SAM3D"))       # -> sam3d_objects = MV-SAM3D fork

import numpy as np
import torch
import trimesh
import imageio

from wheels.MV_SAM3D.notebook.inference import Inference, render_utils
from wheels.MV_SAM3D.sam3d_objects.utils.latent_weighting import WeightingConfig

import ui_common
from pose_est import PoseEstimator, preprocess_images_crop

# ── config ────────────────────────────────────────────────────────────────────
SAM2_CKPT       = os.path.join(_HERE, "weights/sam2.1_hiera_large.pt")
SAM2_CFG        = "configs/sam2.1/sam2.1_hiera_l.yaml"
MV_PIPELINE_CFG = os.path.join(_HERE, "wheels/MV_SAM3D/checkpoints/hf/pipeline.yaml")
DA3_WEIGHTS     = os.path.join(_HERE, "weights/DA3")
TMP_DIR         = os.path.join(_HERE, "output_wild")
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(TMP_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", TMP_DIR)
ui_common.TMP_DIR = TMP_DIR
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── load models ───────────────────────────────────────────────────────────────
print("Loading SAM2 ...")
ui_common.init_sam2(SAM2_CKPT, SAM2_CFG, DEVICE)
print(f"Loading MV-SAM3D pipeline from {MV_PIPELINE_CFG} ...")
inference = Inference(MV_PIPELINE_CFG, compile=False)
weighting_config = WeightingConfig(
    use_entropy=True, entropy_alpha=30.0, attention_layer=6, attention_step=0,
    min_weight=0.001, weight_source="entropy", visibility_alpha=30.0,
    weight_combine_mode="average", visibility_weight_ratio=0.5, visibility_callback=None,
)
print("Loading pose models (DepthAnything3 + FoundationPose) ...")
pose_estimator = PoseEstimator(DA3_WEIGHTS, fp_debug_dir=os.path.join(TMP_DIR, "fpose"), device=DEVICE)
print("All models loaded.")


# ── backend: reconstruct with MV-SAM3D run_multi_view ─────────────────────────
def reconstruct(rgb_images, object_masks, hand_masks, seed):
    # MV-SAM3D was trained on object-centred renders -> crop to 518 first.
    rgb_c, _hand_c, obj_c = preprocess_images_crop(rgb_images, hand_masks, object_masks)
    result = inference._pipeline.run_multi_view(
        view_images=rgb_c, view_masks=obj_c, view_pointmaps=None,
        seed=seed, mode="stochastic", stage1_inference_steps=50, stage2_inference_steps=25,
        decode_formats=["gaussian", "mesh"], with_mesh_postprocess=True,
        with_texture_baking=True, use_vertex_color=True, weighting_config=weighting_config,
        ss_weighting=True, ss_entropy_layer=9, ss_entropy_alpha=30.0, ss_warmup_steps=1,
    )
    obj_mesh = result.get("glb")
    gaussian = result["gaussian"][0] if result.get("gaussian") else None
    mesh_extract = result["mesh"][0] if result.get("mesh") else None
    torch.cuda.empty_cache()
    return obj_mesh, gaussian, mesh_extract


def render_turntable(gaussian, mesh_extract, path, num_frames=90):
    frames = None
    if gaussian is not None:
        try:
            frames = render_utils.render_video(gaussian, resolution=512, num_frames=num_frames,
                                               bg_color=(0, 0, 0))["color"]
        except Exception as e:
            print(f"Gaussian turntable failed ({e}); trying mesh normals."); frames = None
    if frames is None and mesh_extract is not None:
        frames = render_utils.render_video(mesh_extract, resolution=512, num_frames=num_frames, bg_color=(0, 0, 0))["normal"]
    if frames is not None:
        imageio.mimsave(path, frames, fps=30)
        return path
    return None


demo = ui_common.create_interface(
    reconstruct, render_turntable, pose_estimator,
    title="MV-SAM3D baseline",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866, debug=False, share=False)
