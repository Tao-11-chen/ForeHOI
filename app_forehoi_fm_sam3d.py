"""ForeHOI app (feature-matching pose variant) — trained stage-1 + sam3d stage-2
SLAT, then DepthAnything3 + MASt3R feature matching (coarse viewpoint search +
PnP tracking) for the 6-DoF pose-overlay video. Alternative to
app_forehoi_sam3d.py's FoundationPose backend; better suited to textured
objects, and needs no FoundationPose weights.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)
sys.path.insert(0, _HERE)
sys.path.append(os.path.join(_HERE, "core"))

import numpy as np
import torch
import trimesh
import imageio

import tools                              # puts root + core on sys.path
import tools.mask_eval_common as C
from tools.pipeline import build_full_pipeline
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

import ui_common
from pose_est_fm import PoseEstimatorFM

# ── config ────────────────────────────────────────────────────────────────────
SAM2_CKPT   = os.path.join(_HERE, "weights/sam2.1_hiera_large.pt")
SAM2_CFG    = "configs/sam2.1/sam2.1_hiera_l.yaml"
MV_CKPT     = os.path.join(_HERE, "checkpoints/forehoi.ckpt")  # LoRA-merged, self-contained
DA3_WEIGHTS = os.path.join(_HERE, "weights/DA3")
# HF hub id (auto-download) or a local .pth path
MAST3R_WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
TMP_DIR     = os.path.join(_HERE, "output_wild")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SS_STEPS, SS_CFG, SLAT_STEPS = 25, 7.0, 12

os.makedirs(TMP_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", TMP_DIR)
ui_common.TMP_DIR = TMP_DIR
if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── load models ───────────────────────────────────────────────────────────────
print("Loading SAM2 ...")
ui_common.init_sam2(SAM2_CKPT, SAM2_CFG, DEVICE)
print(f"Loading reconstruction pipeline from {MV_CKPT} ...")
pipeline, _ = build_full_pipeline(MV_CKPT, DEVICE)
print("Loading pose models (DepthAnything3 + MASt3R) ...")
# low_vram=True offloads MASt3R to CPU between matching rounds (slower, but
# leaves room for the reconstruction pipeline + DA3 on smaller GPUs)
pose_estimator = PoseEstimatorFM(DA3_WEIGHTS, mast3r_weights=MAST3R_WEIGHTS,
                                 device=DEVICE, low_vram=True)
print("All models loaded.")


# ── backend: reconstruct (trained stage-1 + sam3d stage-2) ─────────────────
def reconstruct(rgb_images, object_masks, hand_masks, seed):
    image = torch.stack([torch.from_numpy(r).permute(2, 0, 1).float() / 255.0 for r in rgb_images])
    obj = torch.stack([torch.from_numpy(m.astype(np.float32))[None] for m in object_masks])
    hand = torch.stack([torch.from_numpy(m.astype(np.float32))[None] for m in hand_masks])
    n = image.shape[0]

    cond, raw_ctx = C.build_cond(pipeline, image, obj, hand, DEVICE)
    z_shape, _z_mask = C.joint_sample(pipeline, cond, raw_ctx, n, steps=SS_STEPS, cfg=SS_CFG, device=DEVICE, seed=seed)
    coords = C.decode_shape(pipeline, z_shape)
    if coords is None or len(coords) == 0:
        return None, None, None

    ref_rgb = (image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    ref_alpha = (obj[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    rgba = np.concatenate([ref_rgb, ref_alpha[..., None]], axis=-1)
    slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
    coords_cpu = torch.from_numpy(coords).int()  # always CPU; pin zeros to same device
    coords4 = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32, device=coords_cpu.device), coords_cpu], dim=1).to(DEVICE)
    with torch.no_grad():
        slat = pipeline.sample_slat(slat_input, coords4, inference_steps=SLAT_STEPS)
        outputs = pipeline.decode_slat(slat, ["gaussian", "mesh"])
        outputs = pipeline.postprocess_slat_output(outputs, with_mesh_postprocess=False,
                                                   with_texture_baking=False, use_vertex_color=True)
    torch.cuda.empty_cache()
    obj_mesh = outputs.get("glb")
    gaussian = outputs["gaussian"][0] if outputs.get("gaussian") else None
    mesh_extract = outputs["mesh"][0] if outputs.get("mesh") else None
    if obj_mesh is None and mesh_extract is not None:
        v = mesh_extract.vertices.float().cpu().numpy(); f = mesh_extract.faces.cpu().numpy()
        vc = (np.clip(mesh_extract.vertex_attrs[:, :3].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        vc = np.concatenate([vc, np.full((vc.shape[0], 1), 255, np.uint8)], axis=1)
        obj_mesh = trimesh.Trimesh(v, f, vertex_colors=vc, process=False)
    return obj_mesh, gaussian, mesh_extract


def render_turntable(gaussian, mesh_extract, path, num_frames=90):
    frames = None
    if gaussian is not None:
        try:
            frames = render_utils.render_video(gaussian, resolution=512, num_frames=num_frames,
                                               bg_color=(0, 0, 0), backend="gsplat")["color"]
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
    title="ForeHOI (feature-matching pose)",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7867, debug=False, share=False)
