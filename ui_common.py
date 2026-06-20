"""Shared UI + SAM2 + reconstruction-orchestration for the ForeHOI apps.

This module is intentionally free of any `sam3d_objects` import so it can be
shared by both apps, which use *different* sam3d_objects packages:
  * app_forehoi_sam3d.py  -> local trained pipeline (tools/)
  * app_mv_sam3d.py       -> MV-SAM3D fork (wheels/MV_SAM3D)

Each app supplies backend callbacks and calls `create_interface(...)`:
  * reconstruct_fn(rgb_images, object_masks, hand_masks, seed)
        -> (obj_mesh: trimesh.Trimesh, gaussian, mesh_extract)
  * render_turntable_fn(gaussian, mesh_extract, out_path) -> path | None
  * pose_estimator: pose_est.PoseEstimator | None
"""
import os
import tempfile
import subprocess
from typing import List, Tuple

import cv2
import numpy as np
import torch
import imageio
import gradio as gr

try:
    from gradio_litmodel3d import LitModel3D
except ImportError:
    print("Warning: gradio_litmodel3d not found. Falling back to gr.Model3D.")
    LitModel3D = None

# ── module config (apps may override) ────────────────────────────────────────
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_wild")
MAX_SEED = np.iinfo(np.int32).max
OBJECT_OBJ_ID = 0
HAND_OBJ_ID = 1
_SAM2 = None


def init_sam2(ckpt, cfg, device="cuda"):
    """Load the SAM2 video predictor once (each app process calls this)."""
    global _SAM2
    from sam2.build_sam import build_sam2_video_predictor
    _SAM2 = build_sam2_video_predictor(cfg, ckpt, device=device)
    return _SAM2


# ── video frame sampling ──────────────────────────────────────────────────────
def convert_video_to_mp4(input_path, output_path):
    subprocess.run(['ffmpeg', '-i', input_path, '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '23', '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
                    '-y', output_path], check=True, capture_output=True)


def sample_frames_from_video(video_path, num_samples=10):
    if not video_path:
        raise ValueError("No input video provided")
    if not video_path.endswith('.mp4'):
        tmp_mp4 = tempfile.mktemp(suffix='.mp4')
        convert_video_to_mp4(video_path, tmp_mp4)
        video_path = tmp_mp4
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS)) or 5
    if total_frames == 0:
        raise ValueError("Unable to read video frames")
    if num_samples >= total_frames:
        frame_indices = list(range(total_frames))
    else:
        step = total_frames / num_samples
        frame_indices = [int(i * step) for i in range(num_samples)]

    def sharpness(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    os.makedirs(TMP_DIR, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=TMP_DIR)
    frames, search_window = [], 3
    for i, target_idx in enumerate(frame_indices):
        best_frame, best_sharp = None, -1
        for f_idx in range(max(0, target_idx - search_window), min(total_frames - 1, target_idx + search_window) + 1):
            video.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = video.read()
            if ok:
                s = sharpness(frame)
                if s > best_sharp:
                    best_sharp, best_frame = s, frame.copy()
        if best_frame is not None:
            frames.append(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(temp_dir, f"{len(frames)-1:05d}.jpg"), best_frame)
    video.release()
    return frames, temp_dir, fps


# ── SAM2 dual-mask propagation + masked video ─────────────────────────────────
def propagate_masks_sam2(frames_dir, num_frames, object_points, hand_points):
    with torch.inference_mode():
        state = _SAM2.init_state(video_path=frames_dir)
        _SAM2.reset_state(state)
        H, W = state['video_height'], state['video_width']
        if object_points:
            _SAM2.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=OBJECT_OBJ_ID,
                points=np.array(object_points, dtype=np.float32),
                labels=np.ones(len(object_points), dtype=np.int32))
        if hand_points:
            _SAM2.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=HAND_OBJ_ID,
                points=np.array(hand_points, dtype=np.float32),
                labels=np.ones(len(hand_points), dtype=np.int32))
        obj_by_idx, hand_by_idx = {}, {}
        for f_idx, out_obj_ids, logits in _SAM2.propagate_in_video(state):
            obj_m = np.zeros((H, W), dtype=bool); hand_m = np.zeros((H, W), dtype=bool)
            for i, oid in enumerate(out_obj_ids):
                mm = (logits[i] > 0.0).cpu().numpy().squeeze()
                if oid == OBJECT_OBJ_ID:
                    obj_m = mm
                elif oid == HAND_OBJ_ID:
                    hand_m = mm
            obj_by_idx[f_idx] = obj_m; hand_by_idx[f_idx] = hand_m
    object_masks = [obj_by_idx.get(i, np.zeros((H, W), bool)) for i in range(num_frames)]
    hand_masks = [hand_by_idx.get(i, np.zeros((H, W), bool)) for i in range(num_frames)]
    return object_masks, hand_masks


def create_masked_video(frames, object_masks, hand_masks, fps, output_path, opacity=0.5):
    out_frames = []
    for frame, o_mask, h_mask in zip(frames, object_masks, hand_masks):
        overlay = frame.copy().astype(np.float32)
        color_layer = np.zeros_like(overlay)
        color_layer[o_mask] = (255, 0, 0)
        color_layer[h_mask] = (0, 255, 0)
        idx = o_mask | h_mask
        overlay[idx] = overlay[idx] * (1 - opacity) + color_layer[idx] * opacity
        out_frames.append(np.clip(overlay, 0, 255).astype(np.uint8))
    imageio.mimsave(output_path, out_frames, fps=max(1, fps))
    return output_path


# ── app state + point drawing ─────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.frames = []
        self.base_first_frame = None
        self.temp_dir = None
        self.fps = 5
        self.object_points = []
        self.hand_points = []
        self.object_masks = []
        self.hand_masks = []
        self.mesh_path = None

app_state = AppState()


def draw_points_on_frame(frame, object_points, hand_points):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for idx, p in enumerate(object_points):
        cv2.circle(bgr, p, 12, (255, 255, 255), -1); cv2.circle(bgr, p, 8, (0, 0, 255), -1)
        cv2.putText(bgr, "obj" + str(idx + 1), (p[0] + 15, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for idx, p in enumerate(hand_points):
        cv2.circle(bgr, p, 12, (255, 255, 255), -1); cv2.circle(bgr, p, 8, (0, 255, 0), -1)
        cv2.putText(bgr, "hand" + str(idx + 1), (p[0] + 15, p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_marked_frame():
    if app_state.base_first_frame is None:
        return None
    return draw_points_on_frame(app_state.base_first_frame, app_state.object_points, app_state.hand_points)


# ── reconstruction orchestrator (backend-agnostic) ───────────────────────────
def recon_hand_object(reconstruct_fn, render_turntable_fn, pose_estimator,
                      rgb_images, hand_masks, object_masks, seed=42):
    """reconstruct_fn(rgb, obj_masks, hand_masks, seed) -> (obj_mesh, gaussian, mesh_extract).
    Returns (video_path, mesh_path) or (None, None)."""
    print(f"Starting 3D reconstruction with {len(rgb_images)} frames...")
    obj_mesh, gaussian, mesh_extract = reconstruct_fn(rgb_images, object_masks, hand_masks, seed)
    if obj_mesh is None and mesh_extract is None:
        return None, None
    os.makedirs(TMP_DIR, exist_ok=True)
    temp_dir = tempfile.mkdtemp(dir=TMP_DIR)
    video_path = os.path.join(temp_dir, "reconstruction.mp4")
    mesh_path = os.path.join(temp_dir, "reconstruction.glb")

    if pose_estimator is not None:
        # No fallback: if the pose pipeline fails, let the exception propagate.
        rgbs, _poses, scale = pose_estimator.run(obj_mesh, rgb_images, object_masks, hand_masks)
        print(f"Estimated metric scale = {scale:.4f}")
        imageio.mimsave(video_path, rgbs, fps=1)
    else:
        # No pose estimator configured -> turntable render of the reconstruction.
        video_path = render_turntable_fn(gaussian, mesh_extract, video_path)

    if obj_mesh is not None:
        obj_mesh.export(mesh_path)
    else:
        mesh_path = None
    torch.cuda.empty_cache()
    print(f"3D reconstruction completed: video={video_path}, mesh={mesh_path}")
    return video_path, mesh_path


# ── gradio handlers ───────────────────────────────────────────────────────────
def process_video_upload(video_path, num_samples=10):
    global app_state
    try:
        app_state = AppState()
        frames, temp_dir, fps = sample_frames_from_video(video_path, int(num_samples))
        app_state.frames = frames
        app_state.base_first_frame = frames[0].copy() if frames else None
        app_state.temp_dir = temp_dir
        app_state.fps = fps
        if not frames:
            return gr.update(visible=False), "No frames extracted from video", gr.update(), gr.update()
        return (gr.update(value=get_marked_frame(), visible=True),
                f"Extracted {len(frames)} frames. Click points on the first frame for object (red) and hand (green).",
                gr.update(visible=False), gr.update(visible=False))
    except Exception as e:
        return gr.update(visible=False), f"Error processing video: {e}", gr.update(), gr.update()


def handle_point_selection(mask_type, evt: gr.SelectData):
    x, y = evt.index[0], evt.index[1]
    if mask_type == "object":
        app_state.object_points.append((x, y))
        msg = f"Object point added at ({x}, {y}). Total object points: {len(app_state.object_points)}"
    elif mask_type == "hand":
        app_state.hand_points.append((x, y))
        msg = f"Hand point added at ({x}, {y}). Total hand points: {len(app_state.hand_points)}"
    else:
        return "Invalid mask type", get_marked_frame()
    return msg, get_marked_frame()


def clear_object_points():
    app_state.object_points = []
    return "Object points cleared", get_marked_frame()


def clear_hand_points():
    app_state.hand_points = []
    return "Hand points cleared", get_marked_frame()


def clear_all_points():
    app_state.object_points = []; app_state.hand_points = []
    return "All points cleared", get_marked_frame()


def generate_masks_and_video():
    try:
        if not app_state.frames or not app_state.temp_dir:
            return gr.update(), gr.update(), "No frames available. Please upload a video first.", gr.update(interactive=False)
        if not app_state.object_points:
            return gr.update(), gr.update(), "Please mark at least one Object point on the first frame.", gr.update(interactive=False)
        object_masks, hand_masks = propagate_masks_sam2(
            app_state.temp_dir, len(app_state.frames), app_state.object_points, app_state.hand_points)
        app_state.object_masks = object_masks
        app_state.hand_masks = hand_masks
        out = os.path.join(app_state.temp_dir, "masked_video.mp4")
        create_masked_video(app_state.frames, object_masks, hand_masks, app_state.fps, out)
        return (gr.update(visible=False), gr.update(value=out, visible=True),
                "Masks generated! Ready for 3D reconstruction.", gr.update(interactive=True))
    except Exception as e:
        return gr.update(), gr.update(), f"Error generating masks: {e}", gr.update(interactive=False)


# ── interface ─────────────────────────────────────────────────────────────────
def create_interface(reconstruct_fn, render_turntable_fn, pose_estimator, title, subtitle=""):
    def perform_reconstruction(seed):
        if not app_state.frames:
            return (gr.update(), gr.update(), "No frames available. Please upload and process a video first.",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
        if not app_state.object_masks:
            return (gr.update(), gr.update(), "No masks available. Please generate masks first.",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
        try:
            s = int(seed) if seed is not None else np.random.randint(0, MAX_SEED)
            video_path, mesh_path = recon_hand_object(
                reconstruct_fn, render_turntable_fn, pose_estimator,
                app_state.frames, app_state.hand_masks, app_state.object_masks, seed=s)
        except Exception as e:
            import traceback; traceback.print_exc()
            return (gr.update(), gr.update(), f"Reconstruction failed: {e}",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
        if mesh_path is None:
            return (gr.update(), gr.update(), "Reconstruction failed: no voxels produced. Try a different seed or more frames.",
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
        app_state.mesh_path = mesh_path
        return (gr.update(value=video_path, visible=True), gr.update(value=mesh_path, visible=True),
                "3D reconstruction completed successfully!",
                gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))

    with gr.Blocks(title=title, css="h1, h2, h3 { text-align: center; display: block; }") as app:
        gr.Markdown(f"# {title}\n{subtitle}\nUpload a video → click object/hand points → SAM2 masks → 3D + pose.")
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video", format="mp4", height=512)
                num_samples = gr.Slider(minimum=2, maximum=24, value=15, step=1, label="Number of frames to sample")
                seed_input = gr.Number(label="Seed", value=42, precision=0)
                process_btn = gr.Button("Process Video", variant="primary")
            with gr.Column():
                first_frame_display = gr.Image(
                    label="First Frame - Click to mark object/hand points",
                    interactive=False, type="numpy", height=512, visible=False, format="png")
                output_video = gr.Video(label="Masked Video Output", height=512, visible=False)
                output_status = gr.Textbox(label="Status", interactive=False)
        with gr.Column(visible=True) as controls_col:
            gr.Markdown("### Point Selection")
            mask_type_radio = gr.Radio(choices=["object", "hand"], value="object", label="Mask Type")
            with gr.Row():
                clear_object_btn = gr.Button("Clear Object Points")
                clear_hand_btn = gr.Button("Clear Hand Points")
                clear_all_btn = gr.Button("Clear All Points", variant="stop")
            point_info = gr.Textbox(label="Point Info", interactive=False)
            generate_btn = gr.Button("Generate Masked Video", variant="primary")
            reconstruct_btn = gr.Button("Reconstruct 3D + Pose", variant="primary", interactive=False)
        with gr.Row(visible=False) as reconstruction_row:
            with gr.Column():
                reconstruction_video = gr.Video(label="3D Reconstruction / Pose Overlay", height=512)
                reconstruction_status = gr.Textbox(label="Reconstruction Status", interactive=False)
            with gr.Column():
                model_3d_viewer = (LitModel3D(label="3D Mesh", height=512, exposure=10.0)
                                   if LitModel3D is not None else gr.Model3D(label="3D Mesh", height=512))

        process_btn.click(process_video_upload, inputs=[video_input, num_samples],
                          outputs=[first_frame_display, output_status, num_samples, process_btn])
        first_frame_display.select(handle_point_selection, inputs=[mask_type_radio],
                                   outputs=[point_info, first_frame_display])
        clear_object_btn.click(clear_object_points, outputs=[point_info, first_frame_display])
        clear_hand_btn.click(clear_hand_points, outputs=[point_info, first_frame_display])
        clear_all_btn.click(clear_all_points, outputs=[point_info, first_frame_display])
        generate_btn.click(generate_masks_and_video,
                           outputs=[first_frame_display, output_video, output_status, reconstruct_btn])
        reconstruct_btn.click(perform_reconstruction, inputs=[seed_input],
                              outputs=[reconstruction_video, model_3d_viewer, reconstruction_status,
                                       controls_col, reconstruction_row, output_status])
    return app
