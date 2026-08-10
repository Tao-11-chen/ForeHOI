"""PoseEstimatorFM: DepthAnything3 -> metric scale -> MASt3R feature-matching
pose (coarse init on frame 0, PnP tracking after), producing the same
(overlay_rgbs, poses, scale) triple as pose_est.PoseEstimator so the Gradio
front-end (ui_common.recon_hand_object) works unchanged.
"""
import numpy as np
import trimesh

from pose_est_fm.crop import preprocess_images_crop
from pose_est_fm.scale import estimate_metric_scale
from pose_est_fm.depth import DepthEstimator
from pose_est_fm.renderer import MeshRenderer
from pose_est_fm.matcher import Mast3rMatcher, DEFAULT_MAST3R_WEIGHTS
from pose_est_fm.pose import estimate_poses_fm
from pose_est_fm.overlay import draw_posed_3d_box_with_depth, draw_xyz_axis_with_depth


class PoseEstimatorFM:
    def __init__(self, da3_weights, mast3r_weights=DEFAULT_MAST3R_WEIGHTS, device="cuda",
                 low_vram=False, num_views=42, topk=5, track_iters=2,
                 batch_size=4, reproj_err=3.0):
        self.device = device
        self.num_views, self.topk, self.track_iters = num_views, topk, track_iters
        self.pnp_kwargs = dict(reproj_err=reproj_err)
        self.depth = DepthEstimator(da3_weights, device=device)
        self.matcher = Mast3rMatcher(mast3r_weights, device=device,
                                     batch_size=batch_size, low_vram=low_vram)
        self.renderer = MeshRenderer(device=device)
        self.last_lost = None   # per-frame "track lost" flags of the latest run

    def run(self, obj_mesh, rgb_images, object_masks, hand_masks=None):
        """Scale `obj_mesh` to metric units (in place) and estimate its 6-DoF pose
        in every frame. Returns (overlay_rgbs, poses, scale).

        rgb_images   : list of (H,W,3) uint8 (original resolution)
        object_masks : list of (H,W) bool
        hand_masks   : list of (H,W) bool (optional; only used to frame the crop)
        """
        if hand_masks is None:
            hand_masks = [np.zeros_like(m) for m in object_masks]
        rgb_c, _hand_c, obj_c = preprocess_images_crop(rgb_images, hand_masks, object_masks)
        depth, K, _ext, pointmaps = self.depth.infer(rgb_c)
        matching_id = int(np.argmax([int(np.sum(m)) for m in obj_c]))
        scale = estimate_metric_scale(pointmaps, obj_c, np.asarray(obj_mesh.vertices), matching_id)
        obj_mesh.vertices = obj_mesh.vertices * scale

        self.renderer.set_mesh(obj_mesh)
        poses_c, self.last_lost = estimate_poses_fm(
            self.matcher, self.renderer, rgb_c,
            [np.asarray(m).astype(bool) for m in obj_c], depth, K,
            num_views=self.num_views, topk=self.topk,
            track_iters=self.track_iters, pnp_kwargs=self.pnp_kwargs)

        # renderer poses describe the CENTERED mesh; map back to the original
        # object frame: pose = pose_centered @ translate(-center)
        to_centered = np.eye(4)
        to_centered[:3, 3] = -self.renderer.center
        poses = [p @ to_centered for p in poses_c]

        to_origin, extents = trimesh.bounds.oriented_bounds(obj_mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        axis_len = 0.5 * float(self.renderer.diag)
        rgbs = []
        for k, color in enumerate(rgb_c):
            center_pose = poses[k] @ np.linalg.inv(to_origin)
            rgb_vis, dep = draw_posed_3d_box_with_depth(
                K[k], color, depth[k], center_pose, bbox)
            rgb_vis, _ = draw_xyz_axis_with_depth(
                rgb_vis, dep, center_pose, scale=axis_len, K=K[k], thickness=3)
            rgbs.append(rgb_vis)
        return rgbs, poses, scale
