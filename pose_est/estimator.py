"""PoseEstimator: ties DepthAnything3 -> metric scale -> FoundationPose into one
object that produces a 6-DoF pose-overlay video from a reconstructed mesh."""
import numpy as np

from pose_est.crop import preprocess_images_crop
from pose_est.scale import estimate_metric_scale
from pose_est.depth import DepthEstimator
from pose_est.foundation_pose import load_foundation_pose, estimate_pose


class PoseEstimator:
    def __init__(self, da3_weights, fp_debug_dir, device="cuda"):
        self.fp_debug_dir = fp_debug_dir
        self.depth = DepthEstimator(da3_weights, device=device)
        self.scorer, self.refiner = load_foundation_pose(device=device)

    def run(self, obj_mesh, rgb_images, object_masks, hand_masks=None, est_refine_iter=5):
        """Scale `obj_mesh` to metric units (in place) and run FoundationPose over
        the frames. Returns (overlay_rgbs, poses, scale).

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
        poses, rgbs = estimate_pose(
            obj_mesh, depth, rgb_c, obj_c, K,
            self.scorer, self.refiner, self.fp_debug_dir, est_refine_iter=est_refine_iter,
        )
        return rgbs, poses, scale
