"""FoundationPose loading + per-frame register/track with pose-overlay drawing.

Imports resolve via pose_est/__init__.py adding wheels/FoundationPose to sys.path
(FoundationPose uses `from Utils import *`, `import mycpp.build.mycpp` internally)."""
import os
import cv2
import numpy as np
import trimesh
import nvdiffrast.torch as dr

from estimater import FoundationPose
from learning.training.predict_score import ScorePredictor
from learning.training.predict_pose_refine import PoseRefinePredictor
from Utils import draw_posed_3d_box_with_depth, draw_xyz_axis_with_depth


def load_foundation_pose(device="cuda"):
    scorer = ScorePredictor()
    scorer.model.to(device)
    refiner = PoseRefinePredictor()
    refiner.model.to(device)
    return scorer, refiner


def estimate_pose(obj_mesh, vid_depth, rgb_images, object_masks, da3_intrinsics,
                  scorer, refiner, debug_dir, est_refine_iter=5):
    """Register on frame 0, track on the rest; draw 3D box + axes per frame.
    Returns (poses, overlay_rgbs)."""
    obj_mesh.vertex_normals = None
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=obj_mesh.vertices, model_normals=obj_mesh.vertex_normals,
        mesh=obj_mesh, scorer=scorer, refiner=refiner,
        debug_dir=debug_dir, debug=0, glctx=glctx,
    )
    to_origin, extents = trimesh.bounds.oriented_bounds(obj_mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    poses, rgbs = [], []
    for frame_id, color in enumerate(rgb_images):
        depth = vid_depth[frame_id]
        mask = object_masks[frame_id]
        K = np.array(da3_intrinsics[frame_id])
        if len(mask.shape) == 3:
            for c in range(3):
                if mask[..., c].sum() > 0:
                    mask = mask[..., c]
                    break
        mask = mask.astype(bool)
        if frame_id == 0:
            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
        else:
            pose = est.track_one(K=K, rgb=color, depth=depth, iteration=est_refine_iter)
        poses.append(pose.reshape(4, 4))
        color_bgr = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        center_pose = pose @ np.linalg.inv(to_origin)
        rgb_vis, dep_vis = draw_posed_3d_box_with_depth(K, img=color_bgr, depth=depth, ob_in_cam=center_pose, bbox=bbox)
        rgb_new, _ = draw_xyz_axis_with_depth(rgb_vis, depth=dep_vis, ob_in_cam=center_pose,
                                              scale=0.3, K=K, thickness=3, transparency=0, is_input_rgb=True)
        rgbs.append(rgb_new)
    return poses, rgbs
