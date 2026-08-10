"""forehoi.pose_est_fm — object 6-DoF pose estimation via feature matching
(MASt3R reciprocal-NN matches + PnP / 3D-3D registration), an alternative to
the FoundationPose-based `pose_est` package.

Pipeline: crop around hand∪object -> DepthAnything3 (metric depth, intrinsics,
pointmaps) -> metric scale recovery -> per-frame pose:
  * frame 0  : coarse initialization — render the mesh from a sphere of
               candidate viewpoints, match each against the real frame with
               MASt3R, PnP the top candidates, keep the best-inlier pose.
  * frame k>0: tracking — render from the previous pose, match, PnP, iterate;
               fall back to coarse initialization when tracking is lost.

Deliberately free of any FoundationPose / sam3d_objects imports so it can be
used by both reconstruction apps. Operates on a trimesh mesh + numpy RGB
frames + masks, same contract as pose_est.PoseEstimator:

    from pose_est_fm import PoseEstimatorFM
    pe = PoseEstimatorFM(da3_weights="weights/DA3",
                         mast3r_weights="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    rgbs, poses, scale = pe.run(obj_mesh, rgb_images, object_masks, hand_masks)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Import layout mirrors ReconViaGen:
#   * ROOT on sys.path       -> `from wheels.mast3r.model import ...` (namespace pkgs)
#   * wheels on sys.path     -> mast3r's internal `import mast3r.utils.path_to_dust3r`,
#                               which itself prepends wheels/dust3r for `dust3r.*` imports
#   * DA3 src on sys.path    -> `from depth_anything_3.api import DepthAnything3`
for _p in (ROOT,
           os.path.join(ROOT, "wheels"),
           os.path.join(ROOT, "wheels/Depth-Anything-3/src")):
    if _p not in sys.path:
        sys.path.append(_p)

from pose_est_fm.estimator import PoseEstimatorFM  # noqa: E402

__all__ = ["PoseEstimatorFM"]
