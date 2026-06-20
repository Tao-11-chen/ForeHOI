"""forehoi.pose_est — object 6-DoF pose estimation (DepthAnything3 + FoundationPose).

sam3d_objects-free on purpose, so it can be shared by BOTH reconstruction apps
(app_forehoi_sam3d.py and app_mv_sam3d.py), which use different sam3d_objects
packages. Operates on a trimesh mesh + numpy RGB frames + masks.

Usage:
    from pose_est import PoseEstimator
    pe = PoseEstimator(da3_weights="weights/DA3", fp_debug_dir="output/fpose")
    rgbs, scale = pe.run(obj_mesh, rgb_images, object_masks, hand_masks)
"""
import os
import sys
import glob


def _ensure_cuda_home():
    """FoundationPose renders pose hypotheses with nvdiffrast, which JIT-compiles a
    CUDA plugin on first use and needs <CUDA_HOME>/include/cuda_runtime.h + bin/nvcc.
    `core/inference.py` sets CUDA_HOME=$CONDA_PREFIX, whose CUDA headers live under
    targets/x86_64-linux/include (not include/), so the build fails. Point CUDA_HOME at
    a full toolkit if the current one is missing the headers."""
    def _ok(p):
        return bool(p) and os.path.exists(os.path.join(p, "include", "cuda_runtime.h")) \
            and os.path.exists(os.path.join(p, "bin", "nvcc"))
    target = os.environ.get("CUDA_HOME", "")
    if not _ok(target):
        for cand in ["/usr/local/cuda"] + sorted(glob.glob("/usr/local/cuda-*"), reverse=True):
            if _ok(cand):
                target = cand
                break
    if not _ok(target):
        return
    os.environ["CUDA_HOME"] = target
    os.environ["CUDA_PATH"] = target
    # torch caches CUDA_HOME as a module global at import of cpp_extension (here it was
    # already set to $CONDA_PREFIX by core/inference.py); override it so nvdiffrast's
    # JIT build uses the real toolkit headers.
    try:
        import torch.utils.cpp_extension as _ce
        _ce.CUDA_HOME = target
    except Exception:
        pass


_ensure_cuda_home()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FoundationPose expects its own dir on sys.path (internal `from Utils import *`,
# `import mycpp.build.mycpp`); DA3 package lives under Depth-Anything-3/src.
for _p in (os.path.join(ROOT, "wheels"),
           os.path.join(ROOT, "wheels/FoundationPose"),
           os.path.join(ROOT, "wheels/Depth-Anything-3/src")):
    if _p not in sys.path:
        sys.path.append(_p)

from pose_est.crop import preprocess_images_crop          # noqa: E402
from pose_est.scale import estimate_metric_scale          # noqa: E402
from pose_est.estimator import PoseEstimator              # noqa: E402

__all__ = ["PoseEstimator", "preprocess_images_crop", "estimate_metric_scale"]
