"""DepthAnything3 wrapper: per-frame metric depth + intrinsics + metric pointmaps.

Copied verbatim from pose_est/depth.py so pose_est_fm stays free of the
FoundationPose import chain.

NOTE: pred.extrinsics are world-to-camera (w2c) while unproject_depth expects
camera-to-world (c2w). The pointmaps below are therefore NOT true world-space
points. This is harmless for the only consumer (estimate_metric_scale), because
a rigid transform preserves the bbox diagonal — but do not use these pointmaps
for world-space alignment without inverting the extrinsics first.
"""
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import unproject_depth


class DepthEstimator:
    def __init__(self, weights_dir, device="cuda"):
        self.model = DepthAnything3.from_pretrained(weights_dir).to(device).eval()

    def infer(self, rgb_list, process_res=518):
        """rgb_list: list of (H,W,3) uint8. Returns (depth (N,H,W), intrinsics (N,3,3),
        extrinsics (N,4,4), pointmaps (N,H,W,3) metric)."""
        pred = self.model.inference(
            image=rgb_list, process_res=process_res,
            process_res_method="upper_bound_resize", export_dir=None, export_format="glb",
        )
        depth, K, ext = pred.depth, pred.intrinsics, pred.extrinsics
        pointmaps = unproject_depth(
            torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1),
            torch.from_numpy(K).unsqueeze(0),
            torch.from_numpy(ext).unsqueeze(0),
        ).squeeze(0).cpu().numpy()
        torch.cuda.empty_cache()
        return depth, K, ext, pointmaps
