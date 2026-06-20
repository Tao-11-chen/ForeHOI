"""Recover the reconstructed mesh's metric scale from DepthAnything3's metric
object point-map (robust bbox-diagonal ratio; render-free).

FoundationPose does its own global pose search, so the only thing it needs from
us is a metrically-scaled mesh — this provides it."""
import numpy as np
from PIL import Image


def estimate_metric_scale(da3_pointmaps, object_masks, mesh_vertices, matching_id):
    pts = np.asarray(da3_pointmaps[matching_id])
    m = np.asarray(object_masks[matching_id]).astype(bool)
    if m.shape != pts.shape[:2]:
        m = np.array(Image.fromarray((m.astype(np.uint8) * 255)).resize(
            (pts.shape[1], pts.shape[0]), Image.Resampling.NEAREST)) > 127
    obj_pts = pts[m]
    obj_pts = obj_pts[np.isfinite(obj_pts).all(axis=1)]
    if obj_pts.shape[0] < 50:
        return 1.0
    lo = np.percentile(obj_pts, 2, axis=0)
    hi = np.percentile(obj_pts, 98, axis=0)
    metric_diag = float(np.linalg.norm(hi - lo))
    v = np.asarray(mesh_vertices)
    mesh_diag = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
    if mesh_diag < 1e-6:
        return 1.0
    return metric_diag / mesh_diag
