"""Pose-overlay drawing (3D bounding box + XYZ axes) for the pose video.

Self-contained re-implementation of wheels/FoundationPose/Utils.py's
`draw_posed_3d_box_with_depth` / `draw_xyz_axis_with_depth` so pose_est_fm does
not import FoundationPose at all.

Clean color contract (the FoundationPose call chain mixed BGR/RGB flags and
relied on them cancelling out): both functions take an RGB uint8 image and
return an RGB uint8 image; drawing happens on an internal BGR copy. The depth
map is updated on a copy wherever an overlay primitive lands, so the box and
axes can be depth-chained without mutating the caller's array.
"""
import cv2
import numpy as np


def _to_homo(pts):
    return np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)


def _draw_segment(img, depth, p0, p1, ob_in_cam, K, color_bgr, linewidth):
    """Project the object-space segment p0->p1 and draw it, splatting its
    interpolated camera depth into `depth` along the drawn line."""
    H, W = img.shape[:2]
    cam = _to_homo(np.stack([p0, p1])) @ ob_in_cam.T          # (2,4) -> (2,3)
    proj = K @ cam[:, :3].T
    if (proj[2] <= 1e-6).any():                               # behind the camera
        return img, depth
    uv = (proj[:2] / proj[2]).T                               # (2,2) float
    pt1 = tuple(int(round(v)) for v in uv[0])
    pt2 = tuple(int(round(v)) for v in uv[1])
    img = cv2.line(img, tuple(pt1), tuple(pt2), color=color_bgr,
                   thickness=linewidth, lineType=cv2.LINE_AA)
    seg = np.linalg.norm(uv[1] - uv[0])
    if seg < 1e-9:                                            # degenerate projection
        return img, depth
    canvas = cv2.line(np.zeros((H, W), np.uint8), tuple(pt1), tuple(pt2), 255, linewidth)
    ys, xs = canvas.nonzero()
    for y, x in zip(ys, xs):
        t = np.linalg.norm(np.array([x, y]) - uv[0]) / seg
        depth[y, x] = (1.0 - t) * cam[0, 2] + t * cam[1, 2]
    return img, depth


def draw_posed_3d_box_with_depth(K, img_rgb, depth, ob_in_cam, bbox,
                                 line_color=(0, 255, 0), linewidth=2):
    """Draw the 12 edges of the object-space bbox. Returns (rgb, depth).

    line_color is RGB; it is converted to BGR for cv2 internally.
    """
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    depth = depth.copy()
    xmin, ymin, zmin = np.asarray(bbox).min(axis=0)
    xmax, ymax, zmax = np.asarray(bbox).max(axis=0)
    edges = []
    for y in (ymin, ymax):
        for z in (zmin, zmax):
            edges.append((np.array([xmin, y, z]), np.array([xmax, y, z])))
    for x in (xmin, xmax):
        for z in (zmin, zmax):
            edges.append((np.array([x, ymin, z]), np.array([x, ymax, z])))
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            edges.append((np.array([x, y, zmin]), np.array([x, y, zmax])))
    bgr = (line_color[2], line_color[1], line_color[0])
    for p0, p1 in edges:
        img, depth = _draw_segment(img, depth, p0, p1, ob_in_cam, K, bgr, linewidth)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), depth


def draw_xyz_axis_with_depth(img_rgb, depth, ob_in_cam, scale=0.1, K=np.eye(3),
                             thickness=3):
    """Draw the object-frame axes at the origin: X red, Y green, Z blue.
    Returns (rgb, depth)."""
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    depth = depth.copy()
    origin = np.zeros(3)
    for direction, bgr in ((np.array([1.0, 0, 0]), (0, 0, 255)),      # X: red
                           (np.array([0, 1.0, 0]), (0, 255, 0)),      # Y: green
                           (np.array([0, 0, 1.0]), (255, 0, 0))):     # Z: blue
        img, depth = _draw_segment(img, depth, origin, direction * scale,
                                   ob_in_cam, K, bgr, thickness)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), depth
