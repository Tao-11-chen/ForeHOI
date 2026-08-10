"""Feature-matching pose estimation: coarse initialization + PnP tracking.

Conventions
-----------
* All poses are object-to-camera (ob_in_cam), OpenCV camera frame
  (x right, y down, z forward), for the CENTERED mesh held by `renderer`.
  `estimate_poses_fm` converts them back to the original mesh frame.
* cv2.solvePnP(objectPoints=p_obj, imagePoints) yields exactly R,t with
  p_cam = R @ p_obj + t, i.e. ob_in_cam — no inversion needed.
* Match coordinates coming out of Mast3rMatcher live in the resized (square)
  MASt3R input frame; they are rescaled to the crop resolution (518) as
  float32 before any indexing.
"""
import cv2
import numpy as np

BORDER_PX = 3          # ignore matches this close to the MASt3R input border


def fibonacci_sphere(n):
    """n roughly uniform unit directions (golden-spiral sampling)."""
    i = np.arange(n, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (i / max(n - 1, 1)) * 2.0
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = golden * i
    return np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)


def look_at(eye, center=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    """OpenCV-convention world-to-camera matrix looking from `eye` at `center`.

    Applied to the centered object as its ob_in_cam.
    """
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    z = center - eye
    z /= max(np.linalg.norm(z), 1e-12)
    x = np.cross(up, z)
    nx = np.linalg.norm(x)
    if nx < 1e-9:  # looking straight along `up`: pick any orthogonal axis
        x = np.array([1.0, 0.0, 0.0])
    else:
        x /= nx
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)          # world axes -> camera rows
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = -R @ eye
    return pose


def _rescale_matches(matches, from_shape, to_hw):
    """(M,2) (x,y) from resized MASt3R coords to the (H,W) crop frame, float32."""
    h, w = from_shape
    sx, sy = to_hw[1] / w, to_hw[0] / h
    out = matches.astype(np.float32).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def _filter_matches(match, obj_mask, hw):
    """Border filter (resized frame) + object-mask filter (real frame, crop coords).

    Returns (render_xy (M,2) f32, real_xy (M,2) f32) in crop coordinates.
    The real-side mask filter is essential here (unlike ReconViaGen's static
    scene): background and the hand move rigidly with the camera / freely, so
    their matches would pull PnP away from the object's motion.
    """
    ha, wa = match["shape_a"]
    hb, wb = match["shape_b"]
    ma, mb = match["matches_a"], match["matches_b"]
    keep = (ma[:, 0] >= BORDER_PX) & (ma[:, 0] < wa - BORDER_PX) & \
           (ma[:, 1] >= BORDER_PX) & (ma[:, 1] < ha - BORDER_PX) & \
           (mb[:, 0] >= BORDER_PX) & (mb[:, 0] < wb - BORDER_PX) & \
           (mb[:, 1] >= BORDER_PX) & (mb[:, 1] < hb - BORDER_PX)
    ma = _rescale_matches(ma[keep], match["shape_a"], hw)
    mb = _rescale_matches(mb[keep], match["shape_b"], hw)
    if len(ma) == 0:
        return ma, mb  # both empty (0,2), already in crop coordinates
    xi = np.clip(np.round(mb[:, 0]).astype(int), 0, hw[1] - 1)
    yi = np.clip(np.round(mb[:, 1]).astype(int), 0, hw[0] - 1)
    in_obj = obj_mask[yi, xi]
    return ma[in_obj], mb[in_obj]


def _lift_render_points(render_xy, render_depth, K, ob_in_cam_render):
    """Lift matched render pixels to 3D points of the (centered) object.

    Samples the rendered depth at each match, back-projects to the render
    camera frame, then maps into object coordinates via inv(ob_in_cam_render).
    Matches landing on background (depth 0) are dropped.
    """
    H, W = render_depth.shape
    xi = np.clip(np.round(render_xy[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(render_xy[:, 1]).astype(int), 0, H - 1)
    z = render_depth[yi, xi]
    valid = z > 1e-6
    render_xy, z = render_xy[valid], z[valid]
    if len(render_xy) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), bool)
    Kinv = np.linalg.inv(K)
    uv1 = np.stack([render_xy[:, 0], render_xy[:, 1], np.ones_like(z)], axis=0)
    p_cam = (Kinv @ uv1) * z                                # (3,M) camera frame
    p_obj = np.linalg.inv(ob_in_cam_render)[:3, :3] @ p_cam \
        + np.linalg.inv(ob_in_cam_render)[:3, 3:4]
    return p_obj.T.astype(np.float32), valid


def solve_pnp(pts_obj, pts_img, K, pose_init=None, reproj_err=3.0,
              ransac_iters=2000, min_points=8, min_inliers=8, min_inlier_ratio=0.10):
    """RANSAC PnP (+ LM polish on the inlier set). Returns dict(pose, n_inliers,
    inlier_ratio) or None when the evidence is too weak — never raises."""
    n = len(pts_obj)
    if n < min_points:
        return None
    kwargs = dict(reprojectionError=reproj_err, iterationsCount=ransac_iters,
                  flags=cv2.SOLVEPNP_EPNP)
    if pose_init is not None:
        rvec, _ = cv2.Rodrigues(pose_init[:3, :3].astype(np.float64))
        kwargs.update(rvec=rvec, tvec=pose_init[:3, 3].astype(np.float64).reshape(3, 1),
                      useExtrinsicGuess=True)
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            np.ascontiguousarray(pts_obj, dtype=np.float32),
            np.ascontiguousarray(pts_img, dtype=np.float32),
            K.astype(np.float64), np.zeros(4, np.float32), **kwargs)
    except cv2.error:
        return None
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None
    if len(inliers) / n < min_inlier_ratio:
        return None
    inl = inliers[:, 0]
    try:  # LM polish on the inlier set, seeded with the RANSAC solution
        ok2, rvec2, tvec2 = cv2.solvePnP(pts_obj[inl].astype(np.float32),
                                         pts_img[inl].astype(np.float32),
                                         K.astype(np.float64), np.zeros(4, np.float32),
                                         rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
        if ok2:
            rvec, tvec = rvec2, tvec2
    except cv2.error:
        pass  # keep the RANSAC pose
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec[:, 0]
    return dict(pose=pose, n_inliers=len(inl), inlier_ratio=len(inl) / n)


class _FrameEvidence:
    """Matches + lifted 3D points between one render hypothesis and a real frame."""

    def __init__(self, pose, render_rgb, render_depth, match, real_mask, K, hw):
        self.pose = pose
        self.render_depth = render_depth
        ma, mb = _filter_matches(match, real_mask, hw)
        self.n_matches = len(ma)
        if self.n_matches:
            self.pts_obj, valid = _lift_render_points(ma, render_depth, K, pose)
            self.pts_img = mb[valid] if self.pts_obj.shape[0] else np.zeros((0, 2), np.float32)
        else:
            self.pts_obj = np.zeros((0, 3), np.float32)
            self.pts_img = np.zeros((0, 2), np.float32)

    @property
    def n_corres(self):
        return len(self.pts_obj)


def coarse_init(matcher, renderer, rgb, obj_mask, K, distance, hw,
                num_views=42, topk=5, refine_rounds=1, pnp_kwargs=None):
    """Frame-0 global pose search: render from `num_views` sphere directions at
    `distance`, batch-match all against the real frame, PnP the `topk`
    candidates by surviving correspondence count, keep the best-inlier pose.
    Returns a solve_pnp dict or None."""
    pnp_kwargs = pnp_kwargs or {}
    dirs = fibonacci_sphere(num_views)
    poses = [look_at(d * distance) for d in dirs]
    renders = [renderer.render(K, p, height=hw[0], width=hw[1]) for p in poses]
    matches = matcher.match_pairs([r[0] for r in renders], [rgb] * num_views)
    evids = [_FrameEvidence(p, r[0], r[1], m, obj_mask, K, hw)
             for p, r, m in zip(poses, renders, matches)]
    order = np.argsort([-e.n_corres for e in evids])[:topk]

    best = None
    for idx in order:
        ev = evids[idx]
        res = solve_pnp(ev.pts_obj, ev.pts_img, K, pose_init=ev.pose, **pnp_kwargs)
        if res and (best is None or res["n_inliers"] > best["n_inliers"]):
            best = res
    if best is None:
        return None
    # polish: re-render from the winner and match again (like the tracking loop)
    for _ in range(refine_rounds):
        refined = track_frame(matcher, renderer, rgb, obj_mask, K, best["pose"],
                              hw, iters=1, pnp_kwargs=pnp_kwargs)
        if refined is None:
            break
        best = refined
    return best


def track_frame(matcher, renderer, rgb, obj_mask, K, pose_prev, hw,
                iters=2, pnp_kwargs=None):
    """Track one frame from an initial pose: render -> match -> PnP, repeated
    `iters` times (each round re-renders from the latest estimate). Returns the
    best solve_pnp dict across rounds, or None when tracking fails."""
    pnp_kwargs = pnp_kwargs or {}
    pose, best = pose_prev, None
    for _ in range(iters):
        r_rgb, r_depth = renderer.render(K, pose, height=hw[0], width=hw[1])
        match = matcher.match(r_rgb, rgb)
        ev = _FrameEvidence(pose, r_rgb, r_depth, match, obj_mask, K, hw)
        res = solve_pnp(ev.pts_obj, ev.pts_img, K, pose_init=pose, **pnp_kwargs)
        if res is None:
            continue
        if best is None or res["n_inliers"] > best["n_inliers"]:
            best = res
        pose = res["pose"]
    return best


def _object_distance(depth, obj_mask, mesh_diag):
    """Median metric depth of the object -> camera-to-object distance guess."""
    d = depth[np.asarray(obj_mask).astype(bool)]
    d = d[np.isfinite(d) & (d > 1e-3)]
    if len(d) < 50:
        return max(4.0 * mesh_diag, 1.0)
    return max(float(np.median(d)), 2.0 * mesh_diag)


def estimate_poses_fm(matcher, renderer, rgb_images, object_masks, depths, Ks,
                      num_views=42, topk=5, track_iters=2, pnp_kwargs=None,
                      verbose=True):
    """Per-frame poses of the centered mesh held by `renderer`.

    rgb_images/object_masks/depths: per-frame (518,518,·) crop-space arrays;
    Ks: per-frame (3,3) intrinsics from DepthAnything3.
    Returns (poses (N,4,4) centered-frame ob_in_cam, lost (N,) bool).
    Raises RuntimeError when even frame 0 cannot be initialized.
    """
    n = len(rgb_images)
    hw = rgb_images[0].shape[:2]
    poses, lost = [], []
    pose_last = None
    for k in range(n):
        res = None
        if pose_last is not None:
            res = track_frame(matcher, renderer, rgb_images[k], object_masks[k],
                              Ks[k], pose_last, hw, iters=track_iters,
                              pnp_kwargs=pnp_kwargs)
        if res is None:  # frame 0, or re-initialization after a lost track
            dist = _object_distance(depths[k], object_masks[k], renderer.diag)
            res = coarse_init(matcher, renderer, rgb_images[k], object_masks[k],
                              Ks[k], dist, hw, num_views=num_views, topk=topk,
                              pnp_kwargs=pnp_kwargs)
        if res is None:
            if pose_last is None:
                raise RuntimeError(
                    "coarse pose initialization failed on frame 0 "
                    "(not enough render<->image matches; object too weakly textured?)")
            lost.append(True)
            poses.append(pose_last)  # hold the last valid pose for continuity
            if verbose:
                print(f"[pose_est_fm] frame {k}: track lost, holding last pose")
            continue
        lost.append(False)
        poses.append(res["pose"])
        pose_last = res["pose"]
        if verbose:
            print(f"[pose_est_fm] frame {k}: inliers {res['n_inliers']} "
                  f"({res['inlier_ratio']:.0%})")
    return np.stack(poses), np.asarray(lost, dtype=bool)
