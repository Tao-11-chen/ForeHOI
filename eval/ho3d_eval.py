#!/usr/bin/env python3
"""ForeHOI on HO3D: trained stage-1 + SAM3D stage-2 SLAT, run on the raw processed_ho3d.

Same pipeline as app_forehoi_sam3d.py (build_full_pipeline -> stage-1 shape/amodal
mask + stage-2 colored mesh). Parses the original processed_ho3d directly (no caching
step) — same structure as hot3d_eval.py. GT = voxelized YCB meshes (computed in
ho3d_metrics_cm.py). Per sequence: 8 frames -> build_cond -> stage1 coords + amodal
mask -> stage2 colored mesh/gs. Writes to ho3d_eval_vis/:
  <seq>_<obj>_mask.png  _stage1_voxel.ply  _stage2_mesh.glb  _stage2_gs.ply

  CUDA_VISIBLE_DEVICES=<free> python ho3d_eval.py [--ckpt ...] [--views 8] [--no_stage2]
  python ho3d_metrics_cm.py                       # metric-scale CD(cm) / F@5% / F@10%

HO3D objects are real, hand-held, and the YCB canonical frame != SAM3D canonical
frame, so the shape metrics are ALIGNMENT-INVARIANT (pred matched to GT over 24 axis
rotations). metrics_bestrot / norm_coords / _unit / _ROT24 / voxelize_ycb here are
shared with the HOT3D scripts (hot3d_eval.py, hot3d_metrics_cm.py).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import tools as _tools  # noqa: F401  (puts repo root + core/ on sys.path)
import os, sys, glob, argparse, itertools, pickle, re
import numpy as np

HO3D = "/path/to/processed_ho3d"
YCB = "/path/to/YCB_Video_Models/models"
VIS = "ho3d_eval_vis"
GRID = 64


# ---------------------------------------------------------------------------
# metrics (alignment-invariant via 24 axis rotations; both clouds unit-normalized)
# shared with hot3d_eval.py / hot3d_metrics_cm.py
# ---------------------------------------------------------------------------
def norm_coords(xyz, fill=0.85):
    x = np.asarray(xyz).astype(np.float32)
    if len(x) == 0:
        return x.astype(int)
    x = x - x.mean(0, keepdims=True)
    ext = float((x.max(0) - x.min(0)).max()) or 1.0
    x = x / ext * (GRID * fill) + (GRID - 1) / 2.0
    return np.unique(np.clip(np.round(x).astype(int), 0, GRID - 1), axis=0)


def _rotations24():
    mats = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3), np.float32)
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for s in itertools.product((1.0, -1.0), repeat=3):
            R = np.diag(s).astype(np.float32) @ P
            if abs(np.linalg.det(R) - 1.0) < 1e-5:
                mats.append(R)
    return mats


_ROT24 = _rotations24()


def _unit(xyz):
    x = np.asarray(xyz).astype(np.float32)
    x = x - x.mean(0, keepdims=True)
    ext = float((x.max(0) - x.min(0)).max()) or 1.0
    return x / ext                       # ~[-0.5,0.5], shape+rotation preserved


def metrics_bestrot(pred, gt, tau=0.04):
    """Return dict: iou (best over 24 rots), chamfer+fscore at the IoU-best rot.
    Shared with hot3d_eval.py."""
    if len(pred) == 0 or len(gt) == 0:
        return {"iou": 0.0, "chamfer": float("nan"), "fscore": 0.0, "pred_vox": int(len(pred))}
    from scipy.spatial import cKDTree
    g_set = set(map(tuple, norm_coords(gt)))
    gu = _unit(gt)
    pu0 = _unit(pred)
    gt_tree = cKDTree(gu)
    best = {"iou": -1.0}
    for R in _ROT24:
        pr = pu0 @ R.T
        p_set = set(map(tuple, norm_coords(pr)))
        u = len(p_set | g_set)
        iou = (len(p_set & g_set) / u) if u else 0.0
        if iou > best["iou"]:
            p_tree = cKDTree(pr)
            d_pg, _ = gt_tree.query(pr)      # pred->gt
            d_gp, _ = p_tree.query(gu)       # gt->pred
            best = {"iou": iou,
                    "chamfer": float(d_pg.mean() + d_gp.mean()),
                    "fscore": float(2 * (d_pg < tau).mean() * (d_gp < tau).mean() /
                                    max(1e-9, (d_pg < tau).mean() + (d_gp < tau).mean())),
                    "pred_vox": int(len(pred))}
    return best


def voxelize_ycb(obj_path, n=300000):
    """Voxelize a YCB mesh into a GRID^3 occupancy coord set (GT shape). Shared with metrics."""
    import trimesh
    m = trimesh.load(obj_path, process=False, force="mesh")
    v = m.vertices.astype(np.float64)
    c = (v.max(0) + v.min(0)) / 2.0
    ext = (v.max(0) - v.min(0)).max()
    pts, _ = trimesh.sample.sample_surface(m, n)
    pts = np.concatenate([pts, v], 0)
    p = (pts - c) / ext
    g = np.clip(np.round((p + 0.5) * GRID - 0.5).astype(int), 0, GRID - 1)
    return np.unique(g, axis=0).astype(np.int32)


# ---------------------------------------------------------------------------
# raw processed_ho3d parsing (no freeze step — read the dataset directly)
# ---------------------------------------------------------------------------
def ho3d_seqs():
    return sorted(glob.glob(os.path.join(HO3D, "*", "processed")))


def obj_name_of(sp):
    """objName for a sequence, from its first annotation pkl."""
    annos = sorted(glob.glob(os.path.join(sp, "anno", "*.pkl")))
    return pickle.load(open(annos[0], "rb"))["objName"]


def load_seq(sp, K):
    """K evenly-spaced frames of a processed_ho3d sequence: RGB + binarized obj/hand masks."""
    from PIL import Image
    annos = sorted(glob.glob(os.path.join(sp, "anno", "*.pkl")))
    fids = [re.search(r"(\d+)\.pkl", a).group(1) for a in annos]
    k = min(K, len(fids))
    sel = np.linspace(0, len(fids) - 1, k).round().astype(int)
    sel_ids = [fids[i] for i in sel]
    imgs, objs, hands = [], [], []
    for fid in sel_ids:
        imgs.append(np.array(Image.open(f"{sp}/images/{fid}.png").convert("RGB")).transpose(2, 0, 1))
        objs.append((np.array(Image.open(f"{sp}/object_masks/{fid}.png").convert("L")) > 127).astype(np.uint8))
        hands.append((np.array(Image.open(f"{sp}/hand_masks/{fid}.png").convert("L")) > 127).astype(np.uint8))
    return np.stack(imgs).astype(np.uint8), np.stack(objs), np.stack(hands), len(sel_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/forehoi.ckpt")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--out_dir", default=VIS)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--slat_steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_stage2", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    sys.path.append("core")
    from tools.pipeline import build_full_pipeline, save_mask_png
    import tools.mask_eval_common as C
    dev = "cuda"; K = args.views
    os.makedirs(args.out_dir, exist_ok=True)

    pipeline, ckpt = build_full_pipeline(args.ckpt, dev)
    print(f"[ho3d] pipeline built; ckpt={os.path.basename(ckpt)} views={K}", flush=True)

    import time
    times = []
    for sp in ho3d_seqs():
        seq = sp.split("/")[-2]
        img_u8, obj_u8, hand_u8, n = load_seq(sp, K)
        objn = obj_name_of(sp)
        image = torch.from_numpy(img_u8.astype(np.float32) / 255.0)
        obj = torch.from_numpy(obj_u8.astype(np.float32)).unsqueeze(1)
        hand = torch.from_numpy(hand_u8.astype(np.float32)).unsqueeze(1)

        # stage 1: shape latent + amodal mask (timed)
        torch.cuda.synchronize(); _t0 = time.perf_counter()
        cond, raw_ctx = C.build_cond(pipeline, image, obj, hand, dev)
        z_shape, z_mask = C.joint_sample(pipeline, cond, raw_ctx, n, steps=args.steps,
                                         cfg=args.cfg, device=dev, seed=args.seed)
        coords = C.decode_shape(pipeline, z_shape)
        torch.cuda.synchronize(); t_s1 = time.perf_counter() - _t0
        tag = f"{seq}_{objn}"
        save_mask_png(os.path.join(args.out_dir, f"{tag}_mask.png"), z_mask)
        C.export_voxels(os.path.join(args.out_dir, f"{tag}_stage1_voxel.ply"), coords)

        # stage 2: SAM3D SLAT -> vertex-colored mesh + gaussians (single ref view, like the app; timed)
        t_s2 = 0.0
        if not args.no_stage2 and coords is not None and len(coords):
            ref_rgb = np.ascontiguousarray(img_u8[0].transpose(1, 2, 0))
            ref_alpha = (obj_u8[0].astype(np.uint8) * 255)
            rgba = np.concatenate([ref_rgb, ref_alpha[..., None]], axis=-1).astype(np.uint8)
            slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
            c4 = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32),
                            torch.from_numpy(coords).int()], dim=1).to(dev)
            torch.cuda.synchronize(); _t2 = time.perf_counter()
            glb = None
            with torch.no_grad():
                slat = pipeline.sample_slat(slat_input, c4, inference_steps=args.slat_steps)
                outs = pipeline.decode_slat(slat, ["gaussian", "mesh"])
                gs = outs["gaussian"][0]
                try:
                    glb = pipeline.postprocess_slat_output(outs, False, False, True).get("glb")  # use_vertex_color
                except Exception as e:
                    print(f"[ho3d] {seq}: stage-2 GLB failed ({e})", flush=True)
            torch.cuda.synchronize(); t_s2 = time.perf_counter() - _t2
            gs.save_ply(os.path.join(args.out_dir, f"{tag}_stage2_gs.ply"))
            if glb is not None:
                glb.export(os.path.join(args.out_dir, f"{tag}_stage2_mesh.glb"))
            torch.cuda.empty_cache()
        times.append((t_s1, t_s2))
        print(f"[ho3d {K}v] {seq:22} {objn:22} vox={len(coords)} "
              f"stage1={t_s1:.1f}s stage2={t_s2:.1f}s total={t_s1 + t_s2:.1f}s -> {args.out_dir}/", flush=True)
        torch.cuda.empty_cache()
    if times:
        m1 = float(np.mean([t[0] for t in times])); m2 = float(np.mean([t[1] for t in times]))
        print(f"[ho3d] MEAN per-seq runtime  stage1={m1:.1f}s  stage2={m2:.1f}s  total={m1 + m2:.1f}s  (n={len(times)})", flush=True)
    print(f"[ho3d] DONE -> {args.out_dir}/  (run ho3d_metrics_cm.py for metric-scale CD / F@5% / F@10%)", flush=True)


if __name__ == "__main__":
    main()
