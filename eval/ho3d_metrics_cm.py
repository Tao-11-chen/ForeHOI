import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import tools as _tools  # noqa: F401  (puts repo root + core/ on sys.path)

#!/usr/bin/env python3
"""Metric-scale HO3D reconstruction metrics from the saved stage-1 voxel PLYs (no GPU):
  CD(cm)  = 0.5*(mean_{p->g} + mean_{g->p}) nearest-neighbour L2 distance, in cm
  F@5%    = F-score at tau = 5%  of the GT bbox diagonal (cm)
  F@10%   = F-score at tau = 10% of the GT bbox diagonal (cm)
            precision = frac(pred->gt < tau), recall = frac(gt->pred < tau)

Reads ho3d_eval_vis/<seq>_<obj>_stage1_voxel.ply (written by ho3d_eval.py) and the
voxelized YCB GT (computed on the fly from the raw processed_ho3d). Each prediction is
canonical (scale/rotation-free vs the YCB GT frame), so it is rigidly aligned to GT
(best-of-24 rotation init -> ICP refine), then both clouds are placed at the GT's REAL
metric size (YCB mesh bbox extent, metres -> cm).

  python ho3d_metrics_cm.py
"""
import os, json, glob, itertools
import numpy as np
from scipy.spatial import cKDTree
from ho3d_eval import HO3D, YCB, voxelize_ycb, obj_name_of, ho3d_seqs

VIS = "ho3d_eval_vis"
GRID = 64


def ext_cm(obj):
    """GT object real max bbox extent (cm) from the YCB mesh."""
    import trimesh
    m = trimesh.load(os.path.join(YCB, obj, "textured_simple.obj"), process=False, force="mesh")
    v = m.vertices
    return float((v.max(0) - v.min(0)).max()) * 100.0


def read_ply_xyz(path):
    lines = open(path).read().splitlines()
    i = lines.index("end_header") + 1
    return np.array([[float(t) for t in ln.split()[:3]] for ln in lines[i:] if ln.strip()], np.float32)


def unit(xyz):
    x = np.asarray(xyz).astype(np.float64)
    x = x - x.mean(0, keepdims=True)
    return x / ((x.max(0) - x.min(0)).max() or 1.0)


def _rot24():
    out = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for s in itertools.product((1.0, -1.0), repeat=3):
            R = np.diag(s).astype(float) @ P
            if abs(np.linalg.det(R) - 1.0) < 1e-6:
                out.append(R)
    return out


ROT24 = _rot24()


def chamfer_dirs(a, b):
    """mean nearest-neighbour dist a->b and b->a (same units as inputs)."""
    dab, _ = cKDTree(b).query(a)
    dba, _ = cKDTree(a).query(b)
    return dab, dba


def icp(src, dst, iters=20):
    """rigid point-to-point ICP (rotation+translation, no scale); pure numpy."""
    tree = cKDTree(dst)
    T = src.copy()
    for _ in range(iters):
        _, idx = tree.query(T)
        corr = dst[idx]
        mu_s, mu_d = T.mean(0), corr.mean(0)
        H = (T - mu_s).T @ (corr - mu_d)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        T = (T - mu_s) @ R.T + mu_d
    return T


def align_unit(pred, gt_u):
    """pred coords -> unit cloud rigidly aligned to gt unit cloud (best-24 + ICP)."""
    pu = unit(pred)
    # rank 24 rotations by initial chamfer, refine top-3 with ICP, keep best
    scored = []
    for R in ROT24:
        pr = pu @ R.T
        dab, dba = chamfer_dirs(pr, gt_u)
        scored.append((dab.mean() + dba.mean(), R))
    scored.sort(key=lambda t: t[0])
    best_cloud, best_cd = None, 1e9
    for _, R in scored[:3]:
        aln = icp(pu @ R.T, gt_u)
        dab, dba = chamfer_dirs(aln, gt_u)
        cd = dab.mean() + dba.mean()
        if cd < best_cd:
            best_cd, best_cloud = cd, aln
    return best_cloud


def metrics_one(pred, gt, ecm):
    if len(pred) == 0 or len(gt) == 0:
        return None
    gt_u = unit(gt)
    pred_u = align_unit(pred, gt_u)
    gt_cm = gt_u * ecm
    pred_cm = pred_u * ecm
    diag = float(np.linalg.norm(gt_cm.max(0) - gt_cm.min(0)))
    dab, dba = chamfer_dirs(pred_cm, gt_cm)            # cm
    cd = 0.5 * (dab.mean() + dba.mean())
    out = {"cd_cm": float(cd), "diag_cm": diag}
    for pct in (0.05, 0.10):
        tau = pct * diag
        P = float((dab < tau).mean()); Rc = float((dba < tau).mean())
        out[f"f{int(pct*100)}"] = float(2 * P * Rc / (P + Rc)) if (P + Rc) > 0 else 0.0
    return out


def main():
    ecm_cache = {}
    rows = []
    for sp in ho3d_seqs():
        seq = sp.split("/")[-2]
        obj = obj_name_of(sp)
        ply = os.path.join(VIS, f"{seq}_{obj}_stage1_voxel.ply")
        if not os.path.exists(ply):
            print(f"[ho3d] skip {seq}: no {os.path.basename(ply)} (run ho3d_eval.py first)"); continue
        gt = voxelize_ycb(os.path.join(YCB, obj, "textured_simple.obj"))
        if obj not in ecm_cache:
            ecm_cache[obj] = ext_cm(obj)
        ecm = ecm_cache[obj]
        m = metrics_one(read_ply_xyz(ply), gt, ecm)
        if not m:
            continue
        m.update({"seq": seq, "object": obj, "ext_cm": ecm})
        rows.append(m)
        print(f"[ho3d] {seq:24} {obj:22} ext={ecm:5.1f}cm  CD={m['cd_cm']:.3f}cm  "
              f"F@5%={m['f5']:.3f}  F@10%={m['f10']:.3f}", flush=True)
    if rows:
        cd = float(np.mean([r["cd_cm"] for r in rows]))
        f5 = float(np.mean([r["f5"] for r in rows]))
        f10 = float(np.mean([r["f10"] for r in rows]))
        print(f"\n[ho3d MEAN n={len(rows)}]  CD={cd:.3f} cm   F@5%={f5:.4f}   F@10%={f10:.4f}")
        json.dump({"n": len(rows), "mean": {"cd_cm": cd, "f5": f5, "f10": f10}, "per_seq": rows},
                  open("ho3d_metrics_cm.json", "w"), indent=2)
        print("  -> ho3d_metrics_cm.json")


if __name__ == "__main__":
    main()
