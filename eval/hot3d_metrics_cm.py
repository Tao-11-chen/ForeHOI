import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import tools as _tools  # noqa: F401  (puts repo root + core/ on sys.path)

"""Recompute HOT3D metrics in metric units from the saved stage-1 voxel PLYs (no GPU):
  - CD (cm): best-rot chamfer (sum of directional means, _unit space) x GT real max-extent(cm)
  - F@5%, F@10%: F-score at tau=0.05 / 0.10 of the object max-extent (scale-invariant)
Best rotation = argmax norm_coords IoU (same as metrics_bestrot).

  python hot3d_metrics_cm.py
"""
import os, glob, json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from ho3d_eval import norm_coords, _unit, _ROT24

ROOT = "/path/to/processed_hot3d"
VIS = "hot3d_eval_vis"
GRID = 64


def trellis_voxelize(verts, faces, grid=GRID):
    import open3d as o3d
    v = np.asarray(verts, np.float64)
    c = (v.min(0) + v.max(0)) / 2.0; s = (v.max(0) - v.min(0)).max()
    vn = np.clip((v - c) / s, -0.5 + 1e-6, 0.5 - 1e-6)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vn),
                                   o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        om, voxel_size=1.0 / grid, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    g = np.array([x.grid_index for x in vg.get_voxels()])
    return np.unique(np.clip(g, 0, grid - 1).astype(np.int64), axis=0).astype(np.int32)


def read_ply_xyz(path):
    lines = open(path).read().splitlines()
    i = lines.index("end_header") + 1
    return np.array([[float(t) for t in ln.split()[:3]] for ln in lines[i:] if ln.strip()], np.float32)


def metrics_cm(pred, gt, ext_cm, taus=(0.05, 0.10)):
    g_set = set(map(tuple, norm_coords(gt)))
    gu = _unit(gt); pu0 = _unit(pred)
    gt_tree = cKDTree(gu)
    best = {"iou": -1.0}
    for R in _ROT24:
        pr = pu0 @ R.T
        p_set = set(map(tuple, norm_coords(pr)))
        u = len(p_set | g_set)
        iou = (len(p_set & g_set) / u) if u else 0.0
        if iou > best["iou"]:
            d_pg, _ = gt_tree.query(pr)               # pred->gt (unit-extent)
            d_gp, _ = cKDTree(pr).query(gu)            # gt->pred
            f = {}
            for t in taus:
                pp = (d_pg < t).mean(); rr = (d_gp < t).mean()
                f[t] = float(2 * pp * rr / max(1e-9, pp + rr))
            best = {"iou": iou,
                    "cd_cm": float((d_pg.mean() + d_gp.mean()) * ext_cm),
                    "f": f}
    return best


def main():
    rows = []
    for seq_dir in sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d)):
        seq = os.path.basename(seq_dir)
        ms = sorted(glob.glob(os.path.join(seq_dir, "object_mesh", "*.obj")))
        mid = ms[len(ms) // 2]
        obj_name = os.path.basename(mid).split("_")[1]
        m = trimesh.load(mid, process=False, force="mesh")
        v = np.asarray(m.vertices, np.float64)
        ext_cm = float((v.max(0) - v.min(0)).max()) * 100.0   # mesh is in metres
        gt = trellis_voxelize(v, np.asarray(m.faces))
        ply = os.path.join(VIS, f"{seq}_{obj_name}_stage1_voxel.ply")
        if not os.path.exists(ply):
            print(f"[skip] no ply for {seq} ({obj_name})"); continue
        pred = read_ply_xyz(ply)
        r = metrics_cm(pred, gt, ext_cm)
        rows.append({"seq": seq, "object": obj_name, "ext_cm": round(ext_cm, 2),
                     "iou": round(r["iou"], 4), "cd_cm": round(r["cd_cm"], 4),
                     "f5": round(r["f"][0.05], 4), "f10": round(r["f"][0.10], 4)})
        print(f"[hot3d] {seq:30} {obj_name:11} ext={ext_cm:5.1f}cm  IoU={r['iou']:.3f}  "
              f"CD={r['cd_cm']:.3f}cm  F@5%={r['f'][0.05]:.3f}  F@10%={r['f'][0.10]:.3f}", flush=True)

    if rows:
        cd = float(np.mean([r["cd_cm"] for r in rows]))
        f5 = float(np.mean([r["f5"] for r in rows]))
        f10 = float(np.mean([r["f10"] for r in rows]))
        print(f"\n[hot3d MEAN n={len(rows)}]  CD={cd:.3f} cm   F@5%={f5:.4f}   F@10%={f10:.4f}", flush=True)
        json.dump({"n": len(rows), "mean": {"cd_cm": cd, "f5": f5, "f10": f10}, "per_seq": rows},
                  open("hot3d_metrics_cm.json", "w"), indent=2)
        print("  -> hot3d_metrics_cm.json")


if __name__ == "__main__":
    main()
