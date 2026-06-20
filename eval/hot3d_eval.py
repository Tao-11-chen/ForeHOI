import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import tools as _tools  # noqa: F401  (puts repo root + core/ on sys.path)

"""Test the v3_ft4 multi-view model on the HOT3D dataset (processed_hot3d).

HOT3D layout per sequence:
  img/214-1_camera-rgb/<ts>.png   1408x1408 RGB (main stream; SLAM streams ignored)
  object_mask/<ts>.png            instance-colored mask (binarized >0)
  hand_mask/<ts>.png              hand mask (often empty -> zeros)
  object_mesh/<ts>_<obj>_<id>.obj per-frame POSED object mesh (GT shape)

Per sequence: K=8 RGB views + binarized obj/hand masks -> build_cond (object-centric
518 crop, same path as HO3D/saved_tests) -> stage1 coords + amodal mask -> stage2
colored mesh/gs. GT shape = TRELLIS-voxelized posed object mesh (alignment-invariant
best-of-24-rot IoU, so the arbitrary world pose is handled). Writes to hot3d_eval_vis/:
  <seq>_mask.png  _stage1_voxel.ply  _voxproj.png  _stage2_mesh.glb  _stage2_gs.ply
and hot3d_eval_metrics.json.

  CUDA_VISIBLE_DEVICES=<free> python hot3d_eval.py [--ckpt ...] [--views 8] [--no_stage2]
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import sys, glob, argparse, time
sys.path.append("core")
import numpy as np
import torch
from PIL import Image
from tools.pipeline import build_full_pipeline, save_mask_png
import tools.mask_eval_common as C
from ho3d_eval import metrics_bestrot

ROOT = "/path/to/processed_hot3d"
RGB_STREAM = "214-1_camera-rgb"
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


def _binmask(path, like):
    if path and os.path.exists(path):
        return (np.array(Image.open(path).convert("L")) > 0).astype(np.uint8)
    return np.zeros_like(like)


def load_seq(seq_dir, K):
    rgb_dir = os.path.join(seq_dir, "img", RGB_STREAM)
    fids = [os.path.splitext(f)[0] for f in sorted(os.listdir(rgb_dir)) if f.endswith(".png")]
    sel = sorted(set(np.linspace(0, len(fids) - 1, min(K, len(fids))).round().astype(int).tolist()))
    sel_ids = [fids[i] for i in sel]
    imgs, objs, hands = [], [], []
    for fid in sel_ids:
        imgs.append(np.array(Image.open(os.path.join(rgb_dir, f"{fid}.png")).convert("RGB")).transpose(2, 0, 1))
        o = (np.array(Image.open(os.path.join(seq_dir, "object_mask", f"{fid}.png")).convert("L")) > 0).astype(np.uint8)
        objs.append(o)
        hands.append(_binmask(os.path.join(seq_dir, "hand_mask", f"{fid}.png"), o))
    image = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0)
    obj = torch.from_numpy(np.stack(objs).astype(np.float32)).unsqueeze(1)
    hand = torch.from_numpy(np.stack(hands).astype(np.float32)).unsqueeze(1)
    return image, obj, hand, len(sel_ids), sel_ids


def gt_from_mesh(seq_dir):
    ms = sorted(glob.glob(os.path.join(seq_dir, "object_mesh", "*.obj")))
    mid = ms[len(ms) // 2]
    obj_name = os.path.basename(mid).split("_")[1] if "_" in os.path.basename(mid) else "obj"
    import trimesh
    m = trimesh.load(mid, process=False, force="mesh")
    g = trellis_voxelize(np.asarray(m.vertices), np.asarray(m.faces))
    return g, obj_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/forehoi.ckpt")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--out_dir", default="hot3d_eval_vis")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--slat_steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_stage2", action="store_true")
    args = ap.parse_args()
    dev = "cuda"; K = args.views
    os.makedirs(args.out_dir, exist_ok=True)

    pipeline, ckpt = build_full_pipeline(args.ckpt, dev)
    print(f"[hot3d] pipeline built; ckpt={os.path.basename(ckpt)} views={K}", flush=True)

    seqs = sorted([d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d)])
    results = []
    for seq_dir in seqs:
        seq = os.path.basename(seq_dir)
        image, obj, hand, n, sel_ids = load_seq(seq_dir, K)
        gt, obj_name = gt_from_mesh(seq_dir)

        # stage 1: shape latent + amodal mask (timed)
        torch.cuda.synchronize(); _t0 = time.perf_counter()
        cond, raw_ctx = C.build_cond(pipeline, image, obj, hand, dev)
        z_shape, z_mask = C.joint_sample(pipeline, cond, raw_ctx, n, steps=args.steps,
                                         cfg=args.cfg, device=dev, seed=args.seed)
        coords = C.decode_shape(pipeline, z_shape)
        torch.cuda.synchronize(); t_s1 = time.perf_counter() - _t0
        m = metrics_bestrot(coords, gt)

        tag = f"{seq}_{obj_name}"
        save_mask_png(os.path.join(args.out_dir, f"{tag}_mask.png"), z_mask)
        C.export_voxels(os.path.join(args.out_dir, f"{tag}_stage1_voxel.ply"), coords)
        Image.fromarray(C.voxel_projections(coords), mode="L").save(
            os.path.join(args.out_dir, f"{tag}_voxproj.png"))

        # stage 2: SAM3D SLAT -> vertex-colored mesh + gaussians (timed)
        t_s2 = 0.0
        if not args.no_stage2:
            ref_rgb = (image[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            ref_alpha = (obj[0, 0].numpy() > 0.5).astype(np.uint8) * 255
            rgba = np.concatenate([ref_rgb, ref_alpha[..., None]], axis=-1)
            slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
            coords4 = torch.cat([torch.zeros(len(coords), 1, dtype=torch.int32),
                                 torch.from_numpy(coords).int()], dim=1).to(dev)
            torch.cuda.synchronize(); _t2 = time.perf_counter()
            glb = None
            with torch.no_grad():
                slat = pipeline.sample_slat(slat_input, coords4, inference_steps=args.slat_steps)
                outs = pipeline.decode_slat(slat, ["gaussian", "mesh"])
                gs = outs["gaussian"][0]
                try:
                    glb = pipeline.postprocess_slat_output(outs, False, False, True).get("glb")
                except Exception as e:
                    print(f"[hot3d] {seq}: GLB export failed ({e})", flush=True)
            torch.cuda.synchronize(); t_s2 = time.perf_counter() - _t2
            gs.save_ply(os.path.join(args.out_dir, f"{tag}_stage2_gs.ply"))
            if glb is not None:
                glb.export(os.path.join(args.out_dir, f"{tag}_stage2_mesh.glb"))
            torch.cuda.empty_cache()

        r = {"seq": seq, "object": obj_name, "views": n,
             "iou": round(m["iou"], 4), "fscore": round(m["fscore"], 4),
             "chamfer": round(m["chamfer"], 4), "pred_vox": m["pred_vox"], "gt_vox": int(len(gt)),
             "stage1_s": round(t_s1, 2), "stage2_s": round(t_s2, 2), "total_s": round(t_s1 + t_s2, 2)}
        results.append(r)
        print(f"[hot3d {K}v] {seq:30} {obj_name:12} IoU={r['iou']:.3f} F={r['fscore']:.3f} "
              f"CD={r['chamfer']:.3f}  stage1={r['stage1_s']:.1f}s stage2={r['stage2_s']:.1f}s "
              f"total={r['total_s']:.1f}s (pred {r['pred_vox']} / gt {r['gt_vox']})", flush=True)

    if results:
        miou = float(np.mean([r["iou"] for r in results]))
        mf = float(np.mean([r["fscore"] for r in results]))
        mcd = float(np.nanmean([r["chamfer"] for r in results]))
        ms1 = float(np.mean([r["stage1_s"] for r in results]))
        ms2 = float(np.mean([r["stage2_s"] for r in results]))
        print(f"\n[hot3d {K}v] n={len(results)} mean IoU={miou:.4f} F={mf:.4f} CD={mcd:.4f} "
              f"| runtime stage1={ms1:.1f}s stage2={ms2:.1f}s total={ms1 + ms2:.1f}s  "
              f"(metric-scale CD/F@5%/F@10% -> run hot3d_metrics_cm.py)", flush=True)
    print("[hot3d] DONE | HOT3D_DONE", flush=True)


if __name__ == "__main__":
    main()
