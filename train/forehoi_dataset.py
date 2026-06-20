"""ForeHOI Multi-View Dataset — map-style Dataset over all tar files."""
import sys
import os
import json
import random
import tarfile
import io
import pickle
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader

try:
    import OpenEXR  # for rendered-depth (.depth.exr) loading — depth conditioning
except Exception:  # pragma: no cover - depth is optional
    OpenEXR = None

if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray
    sys.modules['numpy._core.numeric'] = np.core.numeric
    sys.modules['numpy._core._multiarray_umath'] = np.core.multiarray


# Axis-aligned rotation that maps the ForeHoi native object frame into the
# pretrained SAM-3D-Objects canonical frame. Recovered by running demo_ori.py
# + find_orient_rotation.py on one sample (voxel IoU against the pretrained
# ckpt's stage-1 coords jumped from 0.014 at identity to 0.39 under this R).
# R is an involution (R @ R = I), det = +1.
# NOTE: this is SAM-3D-Objects' canonical frame and DIFFERS from ReconViaGen's
# (TRELLIS) GT_TO_PRED_R — keep this one, since we use sam3d's ss_encoder/decoder
# /DiT base. The full_mask / hand_mask added below are 2D (frame-agnostic).
GT_TO_PRED_R = torch.tensor(
    [[-1., 0., 0.],
     [ 0., 0., 1.],
     [ 0., 1., 0.]],
    dtype=torch.float32,
)
_GRID_CENTER = (64 - 1) / 2.0


class ForeHOIDataset(Dataset):
    """Map-style dataset that loads multi-view data from ForeHOI tar files.

    Each tar file contains one 3D object with multiple views/frames.
    For each sample we randomly pick `num_views` frames (preferring distinct views).
    Output format matches MVDataset for training pipeline compatibility.
    """

    def __init__(self, data_json: str, num_views: int = 4,
                 test_mode: bool = False, cache_dir: str = None):
        with open(data_json, 'r') as f:
            all_tar_paths = json.load(f)

        if test_mode:
            self.data = all_tar_paths[-1000:]
        else:
            self.data = all_tar_paths[:-1000]

        self.num_views = num_views
        self.cache_dir = cache_dir

        print(f"ForeHOIDataset: {len(self.data)} tar files "
              f"({'test' if test_mode else 'train'})")

    # ------------------------------------------------------------------
    # Index building / caching  (adapted from forehoi_loader.py)
    # ------------------------------------------------------------------
    def _build_index(self, tar_path: str) -> List[Dict]:
        """Build a flat list of frame entries for one tar file.

        Each entry: {key, view_id, frame_id, files: {data_type: {offset, size}}}
        """
        entries: Dict[str, Dict] = {}  # base_key -> entry

        with tarfile.open(tar_path, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue

                full_name = member.name
                if full_name.endswith('.rgb.webp'):
                    data_type, trim = 'rgb', 9
                elif full_name.endswith('.obj_mask.webp'):
                    data_type, trim = 'obj_mask', 14
                elif full_name.endswith('.hand_mask.webp'):
                    data_type, trim = 'hand_mask', 15
                elif full_name.endswith('.meta.json'):
                    data_type, trim = 'meta', 10
                else:
                    continue

                base_key = full_name[:-trim]
                parts = base_key.split('/')
                if len(parts) != 3:
                    continue
                _, view_id, frame_id = parts

                if base_key not in entries:
                    entries[base_key] = {
                        "key": base_key,
                        "view_id": view_id,
                        "frame_id": frame_id,
                        "files": {}
                    }

                data_offset = (member.offset_data
                               if hasattr(member, 'offset_data') and member.offset_data is not None
                               else member.offset + 512)
                entries[base_key]["files"][data_type] = {
                    "offset": data_offset,
                    "size": member.size
                }

        result = sorted(entries.values(), key=lambda x: (x["view_id"], x["frame_id"]))
        return result

    def _get_index(self, tar_path: str) -> List[Dict]:
        """Load cached index or build + cache it."""
        tar_p = Path(tar_path)
        cache_parent = Path(self.cache_dir) if self.cache_dir else tar_p.parent
        cache_path = cache_parent / f"{tar_p.stem}.index.pkl"

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        index = self._build_index(tar_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(index, f)
        except OSError:
            pass  # non-fatal if cache dir is read-only
        return index

    # ------------------------------------------------------------------
    # full_mask.tar (sibling of data.tar, written by precompute_full_mask.py).
    # Maps base_key "<sample>/<view>/<frame>" -> {offset, size} so a single
    # .full_mask.webp can be read by byte offset, like the data.tar index.
    # ------------------------------------------------------------------
    def _build_fm_index(self, fm_tar_path: str) -> Dict[str, Dict]:
        entries: Dict[str, Dict] = {}
        suffix = '.full_mask.webp'
        with tarfile.open(fm_tar_path, "r") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(suffix):
                    continue
                base_key = member.name[:-len(suffix)]
                data_offset = (member.offset_data
                               if getattr(member, 'offset_data', None) is not None
                               else member.offset + 512)
                entries[base_key] = {"offset": data_offset, "size": member.size}
        return entries

    def _get_fm_index(self, fm_tar_path: str) -> Dict[str, Dict]:
        fm_p = Path(fm_tar_path)
        cache_parent = Path(self.cache_dir) if self.cache_dir else fm_p.parent
        cache_path = cache_parent / f"{fm_p.stem}.index.pkl"  # full_mask.index.pkl

        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        index = self._build_fm_index(fm_tar_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(index, f)
        except OSError:
            pass
        return index

    # ------------------------------------------------------------------
    # depth.tar (sibling of data.tar; rendered perpendicular z-depth, .depth.exr).
    # Same byte-offset index trick as full_mask.tar. Used by depth conditioning.
    # ------------------------------------------------------------------
    def _build_depth_index(self, depth_tar_path: str) -> Dict[str, Dict]:
        entries: Dict[str, Dict] = {}
        suffix = '.depth.exr'
        with tarfile.open(depth_tar_path, "r") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(suffix):
                    continue
                base_key = member.name[:-len(suffix)]
                data_offset = (member.offset_data
                               if getattr(member, 'offset_data', None) is not None
                               else member.offset + 512)
                entries[base_key] = {"offset": data_offset, "size": member.size}
        return entries

    def _get_depth_index(self, depth_tar_path: str) -> Dict[str, Dict]:
        dp = Path(depth_tar_path)
        cache_parent = Path(self.cache_dir) if self.cache_dir else dp.parent
        cache_path = cache_parent / f"{dp.stem}.index.pkl"  # depth.index.pkl
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        index = self._build_depth_index(depth_tar_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(index, f)
        except OSError:
            pass
        return index

    @staticmethod
    def _decode_exr_depth(data: bytes) -> np.ndarray:
        """Decode .depth.exr bytes -> (H,W) float32 (channel 'V')."""
        with tempfile.NamedTemporaryFile(suffix=".exr", delete=False) as f:
            f.write(data); tmp = f.name
        try:
            with OpenEXR.File(tmp) as exr:
                part = exr.parts[0]
                ch = "V" if "V" in part.channels else next(iter(part.channels))
                return np.asarray(part.channels[ch].pixels, dtype=np.float32)
        finally:
            os.unlink(tmp)

    # ------------------------------------------------------------------
    # Tar I/O helpers  (adapted from forehoi_loader.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _read_bytes(tar, offset_info: Dict) -> bytes:
        tar.fileobj.seek(offset_info["offset"])
        return tar.fileobj.read(offset_info["size"])

    def _decode_entry(self, tar, entry: Dict) -> Dict:
        result = {"view_id": entry["view_id"], "frame_id": entry["frame_id"]}

        for data_type, offset_info in entry["files"].items():
            data = self._read_bytes(tar, offset_info)
            if data_type == 'rgb':
                img = Image.open(io.BytesIO(data)).convert("RGB")
                result['rgb'] = np.array(img)
            elif data_type == 'obj_mask':
                img = Image.open(io.BytesIO(data)).convert("L")
                result['obj_mask'] = np.array(img)
            elif data_type == 'hand_mask':
                img = Image.open(io.BytesIO(data)).convert("L")
                result['hand_mask'] = np.array(img)
            elif data_type == 'meta':
                result['meta'] = json.loads(data.decode('utf-8'))

        return result

    # ------------------------------------------------------------------
    # Core dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        while True:
            try:
                return self._load_item(idx)
            except Exception:
                idx = random.randint(0, len(self.data) - 1)

    def _load_item(self, idx):
        tar_path = self.data[idx]
        index = self._get_index(tar_path)

        # --- sample num_views frames, preferring distinct views ---
        views_available = {}
        for entry in index:
            views_available.setdefault(entry["view_id"], []).append(entry)

        view_ids = list(views_available.keys())
        if len(view_ids) >= self.num_views:
            chosen_views = random.sample(view_ids, self.num_views)
            selected = [random.choice(views_available[v]) for v in chosen_views]
        else:
            selected = random.choices(index, k=self.num_views)

        # --- decode frames from tar ---
        imgs_list = []
        masks_list = []
        hand_masks_list = []
        w2c_list = []
        aov_list = []          # camera_angle_x per view (intrinsics for unprojection)

        with tarfile.open(tar_path, "r") as tar:
            for entry_info in selected:
                data = self._decode_entry(tar, entry_info)

                if 'rgb' not in data or 'obj_mask' not in data:
                    raise ValueError(f"Missing rgb/obj_mask for {entry_info['key']}")

                rgb = torch.from_numpy(data['rgb']).permute(2, 0, 1).float() / 255.0   # 3,H,W
                mask = torch.from_numpy(data['obj_mask']).float().unsqueeze(0) / 255.0  # 1,H,W
                mask = (mask > 0.5).float()

                # hand_mask (used by the raw/hand-aware DINO stream); zeros if absent
                if 'hand_mask' in data:
                    hand = torch.from_numpy(data['hand_mask']).float().unsqueeze(0) / 255.0
                    hand = (hand > 0.5).float()
                else:
                    hand = torch.zeros_like(mask)

                meta = data.get('meta', {})
                cam_pose = meta.get('camera_pose', np.eye(4).tolist())
                w2c_list.append(torch.tensor(cam_pose, dtype=torch.float32))
                aov_list.append(float(meta.get('camera_angle_x', 0.6981317)))

                imgs_list.append(rgb)
                masks_list.append(mask)
                hand_masks_list.append(hand)

        # --- complete (amodal) object mask from the sibling full_mask.tar ---
        # full_mask = depth_valid ∪ obj_mask  (precompute_full_mask.py). The
        # dual-branch denoises this per view. Falls back to the visible obj mask
        # when the sibling tar / entry is absent so training never crashes — that
        # view just carries no completion signal.
        tar_dir = os.path.dirname(tar_path)
        fm_path = os.path.join(tar_dir, 'full_mask.tar')
        full_masks_list = [m.clone() for m in masks_list]
        if os.path.exists(fm_path):
            try:
                fm_index = self._get_fm_index(fm_path)
                with tarfile.open(fm_path, "r") as fmt:
                    for i, entry_info in enumerate(selected):
                        oi = fm_index.get(entry_info["key"])
                        if oi is None:
                            continue  # keep obj-mask fallback for this view
                        data = self._read_bytes(fmt, oi)
                        fm_img = Image.open(io.BytesIO(data)).convert("L")
                        H_i, W_i = masks_list[i].shape[-2:]
                        if fm_img.size != (W_i, H_i):
                            fm_img = fm_img.resize((W_i, H_i), Image.NEAREST)
                        fm = torch.from_numpy(np.array(fm_img)).float().unsqueeze(0) / 255.0
                        full_masks_list[i] = (fm > 0.5).float()
            except Exception:
                full_masks_list = [m.clone() for m in masks_list]

        # --- rendered depth from sibling depth.tar (depth conditioning) ---
        # Perpendicular z-depth, 1e10 = background. Missing frame/tar -> all-1e10
        # (the pipeline unprojects that to all-NaN, i.e. an empty pointmap for the view).
        _, H0, W0 = imgs_list[0].shape
        depth_tar_path = os.path.join(tar_dir, 'depth.tar')
        depth_list = [torch.full((H0, W0), 1e10, dtype=torch.float32) for _ in selected]
        if OpenEXR is not None and os.path.exists(depth_tar_path):
            try:
                d_index = self._get_depth_index(depth_tar_path)
                with tarfile.open(depth_tar_path, "r") as dt:
                    for i, entry_info in enumerate(selected):
                        oi = d_index.get(entry_info["key"])
                        if oi is None:
                            continue
                        arr = self._decode_exr_depth(self._read_bytes(dt, oi))
                        if arr.shape == (H0, W0):
                            depth_list[i] = torch.from_numpy(arr).float()
            except Exception:
                pass

        V = len(imgs_list)
        _, H, W = imgs_list[0].shape

        imgs  = torch.stack(imgs_list, dim=0)   # V,3,H,W
        masks = torch.stack(masks_list, dim=0)   # V,1,H,W
        hand_masks = torch.stack(hand_masks_list, dim=0)  # V,1,H,W
        full_masks = torch.stack(full_masks_list, dim=0)  # V,1,H,W  (amodal silhouette)
        occ_masks = torch.zeros(V, 1, H, W, dtype=torch.float32)
        pointmaps = torch.zeros(V, 3, H, W, dtype=torch.float32)

        # first-view w2c — rotate so the camera is expressed relative to the
        # pretrained ckpt's canonical object frame rather than ForeHoi's.
        # w2c maps world -> camera, so post-multiplying the rotation block by
        # R.T rebases the "world" from ForeHoi's frame to the canonical frame.
        w2c_0 = w2c_list[0].clone()  # 4,4
        w2c_0[:3, :3] = w2c_0[:3, :3] @ GT_TO_PRED_R.T

        # global shape voxel grid from precomputed coords.npz
        tar_dir = os.path.dirname(tar_path)
        coords_path = os.path.join(tar_dir, 'coords.npz')
        if os.path.exists(coords_path):
            coords_data = np.load(coords_path)
            global_coord = torch.tensor(coords_data['coords']).to(torch.int32)  # N, 3

            # Rotate voxel indices into the pretrained canonical frame.
            coord_f = global_coord.float() - _GRID_CENTER
            coord_f = coord_f @ GT_TO_PRED_R.T
            coord_f = coord_f + _GRID_CENTER
            global_coord = coord_f.round().clamp(0, 63).to(torch.int32)

            global_ss = torch.zeros(64, 64, 64, dtype=torch.long)
            global_ss = global_ss.index_put_(
                (global_coord[:, 0], global_coord[:, 1], global_coord[:, 2]),
                torch.tensor(1, dtype=global_ss.dtype)
            )
            global_ss = global_ss[None].float()  # 1, 64, 64, 64
        else:
            global_ss = torch.zeros(1, 64, 64, 64, dtype=torch.float32)

        depth = torch.stack(depth_list, dim=0)                       # V, H, W (rendered z-depth)
        camera_angle_x = torch.tensor(aov_list, dtype=torch.float32) # V

        return dict(
            image=imgs,            # V, 3, H, W
            mask=masks,            # V, 1, H, W   (object mask, visible)
            hand_mask=hand_masks,  # V, 1, H, W   (hand mask, for raw/hand-aware stream)
            full_mask=full_masks,  # V, 1, H, W   (complete amodal silhouette, mask branch)
            pointmap=pointmaps,    # V, 3, H, W  (zeros)
            occ_mask=occ_masks,    # V, 1, H, W  (zeros)
            global_ss=global_ss,   # 1, 64, 64, 64
            w2c=w2c_0,             # 4, 4
            depth=depth,           # V, H, W      (rendered perpendicular z-depth; 1e10=bg)
            camera_angle_x=camera_angle_x,  # V    (horizontal FOV -> intrinsics)
        )


def forehoi_custom_collate(batch):
    """Collate for ForeHOIDataset — identical to mv_custom_collate."""
    return {
        'image':     torch.stack([s['image']     for s in batch], dim=0),
        'mask':      torch.stack([s['mask']      for s in batch], dim=0),
        'hand_mask': torch.stack([s['hand_mask'] for s in batch], dim=0),
        'full_mask': torch.stack([s['full_mask'] for s in batch], dim=0),
        'point_map': torch.stack([s['pointmap']  for s in batch], dim=0),
        'occ_mask':  torch.stack([s['occ_mask']  for s in batch], dim=0),
        'global_ss': torch.stack([s['global_ss'] for s in batch], dim=0),
        'w2c':       torch.stack([s['w2c']       for s in batch], dim=0),
        'depth':            torch.stack([s['depth']          for s in batch], dim=0),
        'camera_angle_x':   torch.stack([s['camera_angle_x'] for s in batch], dim=0),
    }


if __name__ == '__main__':
    import time

    dataset = ForeHOIDataset(
        data_json="forehoi_data.json",
        num_views=4,
        test_mode=False,
    )
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=2, shuffle=True,
                        num_workers=0, collate_fn=forehoi_custom_collate)

    t0 = time.time()
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        print(f"\nBatch {i}:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")
    print(f"\nTime for 3 batches: {time.time() - t0:.2f}s")
