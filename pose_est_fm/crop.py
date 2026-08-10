"""Object-crop preprocessing for the pose pipeline: crop around the hand∪object
bbox (15% margin), pad to square, resize to 518. Keeps depth/mask/intrinsics at a
consistent resolution for DepthAnything3 + the feature-matching pose stage.

Copied verbatim from pose_est/crop.py so pose_est_fm stays free of the
FoundationPose import chain (pose_est/__init__.py imports it eagerly).
"""
import numpy as np
from PIL import Image

INPUT_SIZE = 518


def preprocess_images_crop(rgb_images, hand_masks, object_masks):
    processed_rgb, processed_hand, processed_obj = [], [], []
    for rgb, h_mask, o_mask in zip(rgb_images, hand_masks, object_masks):
        H, W = rgb.shape[:2]
        hp = np.where(h_mask > 0); op = np.where(o_mask > 0)
        if len(hp[0]) == 0 and len(op[0]) == 0:
            x_min, y_min, x_max, y_max = 0, 0, W, H
        else:
            rows, cols = [], []
            if len(hp[0]) > 0:
                rows.append(hp[0]); cols.append(hp[1])
            if len(op[0]) > 0:
                rows.append(op[0]); cols.append(op[1])
            y_min, y_max = np.min(np.concatenate(rows)), np.max(np.concatenate(rows))
            x_min, x_max = np.min(np.concatenate(cols)), np.max(np.concatenate(cols))
        bh, bw = y_max - y_min, x_max - x_min
        mh, mw = int(bh * 0.15), int(bw * 0.15)
        cy1, cy2 = max(0, y_min - mh), min(H, y_max + mh)
        cx1, cx2 = max(0, x_min - mw), min(W, x_max + mw)
        crgb = rgb[cy1:cy2, cx1:cx2]; ch = h_mask[cy1:cy2, cx1:cx2]; co = o_mask[cy1:cy2, cx1:cx2]
        chh, cww = crgb.shape[:2]
        side = max(chh, cww)
        if chh > cww:
            pl = (chh - cww) // 2; pt = pb = 0; pr = chh - cww - pl
        elif cww > chh:
            pt = (cww - chh) // 2; pl = pr = 0; pb = cww - chh - pt
        else:
            pt = pb = pl = pr = 0
        prgb = np.zeros((side, side, 3), dtype=rgb.dtype); prgb[pt:pt + chh, pl:pl + cww] = crgb
        ph = np.zeros((side, side), np.uint8); ph[pt:pt + chh, pl:pl + cww] = (ch > 0).astype(np.uint8) * 255
        po = np.zeros((side, side), np.uint8); po[pt:pt + chh, pl:pl + cww] = (co > 0).astype(np.uint8) * 255
        processed_rgb.append(np.array(Image.fromarray(prgb).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.LANCZOS)))
        processed_hand.append(np.array(Image.fromarray(ph).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.NEAREST)) > 127)
        processed_obj.append(np.array(Image.fromarray(po).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.NEAREST)) > 127)
    return processed_rgb, processed_hand, processed_obj
