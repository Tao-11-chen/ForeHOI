"""MASt3R wrapper: pairwise descriptor extraction + reciprocal-NN matching.

Mirrors the ReconViaGen usage (app_fine.py) with two fixes:
  * match coordinates are cast to float32 BEFORE being rescaled back to the
    original resolution (ReconViaGen scales int arrays in place, silently
    truncating to whole pixels);
  * degenerate results (too few matches) are reported as empty arrays instead
    of crashing downstream.

Import order matters: `wheels.mast3r` must be imported first — it pulls in
`mast3r.utils.path_to_dust3r`, which prepends wheels/dust3r to sys.path so the
absolute `dust3r.*` imports inside the dust3r package resolve.
"""
import numpy as np
import torch
from PIL import Image

from wheels.mast3r.model import AsymmetricMASt3R           # noqa: F401  (sets up dust3r path)
from wheels.mast3r.fast_nn import fast_reciprocal_NNs
from wheels.dust3r.dust3r.inference import inference as dust3r_inference
from wheels.dust3r.dust3r.utils.image import load_images_new

# Metric MASt3R checkpoint on the Hugging Face hub; a local .pth path also works.
DEFAULT_MAST3R_WEIGHTS = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


class Mast3rMatcher:
    """Runs MASt3R on (render, real) image pairs and returns 2D-2D matches.

    All images are resized to `size` (long side) by load_images_new; returned
    matches are in the RESIZED coordinate frame together with each image's
    resized shape so callers can map back to the original resolution.
    """

    def __init__(self, weights=DEFAULT_MAST3R_WEIGHTS, device="cuda",
                 batch_size=4, size=512, subsample=8, low_vram=False):
        import os
        self.device, self.batch_size, self.size, self.subsample = device, batch_size, size, subsample
        self.low_vram = low_vram
        if os.path.isfile(weights):
            from wheels.mast3r.model import load_model
            self.model = load_model(weights, device)
        else:  # HF hub repo id (downloaded to the hub cache on first use)
            self.model = AsymmetricMASt3R.from_pretrained(weights)
        self.model = self.model.to(device).eval()
        if low_vram:
            self.model.to("cpu")

    def _to_device(self):
        if self.low_vram:
            self.model.to(self.device)

    def _offload(self):
        if self.low_vram:
            self.model.to("cpu")
            torch.cuda.empty_cache()

    @torch.no_grad()
    def match_pairs(self, images_a, images_b):
        """Match N pairs. images_*: list of (H,W,3) uint8 np arrays.

        Returns a list of N dicts:
            matches_a : (M,2) float32 (x,y) in the resized coords of image a
            matches_b : (M,2) float32 (x,y) in the resized coords of image b
            shape_a   : (H,W) resized shape of image a
            shape_b   : (H,W) resized shape of image b
        """
        pairs = []
        for a, b in zip(images_a, images_b):
            views = load_images_new([Image.fromarray(a), Image.fromarray(b)],
                                    size=self.size, square_ok=True, verbose=False)
            pairs.append(tuple(views))
        self._to_device()
        out = dust3r_inference(pairs, self.model, self.device,
                               batch_size=self.batch_size, verbose=False)
        results = []
        for i in range(len(pairs)):
            desc1 = out["pred1"]["desc"][i]   # (H1,W1,D) cpu tensor
            desc2 = out["pred2"]["desc"][i]   # (H2,W2,D) cpu tensor
            m_a, m_b = fast_reciprocal_NNs(
                desc1, desc2, subsample_or_initxy1=self.subsample,
                device=self.device, dist="dot", block_size=2**13)
            results.append(dict(
                matches_a=np.asarray(m_a, dtype=np.float32),
                matches_b=np.asarray(m_b, dtype=np.float32),
                # views are collated: true_shape is one (N,2) array of (H,W) rows
                shape_a=tuple(int(s) for s in out["view1"]["true_shape"][i]),
                shape_b=tuple(int(s) for s in out["view2"]["true_shape"][i]),
            ))
        self._offload()
        return results

    def match(self, image_a, image_b):
        """Single-pair convenience wrapper around match_pairs."""
        return self.match_pairs([image_a], [image_b])[0]
