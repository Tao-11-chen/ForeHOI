"""forehoi.tools — shared inference core for the local (trained) MV-SS pipeline.

Both the app (app_forehoi_sam3d.py) and the eval scripts import from here:
    from tools.pipeline import build_full_pipeline, save_mask_png
    import tools.mask_eval_common as C

Importing this package puts the repo root and core/ on sys.path so that
`import sam3d_objects`, `from inference import Inference`, and
`from training import load_model_part` resolve regardless of the caller's CWD.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "core")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CHECKPOINTS = os.path.join(ROOT, "checkpoints")
PIPELINE_CONFIG = os.path.join(CHECKPOINTS, "pipeline.yaml")
