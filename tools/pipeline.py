"""Build the full inference pipeline (stage-1 SS + amodal mask + stage-2 SLAT)
for the locally-trained MV-SAM3D model, and load a checkpoint into it.

Extracted from the original wild_full_test.py:
ctx 3072 -> MoGe freed; ctx 4096 -> pointmap stream
(PointPatchEmbed@296 + pointmap_gate), MoGe kept resident for build_cond.
"""
import os
import math

import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, get_peft_model

from tools import PIPELINE_CONFIG  # also triggers sys.path setup (root + core)
from tools import mask_eval_common as C

from inference import Inference
from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import ModulatedMultiViewCond
from sam3d_objects.model.backbone.tdfy_dit.models.mask_branch import MaskBranch


def build_full_pipeline(ckpt, device):
    """Full inference pipeline (ss+slat) + our mask stage-1 modules + ckpt."""
    if ckpt is None:
        ckpt = C.latest_ckpt()
    arch = C.detect_arch(ckpt)
    ctx, use_depth = arch["cond_ctx"], arch["use_depth"]

    # Load the checkpoint once; auto-detect whether ss_generator is LoRA-wrapped
    # (raw training/LoRA ckpt) or already LoRA-merged (forehoi.ckpt).
    states = torch.load(ckpt, map_location="cpu")
    states = states.get("state_dict", states)
    has_lora = any("lora_" in k for k in states)

    pipeline = Inference(PIPELINE_CONFIG, compile=False)._pipeline
    if not use_depth:
        # MoGe unused for stage2-from-coords; free its GPU memory.
        try:
            pipeline.depth_model.model.to("cpu"); torch.cuda.empty_cache()
        except Exception:
            pass

    if has_lora:
        pipeline.models["ss_generator"] = get_peft_model(
            pipeline.models["ss_generator"],
            LoraConfig(r=256, lora_alpha=512, lora_dropout=0.0, target_modules=C.LORA_TARGETS),
        )
    if "ss_condition_embedder" in pipeline.condition_embedders:
        pipeline.models["ss_condition_embedder"] = pipeline.condition_embedders["ss_condition_embedder"]
    dev = next(pipeline.models["ss_generator"].parameters()).device
    pipeline.models["multiview_cond"] = ModulatedMultiViewCond(
        1024, ctx, num_heads=16, mlp_ratio=4, attn_mode="full",
        use_checkpoint=False, use_rope=False, share_mod=False,
        qk_rms_norm=True, qk_rms_norm_cross=False).to(dev)
    emb = nn.Module(); emb.weight = nn.Parameter(torch.empty(2, ctx))
    nn.init.normal_(emb.weight, std=1.0 / math.sqrt(ctx))
    pipeline.models["ref_src_emb"] = emb.to(dev)
    pipeline.models["mask_branch"] = MaskBranch(model_channels=1024, mask_res=C.MASK_RES, patch=4).to(dev)
    rg = nn.Module(); rg.weight = nn.Parameter(torch.zeros(1))
    pipeline.models["raw_gate"] = rg.to(dev)
    prefixes = [("models.ss_generator.", "ss_generator"),
                ("models.multiview_cond.", "multiview_cond"),
                ("models.ref_src_emb.", "ref_src_emb"),
                ("models.mask_branch.", "mask_branch"),
                ("models.raw_gate.", "raw_gate")]
    if use_depth:
        pipeline.use_depth = True
        pipeline.cond_ctx_channels = ctx
        pipeline.models["ss_condition_embedder"].module_list[2].input_size = 296
        pg = nn.Module(); pg.weight = nn.Parameter(torch.zeros(1))
        pipeline.models["pointmap_gate"] = pg.to(dev)
        prefixes += [("models.pointmap_gate.", "pointmap_gate"),
                     ("models.ss_condition_embedder.", "ss_condition_embedder")]

    for prefix, key in prefixes:
        sub = {k.replace(prefix, ""): v for k, v in states.items() if k.startswith(prefix)}
        if sub:
            m, u = pipeline.models[key].load_state_dict(sub, strict=False)
            print(f"  {key}: loaded {len(sub)} (missing={len(m)} unexpected={len(u)})", flush=True)
    pipeline.models["ss_generator"].eval()
    pipeline.rendering_engine = "pytorch3d"  # texture baking without nvdiffrast JIT
    print(f"[build] (ctx={ctx}) {'LoRA' if has_lora else 'merged'} "
          f"full pipeline + mask stage-1, ckpt={os.path.basename(ckpt)}", flush=True)
    return pipeline, ckpt


def save_mask_png(path, z_mask):
    """Save per-view inferred amodal masks as a vertical strip png."""
    import numpy as np
    pred = (z_mask[0, :, 0] > 0).cpu().numpy().astype(np.uint8) * 255
    rows = [pred[v] for v in range(min(pred.shape[0], 8))]
    grid = rows[0]
    for r in rows[1:]:
        grid = np.concatenate([grid, np.full((4, C.MASK_RES), 128, np.uint8), r], axis=0)
    Image.fromarray(grid, mode="L").save(path)
