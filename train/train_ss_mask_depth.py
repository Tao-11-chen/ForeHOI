import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import tools as _tools  # noqa: F401  (puts repo root + core/ on sys.path)

"""Multi-view SS training on ForeHOI with a 4th DEPTH/POINTMAP cond stream.

Extends the mask model (dual-stream DINO [image|mask|raw] + amodal mask output)
with a 4th pointmap stream → cond ctx 3072→4096:

  * pointmap stream = PointPatchEmbed(input_size=296)→projection_nets[2]→1024,
    channel-concatenated as the 4th block; gated by a zero-init `pointmap_gate`;
  * MIXED training: each step samples the pointmap source from
        {rendered depth.tar · MoGe-of-rendered-RGB (on-the-fly) · drop=zeros};
    MoGe is kept RESIDENT (TrainingMVSSPipeline(use_depth=True));
  * object-masked + per-view ObjectCentricSSI-normalized + PyTorch3D frame;
  * `multiview_cond` + `ref_src_emb` RE-INIT (ctx changed 3072→4096, can't warm-start);
    `PointPatchEmbed`/`projection_nets[2]` warm-started from the base (already
    MoGe-pointmap-conditioned); `pointmap_gate` zero-init.

Warm-start (mask ckpt): ss_generator LoRA/norms + mask_branch + raw_gate only
  (multiview_cond / ref_src_emb excluded — shape changed). mask_branch works off the
  raw DINO `raw_ctx` (independent of the fresh cond), so it transfers cleanly.

Checkpoints are STAMPED with `forehoi_arch` so the eval
build can pick the right architecture; ckpts with no stamp (ctx 3072) still load.

Launch (multi-GPU): CUDA_VISIBLE_DEVICES=2,3,4,5,6 torchrun --nproc_per_node=5 this_file
"""
import os

import sys
sys.path.append("core")
import math
import types
import torch
from training import load_model_part
from forehoi_dataset import ForeHOIDataset, forehoi_custom_collate
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from peft import LoraConfig, get_peft_model
import torch.nn as nn

WARMSTART_CKPT = "checkpoints/mv_sam3d_forehoi_mask/epoch=81-step=40000.ckpt"
SAVE_DIR = "checkpoints/mv_sam3d_forehoi_mask_depth_fix"   # fresh: TRELLIS-voxelized coords + 1-8 views


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    torch.autograd.set_detect_anomaly(False)

    dataset = ForeHOIDataset(
        data_json=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "forehoi_data.json"),
        num_views=8,        # get_input samples a random 1-8 view subset per step
        test_mode=False,
    )

    config_file = f"checkpoints/pipeline.yaml"
    pipeline = load_model_part(config_file, depth=True)   # use_depth=True, cond_ctx=4096, MoGe resident
    print("Model loaded successfully (depth).")

    # ------------------------------------------------------------------
    # Freeze / train scheme
    # ------------------------------------------------------------------
    pipeline.models['ss_generator'].eval()
    # ss_condition_embedder: frozen EXCEPT the pointmap embedder (module_list[2]
    # + projection_nets[2]), which carries the new depth stream.
    pipeline.models["ss_condition_embedder"].eval()
    for p in pipeline.models["ss_condition_embedder"].parameters():
        p.requires_grad = False
    for p in pipeline.models["ss_condition_embedder"].module_list[2].parameters():
        p.requires_grad = True
    for p in pipeline.models["ss_condition_embedder"].projection_nets[2].parameters():
        p.requires_grad = True
    pipeline.models["ss_condition_embedder"].module_list[2].train()
    pipeline.models["ss_condition_embedder"].projection_nets[2].train()
    # Conditioning fusion + new modules: trainable.
    for key in ["multiview_cond", "ref_src_emb", "raw_gate", "mask_branch", "pointmap_gate"]:
        for p in pipeline.models[key].parameters():
            p.requires_grad = True

    # LoRA on the SS DiT — SHAPE modality only (pose kept inert: no LoRA).
    trellis_peft_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.0,
        target_modules=["to_q.shape",
                        "to_kv.shape",
                        "to_out.shape",
                        "to_qkv.shape",
                        "shape.to_q",
                        "shape.to_kv",
                        "shape.to_out",
                        "shape.to_qkv",
                        "shape.mlp.0",
                        "shape.mlp.2",
                        "adaLN_modulation.1"]
    )
    pipeline.models['ss_generator'] = get_peft_model(pipeline.models['ss_generator'], trellis_peft_config)
    pipeline.models['ss_generator'].print_trainable_parameters()

    for name, p in pipeline.models['ss_generator'].named_parameters():
        if '.norm' in name or 'gamma' in name:
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Warm-start from the mask ckpt. Keep ONLY ss_generator (shape LoRA/norms)
    # + mask_branch + raw_gate — these are shape-compatible. multiview_cond and
    # ref_src_emb changed shape (ctx 3072→4096) so are excluded (kept fresh); the
    # pointmap embedder stays at its base (already-loaded) weights; pointmap_gate
    # zero-init. strict=False ignores the missing fresh keys.
    # ------------------------------------------------------------------
    import glob as _glob, re as _re
    def _step(p):
        m = _re.search(r"step=(\d+)", p); return int(m.group(1)) if m else -1
    cks = _glob.glob(os.path.join(SAVE_DIR, "*.ckpt"))
    if cks:
        # RESUME from the latest ckpt — keep the (from-scratch) cond + pointmap progress.
        src = max(cks, key=_step)
        keep = ("models.ss_generator.", "models.multiview_cond.", "models.ref_src_emb.",
                "models.raw_gate.", "models.mask_branch.", "models.pointmap_gate.",
                "models.ss_condition_embedder.module_list.2.",
                "models.ss_condition_embedder.projection_nets.2.")
        tag = "resume"
    elif os.path.exists(WARMSTART_CKPT):
        # cold start from the mask ckpt (ss_generator LoRA + mask_branch + raw_gate only;
        # multiview_cond / ref_src_emb changed shape 3072->4096, so kept fresh).
        src = WARMSTART_CKPT
        keep = ("models.ss_generator.", "models.mask_branch.", "models.raw_gate.")
        tag = "warm-start"
    else:
        src = None
    if src:
        sd = torch.load(src, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k: v for k, v in sd.items() if k.startswith(keep)}
        missing, unexpected = pipeline.load_state_dict(sd, strict=False)
        print(f"[{tag}] loaded {len(sd)} tensors from {os.path.basename(src)}; unexpected={len(unexpected)}")
    else:
        print("[warm-start] no ckpt found — cond/LoRA/mask from base release.")

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    batch_size = 1   # 8 views: peak view-crops ~= old 4-view/bs-2; keep VRAM in check
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=forehoi_custom_collate,
        num_workers=16,
        persistent_workers=True,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Optimizer: warmup + cosine. Group1 (5e-5): warm/base params (LoRA + cond +
    # pointmap embedder). Group2 (1e-4): from-scratch mask_branch + pointmap_gate.
    # ------------------------------------------------------------------
    warmup_steps = 500
    max_steps = 200000
    min_lr_ratio = 0.05

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        cond_params = list(self.models['multiview_cond'].parameters()) + \
                      list(self.models['ref_src_emb'].parameters()) + \
                      list(self.models['raw_gate'].parameters()) + \
                      list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                      list(self.models['ss_condition_embedder'].projection_nets[2].parameters())
        new_params = list(self.models['mask_branch'].parameters()) + \
                     list(self.models['pointmap_gate'].parameters())
        opt = torch.optim.AdamW([
            {"params": lora_params + cond_params, "lr": 5e-5},
            {"params": new_params, "lr": 1e-4},
        ], weight_decay=0.0)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    pipeline.configure_optimizers = types.MethodType(configure_optimizers, pipeline)

    # Save only the trainable / new params + an arch stamp.
    def on_save_checkpoint(self, checkpoint):
        keep = ("models.ss_generator.", "models.multiview_cond.", "models.ref_src_emb.",
                "models.raw_gate.", "models.mask_branch.", "models.pointmap_gate.",
                "models.ss_condition_embedder.module_list.2.",
                "models.ss_condition_embedder.projection_nets.2.")
        sd = checkpoint["state_dict"]
        checkpoint["state_dict"] = {k: v for k, v in sd.items() if k.startswith(keep)}
        checkpoint["forehoi_arch"] = {
            "output": "mask",
            "cond_ctx": 4096,
            "streams": ["image", "mask", "raw", "pointmap"],
            "pointmap": {"embed": "PointPatchEmbed@296", "gate": "pointmap_gate",
                         "train_modes": ["rendered", "moge", "drop"]},
        }

    pipeline.on_save_checkpoint = types.MethodType(on_save_checkpoint, pipeline)

    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=SAVE_DIR,
        every_n_train_steps=2000,
        save_top_k=-1,
        save_weights_only=True,
    )

    from pytorch_lightning.strategies import DDPStrategy
    n_devices = torch.cuda.device_count()  # respects CUDA_VISIBLE_DEVICES
    trainer = pl.Trainer(
        devices=n_devices,
        accelerator="cuda",
        max_epochs=500,
        precision="bf16-mixed",
        strategy=DDPStrategy(find_unused_parameters=True, process_group_backend="nccl"),
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=2,
        gradient_clip_val=0.5,
    )

    trainer.fit(pipeline, dataloader)


if __name__ == '__main__':
    main()
