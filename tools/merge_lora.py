"""Merge a LoRA-trained ss_generator checkpoint into a single self-contained,
LoRA-free checkpoint (default: checkpoints/forehoi.ckpt).

Only ss_generator carries LoRA; merge_and_unload folds W += (alpha/r)*B@A (exact).
All other (already-full) modules are kept byte-for-byte from the input checkpoint.

    python tools/merge_lora.py --lora /path/to/lora_checkpoint.ckpt
    python tools/merge_lora.py --lora lora.ckpt --out checkpoints/forehoi.ckpt
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import tools  # noqa: F401  (root + core on path)
from tools.pipeline import build_full_pipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def merge(lora_ckpt, out_ckpt, device=DEVICE):
    pipeline, _ = build_full_pipeline(lora_ckpt, device)     # LoRA path (auto-detected)
    ssg = pipeline.models["ss_generator"]
    try:
        merged = ssg.merge_and_unload()
    except AttributeError:
        merged = ssg.base_model.merge_and_unload()
    merged_sd = merged.state_dict()                          # plain ss_generator keys, LoRA folded in

    orig = torch.load(lora_ckpt, map_location="cpu")
    orig = orig.get("state_dict", orig)

    new_sd = {f"models.ss_generator.{k}": v.detach().cpu() for k, v in merged_sd.items()}
    kept = 0
    for k, v in orig.items():
        if not k.startswith("models.ss_generator."):         # keep side-modules verbatim
            new_sd[k] = v
            kept += 1

    assert not any("lora_" in k for k in new_sd), "LoRA keys still present after merge!"
    torch.save({"state_dict": new_sd,
                "forehoi_arch": {"cond_ctx": 4096, "use_depth": True}}, out_ckpt)
    print(f"[merge] ss_generator merged tensors: {len(merged_sd)} | side tensors kept: {kept} "
          f"| total: {len(new_sd)}", flush=True)
    print(f"[merge] saved -> {out_ckpt}  ({os.path.getsize(out_ckpt)/1e9:.2f} GB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Merge a LoRA ss_generator checkpoint into a LoRA-free checkpoint.")
    ap.add_argument("--lora", required=True, help="path to the LoRA-trained checkpoint to merge")
    ap.add_argument("--out", default=os.path.join(tools.ROOT, "checkpoints/forehoi.ckpt"),
                    help="output path for the merged checkpoint")
    args = ap.parse_args()
    merge(args.lora, args.out)


if __name__ == "__main__":
    main()
