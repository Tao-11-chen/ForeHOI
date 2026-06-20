<div align="center">

# ForeHOI: Feed-forward 3D Object Reconstruction from Daily Hand-Object Interaction Videos
<a href="https://arxiv.org/abs/2602.06226" target="_blank">
  <img src="https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white" alt="arXiv">
</a>
<a href="https://tao-11-chen.github.io/project_pages/ForeHOI/" target="_blank">
  <img src="https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white" alt="Project Page">
</a>

</div>

![teaser](assets/Teaser.png)

## Abstract

We introduce ForeHOI, the first feed-forward 3D object reconstruction model from daily hand-object interaction videos. Given partially observed video input, our framework simultaneously completes 2D/3D objects and estimates their poses.

<details><summary>CLICK for the full abstract</summary>

> The ubiquity of monocular videos capturing daily hand-object interactions presents a valuable resource for embodied intelligence. While 3D hand reconstruction from in-the-wild videos has seen significant progress, reconstructing the involved objects remains challenging due to severe occlusions and the complex, coupled motion of the camera, hands, and object. In this paper, we introduce ForeHOI, a novel feed-forward model that directly reconstructs 3D object geometry from monocular hand-object interaction videos within one minute of inference time, eliminating the need for any pre-processing steps. Our key insight is that, the joint prediction of 2D mask inpainting and 3D shape completion in a feed-forward framework can effectively address the problem of severe occlusion in monocular hand-held object videos, thereby achieving results that outperform the performance of optimization-based methods. The information exchanges between the 2D and 3D shape completion boosts the overall reconstruction quality, enabling the framework to effectively handle severe hand-object occlusion. Furthermore, to support the training of our model, we contribute the first large-scale, high-fidelity synthetic dataset of hand-object interactions with comprehensive annotations. Extensive experiments demonstrate that ForeHOI achieves state-of-the-art performance in object reconstruction, significantly outperforming previous methods with around a 100x speedup.
</details>

## 🚧 Todo

- [✅] Release the ForeHOI dataset.
- [✅] Release the inference code.
- [✅] Release the training code.

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Tao-11-chen/ForeHOI.git
cd ForeHOI
```

**2. Create the conda environment and install dependencies**
```bash
conda create -n forehoi python=3.11 -y
conda activate forehoi

# PyTorch (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# core + inference dependencies
pip install -r requirements.txt
pip install -r requirements.inference.txt

# rendering / 3D
pip install "git+https://github.com/NVlabs/nvdiffrast.git"
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# segmentation + UI
pip install "git+https://github.com/facebookresearch/sam2.git"
pip install gradio_litmodel3d evo "moviepy==1.0.3"
```

> **FoundationPose extension:** the C++/pybind module `wheels/FoundationPose/mycpp` is shipped **prebuilt for Python 3.11**. If you use a different Python version, rebuild it (header-only Eigen + pybind11):
> ```bash
> cd wheels/FoundationPose/mycpp && mkdir -p build && cd build
> cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j
> ```

**3. Download the pretrained models**

Weights and checkpoints are hosted on the Hugging Face model repo [`YuantaoChen/ForeHOI`](https://huggingface.co/YuantaoChen/ForeHOI). Download them into the repo root:
```bash
pip install -U "huggingface_hub[hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download YuantaoChen/ForeHOI \
    --repo-type model --local-dir . --include "weights/**" "checkpoints/**"
```

This creates:
```
weights/        sam2.1_hiera_large.pt, DA3/, foundationpose/{scorer,refiner}/
checkpoints/    forehoi.ckpt, pipeline.yaml, ss_generator.ckpt, slat_generator.ckpt,
                ss_decoder.ckpt, slat_decoder_{gs,gs_4,mesh}.ckpt
```

## Demo

Two Gradio apps reconstruct the held object from a hand-object video and produce a 6-DoF pose-overlay video + a textured mesh:

```bash
python app_forehoi_sam3d.py   # our trained ForeHOI model   ->  http://localhost:7865
python app_mv_sam3d.py        # MV-SAM3D baseline            ->  http://localhost:7866
```

**Sample videos.** Six ready-to-use hand-object clips are provided in [`test_data/`](test_data) (`wild1.mp4` … `wild6.mp4`) — drop one into the **Upload Video** box to try the app without your own footage.

**Steps in the web UI:**

1. **Upload Video** — drag in one of the `test_data/wild*.mp4` samples (or your own hand-object clip).
2. Click **Process Video** — samples frames (count set by the *Number of frames to sample* slider) and shows the first frame.
3. **Mark points on the first frame:** with *Mask Type* = **object**, click on the held object (markers shown in **red**); then switch *Mask Type* to **hand** and click on the hand(s) (shown in **green**).
4. Click **Generate Masked Video** — SAM2 propagates the object/hand masks across all sampled frames.
5. Click **Reconstruct 3D + Pose** — runs the multi-view stage-1 (+amodal mask) and stage-2 SLAT reconstruction, then DepthAnything3 + FoundationPose for the pose-overlay video and a textured 3D mesh.

## Evaluation

Four eval scripts live in `eval/`. First set the dataset locations (placeholders at the top of the files): `HO3D` / `YCB` in `eval/ho3d_eval.py`, and `HOT3D` in `eval/hot3d_eval.py` / `eval/hot3d_metrics_cm.py` (the metrics scripts reuse the eval scripts' paths). Run all commands **from the repo root**. Both datasets follow the same two-step flow: `*_eval.py` runs the full ForeHOI pipeline (trained stage-1 + SAM3D stage-2, same as `app_forehoi_sam3d.py`) directly on the raw dataset and writes visualizations; `*_metrics_cm.py` then reports metric-scale **CD (cm)**, **F@5%**, and **F@10%** (CPU only).

**HO3D** (vs. voxelized YCB GT; vis -> `ho3d_eval_vis/`: amodal masks + stage-2 textured meshes):
```bash
CUDA_VISIBLE_DEVICES=0 python eval/ho3d_eval.py --views 8   # inference + vis
python eval/ho3d_metrics_cm.py                              # metrics: CD(cm) / F@5% / F@10%
```

**HOT3D** (vis -> `hot3d_eval_vis/`):
```bash
CUDA_VISIBLE_DEVICES=0 python eval/hot3d_eval.py --views 8   # inference + vis
python eval/hot3d_metrics_cm.py                              # metrics: CD(cm) / F@5% / F@10%
```

## Training

Set the dataset paths in `train/forehoi_data.json` (entries like `/data/graspxl_renders/<obj>/data.tar`), then launch with `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 train/train_ss_mask_depth.py
```


## Dataset

Training dataset is available on [huggingface](https://huggingface.co/datasets/YuantaoChen/ForeHOI/).

## Acknowledgements

This project is built on top of several awesome open-source works, vendored under `wheels/`:

- [FoundationPose](https://github.com/NVlabs/FoundationPose) (NVIDIA) — 6-DoF object pose estimation & tracking.
- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (ByteDance) — metric depth & point maps.
- [SAM 3D Objects / MV-SAM3D](https://github.com/facebookresearch/sam-3d-objects) (Meta) — single-/multi-view 3D object reconstruction.

We also gratefully build on [SAM 2](https://github.com/facebookresearch/sam2), [MoGe](https://github.com/microsoft/MoGe), and [TRELLIS](https://github.com/microsoft/TRELLIS). Many thanks to the authors for releasing their code.
