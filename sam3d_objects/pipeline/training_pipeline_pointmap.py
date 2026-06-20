# Copyright (c) Meta Platforms, Inc. and affiliates.
from typing import Union, Optional
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from loguru import logger
from PIL import Image

from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Transform3d

from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed
from sam3d_objects.pipeline.training_pipeline import TrainingPipeline
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.model.backbone.tdfy_dit.models.mot_sparse_structure_flow import (
    ModulatedMultiViewCond
)
from sam3d_objects.model.backbone.tdfy_dit.models.mask_branch import MaskBranch
import torch.nn.functional as F
from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae_xyz import (
    SparseStructureEncoderXYZ,
    SparseStructureDecoderXYZ,
)
from sam3d_objects.data.dataset.tdfy.transforms_3d import (
    DecomposedTransform,
)
from sam3d_objects.pipeline.utils.pointmap import infer_intrinsics_from_pointmap
from sam3d_objects.pipeline.inference_utils import o3d_plane_estimation, estimate_plane_area
from sam3d_objects.model.layers.llama3.ff import FeedForward
import math
import random

def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    """
    R3 camera space --> PyTorch3D camera space
    Also needed for pointmaps
    """
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


def recursive_fn_factory(fn):
    def recursive_fn(b):
        if isinstance(b, dict):
            return {k: recursive_fn(b[k]) for k in b}
        if isinstance(b, list):
            return [recursive_fn(t) for t in b]
        if isinstance(b, tuple):
            return tuple(recursive_fn(t) for t in b)
        if isinstance(b, torch.Tensor):
            return fn(b)
        # Yes, writing out an explicit white list of
        # trivial types is tedious, but so are bugs that
        # come from not applying fn, when expected to have
        # applied it.
        if b is None:
            return b
        trivial_types = [bool, int, float]
        for t in trivial_types:
            if isinstance(b, t):
                return b
        raise TypeError(f"Unexpected type {type(b)}")

    return recursive_fn


recursive_contiguous = recursive_fn_factory(lambda x: x.contiguous())
recursive_clone = recursive_fn_factory(torch.clone)


def compile_wrapper(
    fn, *, mode="max-autotune", fullgraph=True, dynamic=False, name=None
):
    compiled_fn = torch.compile(fn, mode=mode, fullgraph=fullgraph, dynamic=dynamic)

    def compiled_fn_wrapper(*args, **kwargs):
        with torch.autograd.profiler.record_function(
            f"compiled {fn}" if name is None else name
        ):
            cont_args = recursive_contiguous(args)
            cont_kwargs = recursive_contiguous(kwargs)
            result = compiled_fn(*cont_args, **cont_kwargs)
            cloned_result = recursive_clone(result)
            return cloned_result

    return compiled_fn_wrapper

class TrainingSSPipeline(TrainingPipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=None, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)
        for key in ['slat_generator', 'slat_decoder_gs', 'slat_decoder_gs_4', 
                    'slat_decoder_mesh']:
            if key in self.models:
                del self.models[key]
        for key in ["slat_condition_embedder"]:
            if key in self.condition_embedders:
                del self.condition_embedders[key]

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        for model in self.models.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

        for model in self.condition_embedders.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)
        
        if dtype is not None and device is not None:
            self.depth_model.model.to(device, dtype)
        elif device is not None:
            self.depth_model.model.to(device)
        elif dtype is not None:
            self.depth_model.model.type(dtype)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
            return pointmap

        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # bs, h, w, _ = pointmap.shape
        pointmap_flat = pointmap
        # Get valid points from the mask
        mask_bool = mask_resized[:,0] > 0.5
        mask_points = pointmap_flat[mask_bool]
        mask_distance = mask_points.nanmedian(dim=-1).values[-1]
        logger.info(f"mask_distance: {mask_distance}")
        pointmap_clipped_flat = torch.where(
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped

    def compute_pointmap(self, image, pointmap=None):
        loaded_image = image
        loaded_mask = loaded_image[:, 3]
        loaded_image = loaded_image.contiguous()[:, :3]

        if pointmap is None:
            with torch.no_grad():
                with torch.autocast(device_type=str(loaded_image.device), dtype=self.dtype):
                    output = self.depth_model(loaded_image)
            pointmaps = output["pointmaps"]
            camera_convention_transform = (
                Transform3d()
                .rotate(camera_to_pytorch3d_camera(device=self.device).rotation)
                .to(self.device)
            )
            bs, h, w, _ = pointmaps.shape
            points_tensor = camera_convention_transform.transform_points(pointmaps.reshape(bs, -1, 3)).reshape(bs, h, w, 3)
            intrinsics = output.get("intrinsics", None)
        else:
            output = {}
            points_tensor = pointmap.to(self.device)
            if loaded_image.shape != points_tensor.shape:
                # Interpolate points_tensor to match loaded_image size
                # loaded_image has shape [3, H, W], we need H and W
                points_tensor = torch.nn.functional.interpolate(
                    points_tensor.permute(2, 0, 1).unsqueeze(0),
                    size=(loaded_image.shape[1], loaded_image.shape[2]),
                    mode="nearest",
                ).squeeze(0).permute(1, 2, 0)
            intrinsics = None

        points_tensor = self._clip_pointmap(points_tensor, loaded_mask).permute(0,3,1,2)
        
        # Prepare the point map tensor
        point_map_tensor = {
            "pointmap": points_tensor,
            "pts_color": loaded_image,
        }

        # If depth model doesn't provide intrinsics, infer them
        if intrinsics is None:
            intrinsics_result = infer_intrinsics_from_pointmap(
                points_tensor.permute(1, 2, 0), device=self.device
            )
            point_map_tensor["intrinsics"] = intrinsics_result["intrinsics"]

        return point_map_tensor

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
    ) -> torch.Tensor:
        # canonical type is numpy

        assert image.ndim == 4  # no batch dimension as of now
        assert image.shape[1] == 4  # rgba format
        # assert image.dtype == np.uint8  # [0,255] range

        rgba_image = image
        rgba_image = rgba_image.contiguous()
        rgb_image = rgba_image[:, :3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        rgb_image, rgb_image_mask, pointmap = list(rgb_image.unbind(dim=0)), list(rgb_image_mask.unbind(dim=0)), list(pointmap.unbind(dim=0))
        masks = []
        images = []
        rgb_images = []
        rgb_image_masks = []
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            pointmaps = []
            rgb_pointmaps = []
            pointmap_scales = []
            pointmap_shifts = []
            rgb_pointmap_scales = []
            rgb_pointmap_shifts = []
        
        for i in range(len(rgb_image)):
            preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
                rgb_image[i], rgb_image_mask[i], pointmap[i]
            )
            masks.append(preprocessor_return_dict["mask"])
            images.append(preprocessor_return_dict["image"])
            rgb_images.append(preprocessor_return_dict["rgb_image"])
            rgb_image_masks.append(preprocessor_return_dict["rgb_image_mask"])
            if pointmap is not None and preprocessor.pointmap_transform != (None,):
                pointmaps.append(preprocessor_return_dict["pointmap"])
                rgb_pointmaps.append(preprocessor_return_dict["rgb_pointmap"])
                pointmap_scales.append(preprocessor_return_dict["pointmap_scale"])
                pointmap_shifts.append(preprocessor_return_dict["pointmap_shift"])
                rgb_pointmap_scales.append(preprocessor_return_dict["rgb_pointmap_scale"])
                rgb_pointmap_shifts.append(preprocessor_return_dict["rgb_pointmap_shift"])
        # Put in a for loop?
        item = {
            "mask": torch.stack(masks, dim=0).to(self.device),
            "image": torch.stack(images, dim=0).to(self.device),
            "rgb_image": torch.stack(rgb_images, dim=0).to(self.device),
            "rgb_image_mask": torch.stack(rgb_image_masks, dim=0).to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = torch.stack(pointmaps, dim=0).to(self.device)
            item["rgb_pointmap"] = torch.stack(rgb_pointmaps, dim=0).to(self.device)
            item["pointmap_scale"] = torch.stack(pointmap_scales, dim=0).to(self.device)
            item["pointmap_shift"] = torch.stack(pointmap_shifts, dim=0).to(self.device)
            item["rgb_pointmap_scale"] = torch.stack(rgb_pointmap_scales, dim=0).to(self.device)
            item["rgb_pointmap_shift"] = torch.stack(rgb_pointmap_shifts, dim=0).to(self.device)

        return item

    def get_input(self, batch):
        pointmap_dict = self.compute_pointmap(batch['image'], None)
        pointmap = pointmap_dict["pointmap"] # B, 3, H, W

        ss_input_dict = self.preprocess_image(
            batch['image'], self.ss_preprocessor, pointmap=pointmap
        )
        
        condition_args, condition_kwargs = self.get_condition_input(
            self.condition_embedders["ss_condition_embedder"],
            ss_input_dict,
            self.ss_condition_input_mapping,
        )
        return condition_args, condition_kwargs
        

    def training_step(self, batch, batch_idx):
        condition_args, condition_kwargs = self.get_input(batch)
        x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        condition_args, condition_kwargs = self.get_input(batch)
        x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss
    
    def configure_optimizers(self):
        params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        return opt
    
class TrainingPartSSPipeline(TrainingPipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=None, clip_pointmap_beyond_scale=None, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        super().__init__(*args, **kwargs)
        for key in ['slat_generator', 'slat_decoder_gs', 'slat_decoder_gs_4', 
                    'slat_decoder_mesh']:
            if key in self.models:
                del self.models[key]
        for key in ["slat_condition_embedder"]:
            if key in self.condition_embedders:
                del self.condition_embedders[key]
        del self.depth_model
        self.models['ss_condition_embedder'] = deepcopy(self.condition_embedders['ss_condition_embedder'])
        del self.condition_embedders['ss_condition_embedder']
        self.models["global_ss_condition_embedder"] = torch.nn.Sequential(
            torch.nn.Linear(8, 1024),
            torch.nn.LayerNorm(1024),
            FeedForward(1024, 4096, 1024)
            ).to(self.device)
        self.models['ss_encoder_xyz'] = SparseStructureEncoderXYZ(
            in_channels=3,
            latent_channels=8,
            num_res_blocks=2,
            num_res_blocks_middle=2,
            channels=[32, 128, 512],
            # use_fp16=True
        ).to(self.device)

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        for model in self.models.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

        for model in self.condition_embedders.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _clip_pointmap(self, pointmap: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.clip_pointmap_beyond_scale is None:
            return pointmap

        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask_resized = torchvision.transforms.functional.resize(
            mask, pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST
        )

        # bs, h, w, _ = pointmap.shape
        pointmap_flat = pointmap
        # Get valid points from the mask
        mask_bool = mask_resized[:,0] > 0.5
        mask_points = pointmap_flat[mask_bool]
        mask_distance = mask_points.nanmedian(dim=-1).values[-1]
        logger.info(f"mask_distance: {mask_distance}")
        pointmap_clipped_flat = torch.where(
            pointmap_flat[2, ...].abs() > self.clip_pointmap_beyond_scale * mask_distance,
            torch.full_like(pointmap_flat, float('nan')),
            pointmap_flat
        )
        pointmap_clipped = pointmap_clipped_flat.reshape(pointmap.shape)
        return pointmap_clipped

    def preprocess_image(
        self,
        image: Union[Image.Image, np.ndarray],
        preprocessor,
        pointmap=None,
        normalize_pointmap=False,
    ) -> torch.Tensor:
        # canonical type is numpy

        assert image.ndim == 4  # no batch dimension as of now
        assert image.shape[1] == 4  # rgba format
        # assert image.dtype == np.uint8  # [0,255] range

        rgba_image = image
        rgba_image = rgba_image.contiguous()
        rgb_image = rgba_image[:, :3]
        rgb_image_mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        rgb_image, rgb_image_mask, pointmap = list(rgb_image.unbind(dim=0)), list(rgb_image_mask.unbind(dim=0)), list(pointmap.unbind(dim=0))
        masks = []
        images = []
        rgb_images = []
        rgb_image_masks = []
        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            pointmaps = []
            rgb_pointmaps = []
            pointmap_scales = []
            pointmap_shifts = []
            rgb_pointmap_scales = []
            rgb_pointmap_shifts = []
        
        for i in range(len(rgb_image)):
            preprocessor_return_dict = preprocessor._process_image_mask_pointmap_mess(
                rgb_image[i], rgb_image_mask[i], pointmap[i], normalize_pointmap=normalize_pointmap
            )
            masks.append(preprocessor_return_dict["mask"])
            images.append(preprocessor_return_dict["image"])
            rgb_images.append(preprocessor_return_dict["rgb_image"])
            rgb_image_masks.append(preprocessor_return_dict["rgb_image_mask"])
            if pointmap is not None and preprocessor.pointmap_transform != (None,):
                pointmaps.append(preprocessor_return_dict["pointmap"])
                rgb_pointmaps.append(preprocessor_return_dict["rgb_pointmap"])
                pointmap_scales.append(preprocessor_return_dict["pointmap_scale"])
                pointmap_shifts.append(preprocessor_return_dict["pointmap_shift"])
                rgb_pointmap_scales.append(preprocessor_return_dict["rgb_pointmap_scale"])
                rgb_pointmap_shifts.append(preprocessor_return_dict["rgb_pointmap_shift"])
        # Put in a for loop?
        item = {
            "mask": torch.stack(masks, dim=0).to(self.device),
            "image": torch.stack(images, dim=0).to(self.device),
            "rgb_image": torch.stack(rgb_images, dim=0).to(self.device),
            "rgb_image_mask": torch.stack(rgb_image_masks, dim=0).to(self.device),
        }

        if pointmap is not None and preprocessor.pointmap_transform != (None,):
            item["pointmap"] = torch.stack(pointmaps, dim=0).to(self.device)
            item["rgb_pointmap"] = torch.stack(rgb_pointmaps, dim=0).to(self.device)
            item["pointmap_scale"] = torch.stack(pointmap_scales, dim=0).to(self.device)
            item["pointmap_shift"] = torch.stack(pointmap_shifts, dim=0).to(self.device)
            item["rgb_pointmap_scale"] = torch.stack(rgb_pointmap_scales, dim=0).to(self.device)
            item["rgb_pointmap_shift"] = torch.stack(rgb_pointmap_shifts, dim=0).to(self.device)

        return item

    def get_input(self, batch):
        pointmap = batch["point_map"]

        ss_input_dict = self.preprocess_image(
            batch['image'], self.ss_preprocessor, pointmap=pointmap
        )
        
        condition_args, condition_kwargs = self.get_condition_input(
            self.models['ss_condition_embedder'],
            ss_input_dict,
            self.ss_condition_input_mapping,
        )
        condition_args = self._post_process_condition_args(condition_args)
        with torch.no_grad():
            gt_part_ss_occ = self.models['ss_encoder'](batch['part_ss'])['z'].reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)
        xyz_list = []
        resolution = 64
        for i in range(gt_part_ss_occ.shape[0]):

            coords = torch.nonzero(batch['part_ss'][i, 0], as_tuple=False)
            xyz = coords / (resolution - 1) - 0.5
            xyz = xyz / batch['part_scale'][i] + batch['part_translation'][i][None]
            xyz_ss = torch.zeros(3, resolution, resolution, resolution, dtype=torch.float32).to(gt_part_ss_occ.device)
            xyz_ss[:,coords[:, 0], coords[:, 1], coords[:, 2]] = xyz.t()
            xyz_list.append(xyz_ss.unsqueeze(0))
        xyz_list = torch.cat(xyz_list, dim=0)
        with torch.no_grad():
            gt_part_ss_xyz = self.models['ss_encoder_xyz'](xyz_list).reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)

        gt_part_ss_latent = torch.cat([gt_part_ss_occ, gt_part_ss_xyz], dim=-1)
        cond_global_ss = self._encode_global_ss(batch)
        condition_args = (torch.cat([condition_args[0], cond_global_ss], dim=1),)
        gt_part_scale = torch.log(batch['part_scale'])[:,None]
        gt_part_translation = batch['part_translation'][:,None]
        gt_latent = {
            '6drotation_normalized': torch.zeros(cond_global_ss.shape[0], 1, 6, device=cond_global_ss.device),
            'scale': gt_part_scale.repeat(1, 1, 3),
            'translation': gt_part_translation,
            'translation_scale': gt_part_scale,
            'shape': gt_part_ss_latent.contiguous(),
        }
        return condition_args, condition_kwargs, gt_latent

    def _post_process_condition_args(self, condition_args):
        return condition_args

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            gt_global_ss_latent = self.models['ss_encoder'](batch['global_ss'])['z'].reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)
        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['ss_condition_embedder'].idx_emb[1:2, None]
        return cond_global_ss

    def training_step(self, batch, batch_idx):
        condition_args, condition_kwargs, x1 = self.get_input(batch)
        # x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        condition_args, condition_kwargs = self.get_input(batch)
        x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models["global_ss_condition_embedder"].parameters())  + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets[2].parameters())
        params = lora_params + other_params
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        return opt

class TrainingPartSSPipeline_compress(TrainingPartSSPipeline):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 用 nn.Module 包裹 Parameter，因为 self.models 是 ModuleDict
        emb_module = torch.nn.Module()
        emb_module.weight = torch.nn.Parameter(torch.empty(1, 1024))
        torch.nn.init.normal_(emb_module.weight, mean=0.0, std=1.0 / math.sqrt(1024))
        self.models['global_ss_condition_type_emb'] = emb_module
    
    def fuse_cond(self, cond_tokens):
        cond_crop = torch.cat([
            (cond_tokens[:,0:1369] + cond_tokens[:,2740:4109] + cond_tokens[:,5480:6849]) / 3,
            (cond_tokens[:,1369:1370] + cond_tokens[:,4109:4110]) / 2
        ], dim=1)
        cond_whole = torch.cat([
            (cond_tokens[:,1370:2739] + cond_tokens[:,4110:5479] + cond_tokens[:,6849:8218]) / 3,
            (cond_tokens[:,2739:2740] + cond_tokens[:,5479:5480]) / 2
        ], dim=1)
        return torch.cat([cond_crop, cond_whole], dim=1)

    def _encode_global_ss(self, batch):
        with torch.no_grad():
            gt_global_ss_latent = self.models['ss_encoder'](batch['global_ss'])['z'].reshape(batch['part_ss'].shape[0], 8, 4096).permute(0, 2, 1)
        cond_global_ss = self.models['global_ss_condition_embedder'](gt_global_ss_latent)
        cond_global_ss = cond_global_ss + self.models['global_ss_condition_type_emb'].weight[0:1, None]
        return cond_global_ss

    def _post_process_condition_args(self, condition_args):
        return (self.fuse_cond(condition_args[0]),)

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models["global_ss_condition_embedder"].parameters())  + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets.parameters()) + \
                       list(self.models['global_ss_condition_type_emb'].parameters()) + \
                       list(self.models['ss_condition_embedder'].idx_emb)
        params = lora_params + other_params
        opt = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0)
        return opt


ROTATION_6D_MEAN = torch.tensor(
    [
        -0.06366084883674913,
        0.008438224692279752,
        0.00017084786438302483,
        0.0007126610473540038,
        -0.0030916726538816417,
        0.5166093753457688,
    ]
)
ROTATION_6D_STD = torch.tensor(
    [
        0.6656971967514863,
        0.6787012271867754,
        0.30345010594844524,
        0.4394504420678794,
        0.39817973931717104,
        0.6176286868761914,
    ]
)

class TrainingMVSSPipeline(TrainingPipeline):

    def __init__(
        self, *args, depth_model, layout_post_optimization_method=None, clip_pointmap_beyond_scale=None,
        use_depth=False, cond_ctx_channels=3072, **kwargs
    ):
        self.depth_model = depth_model
        self.layout_post_optimization_method = layout_post_optimization_method
        self.clip_pointmap_beyond_scale = clip_pointmap_beyond_scale
        self.use_pointmap = True
        # depth-conditioning: 4th pointmap cond stream.
        self.use_depth = bool(use_depth)
        self.cond_ctx_channels = int(cond_ctx_channels)
        super().__init__(*args, **kwargs)
        for key in ['slat_generator', 'slat_decoder_gs', 'slat_decoder_gs_4',
                    'slat_decoder_mesh']:
            if key in self.models:
                del self.models[key]
        for key in ["slat_condition_embedder"]:
            if key in self.condition_embedders:
                del self.condition_embedders[key]
        # depth mode keeps MoGe RESIDENT (the `moge` training mode + wild inference);
        # otherwise it is deleted (the mask/pose model never needs depth at train time).
        if not self.use_depth:
            del self.depth_model
        self.models['ss_condition_embedder'] = deepcopy(self.condition_embedders['ss_condition_embedder'])
        del self.condition_embedders['ss_condition_embedder']
        self.models['multiview_cond'] = ModulatedMultiViewCond(
                                        1024,
                                        self.cond_ctx_channels,
                                        num_heads=16,
                                        mlp_ratio=4,
                                        attn_mode='full',
                                        use_checkpoint=False,
                                        use_rope=False,
                                        share_mod=False,
                                        qk_rms_norm=True,
                                        qk_rms_norm_cross=False,
                                    ).to(self.device)

        emb_module = torch.nn.Module()
        emb_module.weight = torch.nn.Parameter(torch.empty(2, self.cond_ctx_channels))
        torch.nn.init.normal_(emb_module.weight, mean=0.0, std=1.0 / math.sqrt(self.cond_ctx_channels))
        self.models['ref_src_emb'] = emb_module

        # --- dual-branch amodal mask output (replaces the pose output) ---
        self.mask_res = 64
        self.mask_loss_weight = 0.5
        self.models['mask_branch'] = MaskBranch(
            model_channels=1024, mask_res=self.mask_res, patch=4
        ).to(self.device)
        # raw-stream gate (zero-init): ramps the hand-aware (union-crop) DINO stream
        # into the cond's 3rd channel slot. Lives in self.models so it is moved by
        # `to()` and saved by the checkpoint filter.
        raw_gate = torch.nn.Module()
        raw_gate.weight = torch.nn.Parameter(torch.zeros(1))
        self.models['raw_gate'] = raw_gate.to(self.device)

        # --- 4th pointmap cond stream (PointPatchEmbed -> projection -> 1024) ---
        if self.use_depth:
            from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import ObjectCentricSSI
            # input_size 296: 296/8 = 37 -> 37^2 = 1369 tokens, aligned with the DINO streams.
            self.models['ss_condition_embedder'].module_list[2].input_size = 296
            self.ssi = ObjectCentricSSI(
                use_scene_scale=True, allow_scale_and_shift_override=True, scale_factor=1.0,
            )
            pointmap_gate = torch.nn.Module()
            pointmap_gate.weight = torch.nn.Parameter(torch.zeros(1))
            self.models['pointmap_gate'] = pointmap_gate.to(self.device)
            # per-step mixed-mode sampling over {rendered, moge, drop}
            self.pointmap_modes = ['rendered', 'moge', 'drop']
            self.pointmap_mode_weights = [1.0 / 3, 1.0 / 3, 1.0 / 3]

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> None:
        for model in self.models.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

        for model in self.condition_embedders.values():
            if dtype is not None and device is not None:
                model.to(device, dtype)
            elif device is not None:
                model.to(device)
            elif dtype is not None:
                model.type(dtype)

        # keep MoGe on the right device (fp32; autocast handles precision in _moge_pointmap)
        if self.use_depth and hasattr(self, 'depth_model') and device is not None:
            self.depth_model.model.to(device)

    def fuse_cond(self, cond_tokens):
        cond = torch.cat([
            torch.cat([cond_tokens[:,0:1369], cond_tokens[:,1370:2739], cond_tokens[:,2740:4109]], dim=-1),
            torch.cat([cond_tokens[:,1369:1370], cond_tokens[:,2739:2740], cond_tokens[:,1369:1370]], dim=-1)
        ], dim=1)
        return cond
    
    def _post_process_condition_args(self, condition_args):
        return (self.fuse_cond(condition_args[0]),)

    # ------------------------------------------------------------------
    # Vectorised square-bbox crop (ported from ReconViaGen prepare_batch_images).
    # Returns the affine matrix so the SAME crop can be applied to several
    # tensors at several resolutions.
    # ------------------------------------------------------------------
    @staticmethod
    def _crop_affine_from_mask(masks, padding_factor=1.1):
        # masks: (M, 1, H, W) -> affine A (M, 2, 3) for a square bbox crop
        M, _, H, W = masks.shape
        device = masks.device
        mask_bool = masks[:, 0] > 0.5  # (M, H, W)
        inf = torch.tensor(1e5, device=device)
        ys = torch.arange(H, device=device).view(1, H, 1).expand(M, H, W).float()
        xs = torch.arange(W, device=device).view(1, 1, W).expand(M, H, W).float()
        y0 = torch.where(mask_bool, ys, inf).flatten(1).min(1)[0].clamp(max=H - 1)
        x0 = torch.where(mask_bool, xs, inf).flatten(1).min(1)[0].clamp(max=W - 1)
        y1 = torch.where(mask_bool, ys, -inf).flatten(1).max(1)[0].clamp(min=0)
        x1 = torch.where(mask_bool, xs, -inf).flatten(1).max(1)[0].clamp(min=0)
        no_fg = (mask_bool.flatten(1).sum(1) == 0)
        y0 = torch.where(no_fg, torch.zeros_like(y0), y0)
        x0 = torch.where(no_fg, torch.zeros_like(x0), x0)
        y1 = torch.where(no_fg, torch.full_like(y1, H - 1), y1)
        x1 = torch.where(no_fg, torch.full_like(x1, W - 1), x1)
        cy = (y0 + y1) * 0.5; cx = (x0 + x1) * 0.5
        side = torch.max(y1 - y0, x1 - x0) * padding_factor
        y0n = (cy - side / 2).clamp(0, H - 1); y1n = (cy + side / 2).clamp(0, H - 1)
        x0n = (cx - side / 2).clamp(0, W - 1); x1n = (cx + side / 2).clamp(0, W - 1)
        scale_y = (y1n - y0n) / (H - 1); scale_x = (x1n - x0n) / (W - 1)
        trans_y = (y0n + y1n - (H - 1)) / (H - 1); trans_x = (x0n + x1n - (W - 1)) / (W - 1)
        A = torch.zeros(M, 2, 3, device=device, dtype=torch.float32)
        A[:, 0, 0] = scale_x; A[:, 1, 1] = scale_y
        A[:, 0, 2] = trans_x; A[:, 1, 2] = trans_y
        return A

    @staticmethod
    def _grid_crop(x, A, resolution, mode='bilinear'):
        B, C = x.shape[0], x.shape[1]
        grid = F.affine_grid(A, [B, C, resolution, resolution], align_corners=False)
        return F.grid_sample(x, grid.to(x.dtype), mode=mode,
                             align_corners=False, padding_mode='zeros')

    @torch.no_grad()
    def _encode_dino(self, dino, img):
        # img: (M, C, 518, 518) in [0,1]; -> (M, 1369, 1024) patch tokens (drop cls)
        return dino(img)[:, 1:]

    def _zero_pose_latents(self, b, device, dtype):
        """Inert zero latents for the kept (but unused) pose modalities so
        project_input still finds them. Shape is `protect_modality`, so these
        zero tokens never influence the shape/mask output."""
        backbone = self.models['ss_generator'].reverse_fn.backbone
        out = {}
        for name, latent in backbone.latent_mapping.items():
            if name == 'shape':
                continue
            token_len = latent.pos_emb.shape[0]
            in_ch = latent.input_layer.in_features
            out[name] = torch.zeros(b, token_len, in_ch, device=device, dtype=dtype)
        return out

    # ------------------------------------------------------------------
    # depth/pointmap stream helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _unproject_rendered(self, depth, aov):
        """depth (M,H,W) perpendicular z (1e10=bg), aov (M,) horizontal FOV ->
        (M,3,H,W) OpenCV-frame pointmap, NaN outside the valid region."""
        M, H, W = depth.shape
        dev = depth.device
        fx = (W / 2.0) / torch.tan(0.5 * aov).clamp_min(1e-6)        # (M,) ; fx=fy
        cx = W / 2.0; cy = H / 2.0
        xs = torch.arange(W, device=dev).view(1, 1, W).float()
        ys = torch.arange(H, device=dev).view(1, H, 1).float()
        valid = (depth < 100.0) & torch.isfinite(depth)
        f = fx.view(M, 1, 1)
        pm = torch.stack([(xs - cx) / f * depth, (ys - cy) / f * depth, depth], dim=1)
        return torch.where(valid.unsqueeze(1), pm, torch.full_like(pm, float('nan')))

    @torch.no_grad()
    def _to_p3d_frame(self, pm):
        """(M,3,H,W) camera/OpenCV frame -> PyTorch3D frame (matches compute_pointmap)."""
        M, _, H, W = pm.shape
        R = camera_to_pytorch3d_camera(device=pm.device).rotation
        tf = Transform3d().rotate(R).to(pm.device)
        flat = pm.permute(0, 2, 3, 1).reshape(M, -1, 3)
        return tf.transform_points(flat).reshape(M, H, W, 3).permute(0, 3, 1, 2)

    @torch.no_grad()
    def _moge_pointmap(self, rgb):
        """rgb (M,3,H,W) in [0,1] -> (M,3,H,W) PyTorch3D-frame MoGe pointmap (full frame)."""
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.depth_model(rgb)
        pm = output['pointmaps'].float()                            # (M,H,W,3) MoGe native
        M, H, W, _ = pm.shape
        R = camera_to_pytorch3d_camera(device=pm.device).rotation
        tf = Transform3d().rotate(R).to(pm.device)
        return tf.transform_points(pm.reshape(M, -1, 3)).reshape(M, H, W, 3).permute(0, 3, 1, 2)

    @torch.no_grad()
    def _ssi_normalize_pm(self, pm, objmask):
        """Per-view ObjectCentricSSI on the object mask. pm (M,3,H,W) NaN-outside,
        objmask (M,1,H,W) -> normalized (M,3,H,W). Views with an EMPTY object mask
        (fully-occluded object — common in HOI) return all-NaN, since raw
        ObjectCentricSSI calls `.max()` on the empty masked-point set and would crash."""
        outs = []
        nan_i = lambda i: torch.full_like(pm[i], float('nan'))
        for i in range(pm.shape[0]):
            # skip empty / near-degenerate masks (raw ObjectCentricSSI does an empty
            # .max() and, for collapsed points, a singular torch.inverse); both crash.
            if (objmask[i] > 0.5).sum() < 16:
                outs.append(nan_i(i)); continue
            try:
                outs.append(self.ssi.normalize(pm[i], objmask[i]).pointmap)
            except Exception:
                outs.append(nan_i(i))
        return torch.stack(outs, dim=0)

    def _build_pointmap_feat(self, batch, n_use, b, h, w, dev, rgb_flat, obj_flat, obj_c, A, ref_feat):
        """4th cond stream (M,1369,1024). Per-step mixed mode {rendered, moge, drop}.

        The mode is seeded by `global_step` so it is IDENTICAL across DDP ranks — otherwise
        ranks would use different pointmap params each step and DDP grad all-reduce would hang
        (the pointmap embedder gets grad on rendered/moge steps but not on drop steps)."""
        step = int(getattr(self, 'global_step', 0) or 0)
        mode = random.Random(step).choices(
            self.pointmap_modes, weights=self.pointmap_mode_weights, k=1)[0]
        if mode == 'drop':
            return torch.zeros_like(ref_feat)
        if mode == 'rendered' and ('depth' in batch) and ('camera_angle_x' in batch):
            depth = batch['depth'][:, :n_use].reshape(b * n_use, h, w).to(dev).float()
            aov = batch['camera_angle_x'][:, :n_use].reshape(b * n_use).to(dev).float()
            pm = self._to_p3d_frame(self._unproject_rendered(depth, aov))
        else:                                   # moge (also the rendered-missing fallback)
            pm = self._moge_pointmap(rgb_flat)
        objm = (obj_flat > 0.5)
        pm = torch.where(objm, pm, torch.full_like(pm, float('nan')))   # object-mask (full res)
        pm = self._ssi_normalize_pm(pm, obj_flat)                       # per-view ObjectCentricSSI
        # SAME union crop as the DINO streams (token alignment); nearest preserves NaN.
        # Re-mask with the cropped obj mask so grid_sample zero-padding becomes NaN (invalid).
        pm_c = self._grid_crop(pm, A, 518, 'nearest')
        pm_c = torch.where(obj_c > 0.5, pm_c, torch.full_like(pm_c, float('nan')))
        emb = self.models['ss_condition_embedder'].module_list[2](pm_c)        # (M,1369,Dpm)
        feat = self.models['ss_condition_embedder'].projection_nets[2](emb)    # (M,1369,1024)
        return feat * self.models['pointmap_gate'].weight

    def get_input(self, batch):
        images = batch['image']        # (B, N, 3, H, W)
        obj    = batch['mask']         # (B, N, 1, H, W) visible object
        hand   = batch['hand_mask']    # (B, N, 1, H, W)
        full   = batch['full_mask']    # (B, N, 1, H, W) amodal silhouette
        b, n, c, h, w = images.shape
        n_use = random.randint(1, n)
        images, obj, hand, full = images[:, :n_use], obj[:, :n_use], hand[:, :n_use], full[:, :n_use]

        dev = self.device
        rgb_flat   = images.reshape(b * n_use, c, h, w).to(dev).float()
        obj_flat   = obj.reshape(b * n_use, 1, h, w).to(dev).float()
        hand_flat  = hand.reshape(b * n_use, 1, h, w).to(dev).float()
        full_flat  = full.reshape(b * n_use, 1, h, w).to(dev).float()
        union_flat = ((obj_flat + hand_flat) > 0.5).float()

        # ONE shared union (obj∪hand) crop per view → 3 maskings → DINO.
        # Sharing the crop keeps the 3072 channel-concat token-aligned.
        A = self._crop_affine_from_mask(union_flat, padding_factor=1.1)
        rgb_c   = self._grid_crop(rgb_flat,   A, 518, 'bilinear')
        obj_c   = self._grid_crop(obj_flat,   A, 518, 'nearest')
        union_c = self._grid_crop(union_flat, A, 518, 'nearest')
        image_img = rgb_c * obj_c       # object only (hand blacked)
        raw_img   = rgb_c * union_c     # object + hand (hand-aware)
        mask_img  = obj_c               # silhouette (1ch)

        img_dino  = self.models['ss_condition_embedder'].module_list[0]
        mask_dino = self.models['ss_condition_embedder'].module_list[1]
        image_feat = self._encode_dino(img_dino,  image_img)   # (B*n, 1369, 1024)
        mask_feat  = self._encode_dino(mask_dino, mask_img)
        raw_feat   = self._encode_dino(img_dino,  raw_img)

        # cond = [image | mask | raw·raw_gate] (+ pointmap·pointmap_gate when depth)
        raw_gate = self.models['raw_gate'].weight
        if self.use_depth:
            pm_feat = self._build_pointmap_feat(
                batch, n_use, b, h, w, dev, rgb_flat, obj_flat, obj_c, A, image_feat)
            ctx = torch.cat([image_feat, mask_feat, raw_feat * raw_gate, pm_feat], dim=-1)
            ctx = ctx.reshape(b, n_use, -1, self.cond_ctx_channels)   # (B*n -> B,n,1369,4096)
        else:
            ctx = torch.cat([image_feat, mask_feat, raw_feat * raw_gate], dim=-1)  # (B*n,1369,3072)
            ctx = ctx.reshape(b, n_use, -1, 3072)

        # ref/src view tags + multi-view fusion (multiview_cond unchanged)
        ref_src = self.models['ref_src_emb'].weight  # (2, 3072)
        ref_cond = ctx[:, :1] + ref_src[0:1, None, None]
        if n_use == 1:
            condition = ref_cond
        else:
            src_cond = ctx[:, 1:] + ref_src[1:, None, None]
            condition = torch.cat([ref_cond, src_cond], dim=1)
        cond = self.models['multiview_cond'](condition)        # (B, 4096, 1024)

        # raw_ctx for the mask branch: ungated raw DINO in the same union frame
        raw_ctx = raw_feat.reshape(b, n_use, -1, 1024)

        # shape GT latent (frozen SS-VAE on the canonical voxel grid)
        with torch.no_grad():
            enc_dtype = next(self.models['ss_encoder'].parameters()).dtype
            gt_ss = self.models['ss_encoder'](batch['global_ss'].to(dev).to(enc_dtype))['z']
            gt_ss = gt_ss.reshape(b, 8, 4096).permute(0, 2, 1).float()

        # amodal mask GT in the SAME union frame, mask_res², {-1, +1}
        full_c = self._grid_crop(full_flat, A, self.mask_res, 'nearest')   # (B*n,1,64,64)
        mask01 = (full_c > 0.5).float().reshape(b, n_use, 1, self.mask_res, self.mask_res)
        mask_x1 = mask01 * 2.0 - 1.0

        return cond, raw_ctx, gt_ss.contiguous(), mask_x1.float(), n_use

    def training_step(self, batch, batch_idx):
        cond, raw_ctx, shape_x1, mask_x1, n_use = self.get_input(batch)
        gen = self.models['ss_generator']
        b = shape_x1.shape[0]

        # one t per sample (shared by shape + mask); rectified flow (sigma_min=0)
        t = gen._generate_t({'shape': shape_x1}).to(shape_x1.device)   # (b,)

        x0_s = torch.randn_like(shape_x1)
        tb_s = t.view(b, *([1] * (shape_x1.dim() - 1)))
        x_t_s = (1 - tb_s) * x0_s + tb_s * shape_x1
        tgt_s = shape_x1 - x0_s

        x0_m = torch.randn_like(mask_x1)
        tb_m = t.view(b, *([1] * (mask_x1.dim() - 1)))
        x_t_m = (1 - tb_m) * x0_m + tb_m * mask_x1
        tgt_m = mask_x1 - x0_m
        mask_tokens = self.models['mask_branch'].to_tokens(x_t_m, raw_ctx)

        latents = {'shape': x_t_s}
        latents.update(self._zero_pose_latents(b, x_t_s.device, x_t_s.dtype))

        # Take the CFG wrapper's TRAINING path (random cond-drop), not the
        # inference path (which calls get_strength on a tensor t and errors).
        # FlowMatching.loss sets this flag; we bypass .loss(), so set it here.
        gen.reverse_fn.training = True
        pred = gen.reverse_fn(latents, t * gen.time_scale, cond, mask_tokens=mask_tokens)
        v_shape = pred['shape']
        v_mask = self.models['mask_branch'].from_tokens(pred['mask_out'], n_use)

        loss_shape = F.mse_loss(v_shape, tgt_s)
        per = ((v_mask - tgt_m) ** 2).flatten(1).mean(1)           # (b,)
        loss_mask = (self.mask_loss_weight * (1.0 + 2.0 * t) * per).mean()
        loss = loss_shape + loss_mask
        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_shape', loss_shape, prog_bar=True)
        self.log('loss_mask', loss_mask, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        condition_args, condition_kwargs = self.get_input(batch)
        x1 = batch['gt_latent']
        total_loss, detail_losses = self.models['ss_generator'].loss(x1, *condition_args, **condition_kwargs)
        self.log('train_loss', total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        lora_params = [p for p in self.models['ss_generator'].parameters() if p.requires_grad]
        other_params = list(self.models['multiview_cond'].parameters())  + \
                       list(self.models['ss_condition_embedder'].module_list[2].parameters()) + \
                       list(self.models['ss_condition_embedder'].projection_nets.parameters()) + \
                       list(self.models['ref_src_emb'].parameters()) + \
                       [self.models['ss_condition_embedder'].idx_emb]
        opt = torch.optim.AdamW([
            {"params": lora_params, "lr": 5e-5},
            {"params": other_params, "lr": 1e-4},
        ], weight_decay=0.0)
        return opt