# Copyright (c) Meta Platforms, Inc. and affiliates.
from functools import partial
from typing import *
from torch.utils import _pytree
import torch
import torch.nn as nn
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from collections import namedtuple
from ..modules.utils import FP16_TYPE
from ..modules.transformer import (
    MOTModulatedTransformerCrossBlock,
    ModulatedTransformerCrossBlock,
    ModulatedTransformerCrossBlock_woT
)
from sam3d_objects.data.utils import (
    tree_reduce_unique,
)
from .timestep_embedder import TimestepEmbedder
from omegaconf import OmegaConf

class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        freeze_shared_parameters: bool = False,
        is_shortcut_model: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = FP16_TYPE if use_fp16 else torch.float32
        self.is_shortcut_model = is_shortcut_model
        if is_shortcut_model:
            self.d_embedder = TimestepEmbedder(model_channels) # for shortcut model

        self.t_embedder = TimestepEmbedder(model_channels, freeze=freeze_shared_parameters)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        self.blocks = nn.ModuleList(
            [
                MOTModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode="full",
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                    latent_names=self.latent_names,
                    freeze_shared_parameters=freeze_shared_parameters
                )
                for _ in range(num_blocks)
            ]
        )

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero init like controlnet, for MLP should only zero 
        # the weight of the last layer only
        if self.is_shortcut_model:
            nn.init.constant_(self.d_embedder.mlp[2].weight, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def _cast_type(self, x, dtype):
        return x.type(dtype)

    def forward(
        self,
        h: Dict,
        t: torch.Tensor,
        cond: torch.Tensor,
        d: torch.Tensor = None,
    ) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        if d is not None:
            d_emb = self.d_embedder(d)
            t_emb = t_emb + d_emb
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        input_dtype = tree_reduce_unique(lambda tensor: tensor.dtype, h)
        t_emb = t_emb.type(self.dtype)
        h = _pytree.tree_map(
            partial(self._cast_type, dtype=self.dtype),
            h,
        )
        cond = cond.type(self.dtype)

        for block in self.blocks:
            h = block(h, t_emb, cond)

        h = _pytree.tree_map(
            partial(self._cast_type, dtype=input_dtype),
            h,
        )

        return h


class SparseStructureFlowTdfyWrapper(SparseStructureFlowModel):
    def __init__(
        self,
        latent_mapping: dict,
        latent_share_transformer: dict = {},
        *args,
        **kwargs,
    ):
        condition_embedder = kwargs.pop("condition_embedder", None)
        # if enabled, model will record the condition_shape in one run and uses zeros for all that afterwards
        force_zeros_cond = kwargs.pop("force_zeros_cond", False)
        # backward compatible to models trained before PR #87
        kwargs.pop("shape_attend_pose", None)
        merge_latent_names = [i for _, v in latent_share_transformer.items() for i in v]
        self.latent_names = [
            latent_name
            for latent_name in list(latent_mapping.keys())
            if latent_name not in merge_latent_names
        ] + list(latent_share_transformer.keys())
        super().__init__(*args, **kwargs)
        if condition_embedder is not None:
            self.condition_embedder = condition_embedder
        else:
            self.condition_embedder = lambda x: x
        self.force_zeros_cond = force_zeros_cond
        self.latent_mapping = nn.ModuleDict(latent_mapping)
        if not isinstance(latent_share_transformer, dict):
            self.latent_share_transformer = OmegaConf.to_container(latent_share_transformer)
        else:
            self.latent_share_transformer = latent_share_transformer
        self.input_latent_mappings = list(self.latent_mapping.keys())

    def forward(
        self,
        latents_dict: dict,
        t: torch.Tensor,
        *condition_args,
        **condition_kwargs,
    ) -> dict:
        d = condition_kwargs.pop("d", None)
        # Pre-tokenized per-view mask tokens (B, M, model_channels) from MaskBranch
        # (dual-branch amodal mask). Appended to the `shape` token stream so they
        # share self-attention with the shape tokens and cross-attend to the same
        # cond, then split back out and returned as 'mask_out'. None => SS-only
        # (original behaviour, bit-identical).
        mask_tokens = condition_kwargs.pop("mask_tokens", None)

        cfg_activate = condition_kwargs.pop("cfg", False)
        if self.force_zeros_cond and cfg_activate:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)
            cond = cond * 0
        else:
            cond = self.condition_embedder(*condition_args, **condition_kwargs)

        # concatenate input
        latent_dict = self.project_input(latents_dict)

        n_shape = None
        if mask_tokens is not None:
            n_shape = latent_dict["shape"].shape[1]
            latent_dict["shape"] = torch.cat(
                [latent_dict["shape"], mask_tokens.type(latent_dict["shape"].dtype)], dim=1
            )

        output = super().forward(latent_dict, t, cond, d)

        mask_out = None
        if mask_tokens is not None:
            mask_out = output["shape"][:, n_shape:]
            output["shape"] = output["shape"][:, :n_shape]

        # split input to multiple output modalities
        output_latents = self.project_output(output)
        if mask_out is not None:
            output_latents["mask_out"] = mask_out

        return output_latents

    def project_input(
        self,
        latents_dict: Dict,
    ) -> Dict:
        # concatenate input from multiple modalities
        latent_dict = {}
        for latent_name in self.input_latent_mappings:
            assert (
                latent_name in latents_dict
            ), f"'{latent_name}' not found in latents_dict"
            latent_input = latents_dict[latent_name]
            x = self.latent_mapping[latent_name].to_input(latent_input)
            latent_dict[latent_name] = x

        latent_dict = self.merge_latent_share_transformer(latent_dict)
        return latent_dict

    def project_output(self, output: Dict) -> Dict:
        output = self.split_latent_share_transformer(output)
        output_latents = {}
        for latent_name in self.input_latent_mappings:
            latent = self.latent_mapping[latent_name].to_output(output[latent_name])
            output_latents[latent_name] = latent

        return output_latents

    def merge_latent_share_transformer(self, latent_dict):
        visited_latent_names = set()
        return_dict = {}
        for merged_name, latent_names in self.latent_share_transformer.items():
            tensors = []
            for latent_name in latent_names:
                visited_latent_names.add(latent_name)
                tensors.append(latent_dict[latent_name])
            tensors = torch.cat(tensors, dim=1)
            return_dict[merged_name] = tensors

        for latent_name in latent_dict:
            if latent_name not in visited_latent_names:
                return_dict[latent_name] = latent_dict[latent_name]

        return return_dict

    def split_latent_share_transformer(self, output_latents):
        return_dict = {}
        visited_latent_names = set()
        for merged_name, latent_names in self.latent_share_transformer.items():
            start = 0
            visited_latent_names.add(merged_name)
            tensors = output_latents[merged_name]
            for latent_name in latent_names:
                token_len = self.latent_mapping[latent_name].pos_emb.shape[0]
                latent = tensors[:, start : start + token_len]
                return_dict[latent_name] = latent
                start += token_len

        for latent_name in output_latents:
            if latent_name not in visited_latent_names:
                return_dict[latent_name] = output_latents[latent_name]

        return return_dict

class ModulatedMultiViewCond(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        num_init_tokens: int = 4096,
        dtype: Optional[torch.dtype] = torch.float32,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.cond_blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock_woT(
                channels,
                ctx_channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_mode=attn_mode,
                use_checkpoint=use_checkpoint,
                use_rope=use_rope,
                share_mod=share_mod,
                qk_rms_norm=qk_rms_norm,
                qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(4)
        ])
        self.use_fp16 = use_fp16
        if use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.multiview_cond_tokens = nn.Parameter(torch.randn(1, num_init_tokens, channels).to(dtype))
        nn.init.normal_(self.multiview_cond_tokens, std=1e-6)
        if use_fp16:
            self.convert_to_fp16()


    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.cond_blocks.apply(convert_module_to_f16)
        self.multiview_cond_tokens = nn.Parameter(self.multiview_cond_tokens.data.to(self.dtype))
    def forward(self, condition: torch.Tensor):
        b = condition.shape[0]
        cond = self.multiview_cond_tokens.repeat(b, 1, 1)
        # channel-agnostic: last dim = ctx_channels (3072 without depth / 4096 with depth)
        x = condition.reshape(b, -1, condition.shape[-1])
        for block in self.cond_blocks:
            cond = block(cond, x)
        return cond
