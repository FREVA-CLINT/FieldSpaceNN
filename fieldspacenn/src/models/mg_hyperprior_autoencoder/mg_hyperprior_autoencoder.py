"""Hyper-prior FieldSpace autoencoder for HEALPix climate tensors."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..mg_transformer.mg_base_model import MG_base_model, create_encoder_decoder_block, create_missing_zooms
from ...modules.field_space.field_space_base import DiffDecoder
from ...modules.hyperprior.distributions import MGDiagonalGaussianDistribution
from ...modules.hyperprior.entropy_adapters import (
    FieldSpaceEntropyBottleneckAdapter,
    FieldSpaceGaussianConditionalAdapter,
)
from ...modules.hyperprior.projections import MultiZoomPointwiseProjection
from ...modules.hyperprior.quantization import map_nested, quantize_training


class MGHyperpriorFieldSpaceAutoEncoder(MG_base_model):
    """CRA5-style hyper-prior compression model using FieldSpace blocks."""

    VALID_STAGES = {"pretrain", "entropy_finetune", "joint"}

    def __init__(
        self,
        mgrids: Any,
        in_zooms: Sequence[int],
        analysis_block_configs: Mapping[str, Any],
        synthesis_block_configs: Mapping[str, Any],
        hyper_analysis_block_configs: Mapping[str, Any],
        hyper_synthesis_block_configs: Mapping[str, Any],
        in_features: int = 1,
        out_features: int = 1,
        latent_features: int = 64,
        hyperlatent_features: int = 64,
        n_groups_variables: Sequence[int] = (1,),
        sample_posterior: bool = True,
        detach_hyper_analysis_input: bool = True,
        entropy_model: str = "compressai",
        scale_min: float = 1e-9,
        use_ste_quantization: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(mgrids)
        if entropy_model != "compressai":
            raise ValueError(f"Unsupported entropy_model '{entropy_model}'. Only 'compressai' is implemented.")

        self.in_zooms = [int(zoom) for zoom in in_zooms]
        self.max_zoom = max(self.in_zooms)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.latent_features = int(latent_features)
        self.hyperlatent_features = int(hyperlatent_features)
        self.n_groups_variables = list(n_groups_variables)
        self.sample_posterior = bool(sample_posterior)
        self.detach_hyper_analysis_input = bool(detach_hyper_analysis_input)
        self.use_ste_quantization = bool(use_ste_quantization)

        self.analysis_blocks, self.bottleneck_zooms, analysis_features = self._build_block_stack(
            analysis_block_configs,
            self.in_zooms,
            [self.in_features] * len(self.in_zooms),
            self.n_groups_variables,
            **kwargs,
        )
        self.moment_projection = MultiZoomPointwiseProjection(
            self.bottleneck_zooms,
            analysis_features,
            2 * self.latent_features,
        )

        self.hyper_analysis_blocks, self.hyper_bottleneck_zooms, hyper_analysis_features = self._build_block_stack(
            hyper_analysis_block_configs,
            self.bottleneck_zooms,
            [self.latent_features] * len(self.bottleneck_zooms),
            self.n_groups_variables,
            **kwargs,
        )
        self.hyperlatent_projection = MultiZoomPointwiseProjection(
            self.hyper_bottleneck_zooms,
            hyper_analysis_features,
            self.hyperlatent_features,
        )

        self.hyper_synthesis_blocks, hyper_synthesis_zooms, hyper_synthesis_features = self._build_block_stack(
            hyper_synthesis_block_configs,
            self.hyper_bottleneck_zooms,
            [self.hyperlatent_features] * len(self.hyper_bottleneck_zooms),
            self.n_groups_variables,
            **kwargs,
        )
        self.gaussian_params_projection = MultiZoomPointwiseProjection(
            hyper_synthesis_zooms,
            hyper_synthesis_features,
            2 * self.latent_features,
        )

        self.synthesis_blocks, synthesis_zooms, synthesis_features = self._build_block_stack(
            synthesis_block_configs,
            self.bottleneck_zooms,
            [self.latent_features] * len(self.bottleneck_zooms),
            self.n_groups_variables,
            **kwargs,
        )
        self.reconstruction_projection = MultiZoomPointwiseProjection(
            synthesis_zooms,
            synthesis_features,
            self.out_features,
        )
        self.output_zooms = synthesis_zooms

        channels_by_key = {
            f"group{group_idx}_zoom{zoom}": self.hyperlatent_features
            for group_idx in range(len(self.n_groups_variables))
            for zoom in self.hyper_bottleneck_zooms
        }
        self.entropy_bottleneck_adapter = FieldSpaceEntropyBottleneckAdapter(channels_by_key=channels_by_key)
        self.gaussian_conditional_adapter = FieldSpaceGaussianConditionalAdapter(scale_min=scale_min)
        self.decoder = DiffDecoder()

    def _build_block_stack(
        self,
        block_configs: Mapping[str, Any],
        in_zooms: Sequence[int],
        in_features: int | Sequence[int],
        n_groups_variables: Sequence[int],
        **kwargs: Any,
    ) -> Tuple[nn.ModuleDict, List[int], List[int]]:
        """Build a FieldSpace block stack with the shared factory."""

        module_dict = nn.ModuleDict()
        current_zooms = [int(zoom) for zoom in in_zooms]
        if isinstance(in_features, int):
            current_features = [int(in_features)] * len(current_zooms)
        else:
            current_features = [int(feature) for feature in in_features]
        for block_key, block_conf in (block_configs or {}).items():
            if not isinstance(block_key, str):
                raise TypeError("FieldSpace block config keys must be strings.")
            block = create_encoder_decoder_block(
                block_conf,
                current_zooms,
                current_features,
                n_groups_variables,
                self.grid_layers,
                **kwargs,
            )
            module_dict[block_key] = block
            current_zooms = [int(zoom) for zoom in block.out_zooms]
            current_features = [int(feature) for feature in block.out_features]
        return module_dict, current_zooms, current_features

    def _run_blocks(
        self,
        blocks: nn.ModuleDict,
        groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any],
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        output = [dict(group) if group is not None else None for group in groups]
        for block in blocks.values():
            output = block(output, sample_configs=sample_configs, mask_groups=mask_groups, emb_groups=emb_groups)
        return output

    def encode_posterior(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> MGDiagonalGaussianDistribution:
        """Run analysis blocks and return the primary latent posterior."""

        if mask_zooms_groups is None:
            mask_zooms_groups = [
                {int(zoom): torch.ones_like(tensor, dtype=torch.bool) for zoom, tensor in group.items()}
                if group is not None
                else None
                for group in x_zooms_groups
            ]
        if emb_groups is None:
            emb_groups = [{} for _ in x_zooms_groups]
        x_zooms_groups, mask_zooms_groups, emb_groups, sample_configs = create_missing_zooms(
            x_zooms_groups,
            self.in_zooms,
            mask_zooms_groups,
            emb_groups,
            sample_configs=sample_configs,
        )
        encoded = self._run_blocks(self.analysis_blocks, x_zooms_groups, sample_configs, mask_zooms_groups, emb_groups)
        moments = self.moment_projection(encoded)
        return MGDiagonalGaussianDistribution(moments)

    def sample_latent(self, posterior: MGDiagonalGaussianDistribution) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Sample or take the mode of the primary posterior."""

        if self.training and self.sample_posterior:
            return posterior.sample()
        return posterior.mode()

    def hyper_encode(
        self,
        y_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        stage: Optional[str] = None,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Run hyper-analysis blocks and project to hyperlatents."""

        hyper_input = y_groups
        if self.training and self.detach_hyper_analysis_input and stage in {"entropy_finetune", "joint"}:
            hyper_input = map_nested(lambda tensor: tensor.detach(), y_groups)
        z_features = self._run_blocks(self.hyper_analysis_blocks, hyper_input, sample_configs, mask_groups, emb_groups)
        return self.hyperlatent_projection(z_features)

    def hyper_decode(
        self,
        z_hat_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[List[Optional[Dict[int, torch.Tensor]]], List[Optional[Dict[int, torch.Tensor]]]]:
        """Run hyper-synthesis and split Gaussian scales and means."""

        params_features = self._run_blocks(self.hyper_synthesis_blocks, z_hat_groups, sample_configs, mask_groups, emb_groups)
        params = self.gaussian_params_projection(params_features)
        scales_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        means_groups: List[Optional[Dict[int, torch.Tensor]]] = []
        for group in params:
            if group is None:
                scales_groups.append(None)
                means_groups.append(None)
                continue
            scales_group: Dict[int, torch.Tensor] = {}
            means_group: Dict[int, torch.Tensor] = {}
            for zoom, tensor in group.items():
                scales, means = torch.chunk(tensor, 2, dim=-1)
                scales_group[int(zoom)] = scales
                means_group[int(zoom)] = means
            scales_groups.append(scales_group)
            means_groups.append(means_group)
        return scales_groups, means_groups

    def ae_encode(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        mode: str = "y",
    ) -> Any:
        """Encode inputs and return posterior, ``y``, or ``y_hat``."""

        posterior = self.encode_posterior(x_zooms_groups, sample_configs, mask_groups, emb_groups)
        if mode == "posterior":
            return posterior
        y = self.sample_latent(posterior)
        if mode == "y":
            return y
        if mode == "y_hat":
            quant_mode = "ste" if self.use_ste_quantization else "noise"
            if not self.training:
                quant_mode = "ste"
            return map_nested(lambda tensor: quantize_training(tensor, quant_mode), y)
        raise ValueError(f"Unknown ae_encode mode '{mode}'.")

    def ae_decode(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        out_zoom: Optional[int] = None,
    ) -> List[Optional[Dict[int, torch.Tensor]]]:
        """Decode primary latents into reconstructed FieldSpace tensors."""

        decoded = self._run_blocks(self.synthesis_blocks, x_zooms_groups, sample_configs, mask_groups, emb_groups)
        decoded = self.reconstruction_projection(decoded)
        if out_zoom is not None:
            out_groups: List[Optional[Dict[int, torch.Tensor]]] = []
            for group in decoded:
                out_groups.append(self.decoder(group, sample_configs=sample_configs, out_zoom=out_zoom) if group else {})
            decoded = out_groups
        return decoded

    def forward(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        out_zoom: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the requested training stage."""

        stage = stage or "joint"
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Unknown hyperprior training stage '{stage}'.")

        posterior = self.encode_posterior(x_zooms_groups, sample_configs, mask_zooms_groups, emb_groups)
        y = self.sample_latent(posterior)

        if stage == "pretrain":
            y_hat = y
            x_hat = self.ae_decode(y_hat, sample_configs, mask_zooms_groups, emb_groups, out_zoom)
            return {
                "x_hat": x_hat,
                "likelihoods": {"y": None, "z": None},
                "posterior": posterior,
                "y": y,
                "y_hat": y_hat,
                "z": None,
                "z_hat": None,
                "gaussian_params": {"scales": None, "means": None},
            }

        z = self.hyper_encode(y, sample_configs, mask_zooms_groups, emb_groups, stage=stage)
        z_hat, z_likelihoods = self.entropy_bottleneck_adapter(z)
        scales, means = self.hyper_decode(z_hat, sample_configs, mask_zooms_groups, emb_groups)
        y_hat, y_likelihoods = self.gaussian_conditional_adapter(y, scales, means)
        x_hat = self.ae_decode(y_hat, sample_configs, mask_zooms_groups, emb_groups, out_zoom)
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "posterior": posterior,
            "y": y,
            "y_hat": y_hat,
            "z": z,
            "z_hat": z_hat,
            "gaussian_params": {"scales": scales, "means": means},
        }

    @torch.no_grad()
    def compress(
        self,
        x_zooms_groups: Sequence[Optional[Dict[int, torch.Tensor]]],
        sample_configs: Mapping[int, Any] = {},
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        sample_posterior_for_compress: bool = False,
    ) -> Dict[str, Any]:
        """Entropy-code latents with the hyper-prior path."""

        was_training = self.training
        self.eval()
        posterior = self.encode_posterior(x_zooms_groups, sample_configs, mask_zooms_groups, emb_groups)
        y = posterior.sample() if sample_posterior_for_compress else posterior.mode()
        z = self.hyper_encode(y, sample_configs, mask_zooms_groups, emb_groups, stage="joint")
        self.update(force=False)
        z_strings, z_metadata, z_specs = self.entropy_bottleneck_adapter.compress(z)
        z_hat = self.entropy_bottleneck_adapter.decompress(z_strings, z_metadata, z_specs, device=self._device())
        scales, means = self.hyper_decode(z_hat, sample_configs, mask_zooms_groups, emb_groups)
        y_strings, y_metadata, y_specs = self.gaussian_conditional_adapter.compress(y, scales, means)
        if was_training:
            self.train()
        return {
            "strings": {"y": y_strings, "z": z_strings},
            "metadata": {
                "y": y_metadata,
                "z": z_metadata,
                "y_specs": y_specs,
                "z_specs": z_specs,
                "sample_configs": sample_configs,
                "embedding_side_info": self._extract_codec_embedding_side_info(emb_groups),
            },
            "estimated_num_bits": None,
        }

    @torch.no_grad()
    def decompress(
        self,
        strings: Mapping[str, Any],
        metadata: Mapping[str, Any],
        sample_configs: Optional[Mapping[int, Any]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        out_zoom: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Decode entropy strings into reconstructed FieldSpace tensors."""

        if "y" not in strings or "z" not in strings:
            raise ValueError("decompress requires strings with 'y' and 'z' entries.")
        sample_configs = sample_configs or metadata.get("sample_configs", {})
        if emb_groups is None:
            emb_groups = self._rebuild_codec_embedding_groups(metadata.get("embedding_side_info"), self._device())
        z_hat = self.entropy_bottleneck_adapter.decompress(
            strings["z"],
            metadata["z"],
            metadata["z_specs"],
            device=self._device(),
        )
        scales, means = self.hyper_decode(z_hat, sample_configs, mask_zooms_groups, emb_groups)
        y_hat = self.gaussian_conditional_adapter.decompress(
            strings["y"],
            scales,
            means,
            metadata["y_specs"],
            device=self._device(),
        )
        x_hat = self.ae_decode(y_hat, sample_configs, mask_zooms_groups, emb_groups, out_zoom=out_zoom)
        return {"x_hat": x_hat, "y_hat": y_hat}

    def aux_loss(self) -> torch.Tensor:
        """Return entropy bottleneck auxiliary loss."""

        loss = self.entropy_bottleneck_adapter.aux_loss()
        return loss.to(device=self._device())

    def update(self, force: bool = False) -> bool:
        """Update entropy model CDF tables for compression."""

        eb_updated = self.entropy_bottleneck_adapter.update(force=force)
        gc_updated = self.gaussian_conditional_adapter.update(force=force)
        return bool(eb_updated and gc_updated)

    def _device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _extract_codec_embedding_side_info(
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]],
    ) -> Optional[List[Optional[Dict[str, torch.Tensor]]]]:
        """Store deterministic embedding inputs needed for standalone decoding."""

        if emb_groups is None:
            return None
        side_info: List[Optional[Dict[str, torch.Tensor]]] = []
        has_any = False
        for emb_group in emb_groups:
            if not emb_group:
                side_info.append(None)
                continue
            group_info: Dict[str, torch.Tensor] = {}
            for key in ("VariableEmbedder", "MGEmbedder"):
                value = emb_group.get(key)
                if torch.is_tensor(value):
                    group_info[key] = value.detach().cpu()
            if "MGEmbedder" not in group_info and "VariableEmbedder" in group_info:
                group_info["MGEmbedder"] = group_info["VariableEmbedder"]
            if group_info:
                has_any = True
                side_info.append(group_info)
            else:
                side_info.append(None)
        return side_info if has_any else None

    @staticmethod
    def _rebuild_codec_embedding_groups(
        side_info: Optional[Sequence[Optional[Mapping[str, torch.Tensor]]]],
        device: torch.device,
    ) -> Optional[List[Optional[Dict[str, torch.Tensor]]]]:
        """Rebuild ``emb_groups`` from codec metadata."""

        if side_info is None:
            return None
        emb_groups: List[Optional[Dict[str, torch.Tensor]]] = []
        for group_info in side_info:
            if not group_info:
                emb_groups.append(None)
                continue
            emb_group = {
                key: value.to(device=device) if torch.is_tensor(value) else value
                for key, value in group_info.items()
            }
            if "MGEmbedder" not in emb_group and "VariableEmbedder" in emb_group:
                emb_group["MGEmbedder"] = emb_group["VariableEmbedder"]
            emb_groups.append(emb_group)
        return emb_groups
