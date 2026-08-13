import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ...modules.field_space.field_space_base import DiffDecoder, GLOBAL_EMBEDDER_CACHE_KEY
from ...modules.embedding.embedder import get_embedder
from .block_wrap_operations import (
    BlockWrapContext,
    BlockWrapConfig,
    BlockWrapOperation,
    create_block_wrap_operation,
    validate_block_wrap_operation_sequence,
)
from .mg_base_model import MG_base_model, create_encoder_decoder_block, create_missing_zooms


class BlockExecutionStage(nn.Module):
    def __init__(
        self,
        *,
        wrap_operations: Optional[Mapping[str, BlockWrapOperation]] = None,
        blocks: Optional[Mapping[str, nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.wrap_operations = nn.ModuleDict(wrap_operations or {})
        self.blocks = nn.ModuleDict(blocks or {})
        if self.wrap_operations:
            validate_block_wrap_operation_sequence(self.wrap_operations)

    def forward(
        self,
        x_zooms_groups: Sequence[Dict[int, torch.Tensor]],
        context: BlockWrapContext,
    ) -> List[Dict[int, torch.Tensor]]:
        wrap_states: List[tuple[BlockWrapOperation, Any]] = []

        for operation in self.wrap_operations.values():
            x_zooms_groups, state = operation.pre(list(x_zooms_groups), context)
            wrap_states.append((operation, state))

        for block in self.blocks.values():
            x_zooms_groups = block(
                x_zooms_groups,
                sample_configs=context.sample_configs,
                mask_groups=context.mask_groups,
                emb_groups=context.emb_groups,
            )

        for operation, state in reversed(wrap_states):
            x_zooms_groups = operation.post(x_zooms_groups, state, context)

        return list(x_zooms_groups)


class MG_Transformer(MG_base_model):
    """
    Multi-grid transformer composed of configurable encoder/decoder blocks.
    """

    def __init__(
        self,
        mgrids: Sequence[Mapping[str, Any]],
        block_configs: Optional[Mapping[str, Any]] = None,
        in_zooms: Sequence[int] = (),
        block_wrap_configs: Optional[Mapping[str, Any]] = None,
        in_features: int = 1,
        n_groups_variables: Sequence[int] = [1],
        n_groups_depths: Optional[Sequence[int]] = None,
        shared_indexed_group_variables: Optional[Sequence[bool]] = None,
        shared_indexed_group_depths: Optional[Sequence[bool]] = None,
        shared_indexed_group_space: Optional[Sequence[bool]] = None,
        use_global_embedder: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the multi-grid transformer and its block stack.

        :param mgrids: Multi-grid configuration used by the base model.
        :param block_configs: Mapping of block configurations.
        :param in_zooms: Input zoom levels used by the model.
        :param block_wrap_configs: Optional mapping of wrap stages or global wrap operations.
        :param in_features: Number of input features per variable.
        :param n_groups_variables: Number of variable groups for attention layers.
        :param kwargs: Additional arguments forwarded to block factories.
        :return: None.
        """
        super().__init__(mgrids)

        self.in_zooms: Sequence[int] = in_zooms
        self.in_features: int = in_features
        self.n_groups_variables: Sequence[int] = list(n_groups_variables)
        self.n_groups_depths: Sequence[int] = (
            list(n_groups_depths) if n_groups_depths is not None else [1] * len(n_groups_variables)
        )
        self.shared_indexed_group_variables: Sequence[bool] = (
            list(shared_indexed_group_variables)
            if shared_indexed_group_variables is not None
            else [False] * len(n_groups_variables)
        )
        self.shared_indexed_group_depths: Sequence[bool] = (
            list(shared_indexed_group_depths)
            if shared_indexed_group_depths is not None
            else [False] * len(n_groups_variables)
        )
        self.shared_indexed_group_space: Sequence[bool] = (
            list(shared_indexed_group_space)
            if shared_indexed_group_space is not None
            else [False] * len(n_groups_variables)
        )
        self.use_global_embedder: bool = use_global_embedder
        self.block_build_kwargs: Dict[str, Any] = dict(kwargs)
        self.global_embedders: nn.ModuleDict = (
            self._build_global_embedders(block_configs, block_wrap_configs)
            if self.use_global_embedder
            else nn.ModuleDict()
        )
        if self.global_embedders:
            self.block_build_kwargs["global_embedders"] = self.global_embedders

        self.block_stages: nn.ModuleDict = nn.ModuleDict()
        self.block_wrap_operations: nn.ModuleDict = nn.ModuleDict()
        self.Blocks: nn.ModuleDict = nn.ModuleDict()

        current_in_zooms = list(in_zooms)
        current_in_features = [in_features] * len(in_zooms)
        block_configs = {} if block_configs is None else block_configs

        has_stage_local_blocks = (
            block_wrap_configs is not None
            and any(bool(getattr(wrap_conf, "block_configs", None)) for wrap_conf in block_wrap_configs.values())
        )

        if has_stage_local_blocks:
            for wrap_key, wrap_conf in block_wrap_configs.items():
                assert isinstance(wrap_key, str), "block wrap keys should be strings"
                stage_block_configs = getattr(wrap_conf, "block_configs", None)
                if not stage_block_configs:
                    raise ValueError(
                        "When any block_wrap_configs entry defines block_configs, every block_wrap_configs entry "
                        "must define block_configs."
                    )

                stage_build_kwargs = dict(self.block_build_kwargs)
                stage_wrap_operations = nn.ModuleDict()
                stage_in_zooms = list(current_in_zooms)
                stage_in_features = list(current_in_features)
                if isinstance(wrap_conf, BlockWrapConfig):
                    stage_build_kwargs.update(
                        wrap_conf.get_block_build_overrides(
                            n_groups_variables=self.n_groups_variables,
                            n_groups_depths=self.n_groups_depths,
                            base_block_kwargs=stage_build_kwargs,
                        )
                    )
                    stage_in_zooms = wrap_conf.get_stage_input_zooms(current_in_zooms)
                    stage_in_features = wrap_conf.get_stage_input_features(
                        current_in_zooms=current_in_zooms,
                        current_in_features=current_in_features,
                    )
                stage_wrap_operations[wrap_key] = create_block_wrap_operation(
                    wrap_conf,
                    grid_layers=self.grid_layers,
                )

                stage_blocks, current_in_zooms, current_in_features = self._build_blocks(
                    block_configs=stage_block_configs,
                    in_zooms=stage_in_zooms,
                    in_features=stage_in_features,
                    block_build_kwargs=stage_build_kwargs,
                )
                self.block_stages[wrap_key] = BlockExecutionStage(
                    wrap_operations=stage_wrap_operations,
                    blocks=stage_blocks,
                )

            if block_configs:
                trailing_blocks, current_in_zooms, current_in_features = self._build_blocks(
                    block_configs=block_configs,
                    in_zooms=current_in_zooms,
                    in_features=current_in_features,
                    block_build_kwargs=dict(self.block_build_kwargs),
                )
                self.block_stages["unwrapped_blocks"] = BlockExecutionStage(blocks=trailing_blocks)

            if self.block_stages:
                final_stage = list(self.block_stages.values())[-1]
                if final_stage.blocks:
                    list(final_stage.blocks.values())[-1].out_features = [current_in_features[0]]

        else:
            if block_wrap_configs:
                for wrap_key, wrap_conf in block_wrap_configs.items():
                    assert isinstance(wrap_key, str), "block wrap keys should be strings"
                    if isinstance(wrap_conf, BlockWrapConfig):
                        self.block_build_kwargs.update(
                            wrap_conf.get_block_build_overrides(
                                n_groups_variables=self.n_groups_variables,
                                n_groups_depths=self.n_groups_depths,
                                base_block_kwargs=self.block_build_kwargs,
                            )
                        )
                    self.block_wrap_operations[wrap_key] = create_block_wrap_operation(
                        wrap_conf,
                        grid_layers=self.grid_layers,
                    )
                validate_block_wrap_operation_sequence(self.block_wrap_operations)

            self.Blocks, current_in_zooms, current_in_features = self._build_blocks(
                block_configs=block_configs,
                in_zooms=current_in_zooms,
                in_features=current_in_features,
                block_build_kwargs=dict(self.block_build_kwargs),
            )

            if self.Blocks:
                list(self.Blocks.values())[-1].out_features = [current_in_features[0]]

        self.decoder: DiffDecoder = DiffDecoder()

    def _iter_embedding_block_configs(
        self,
        block_configs: Optional[Mapping[str, Any]],
        block_wrap_configs: Optional[Mapping[str, Any]],
    ) -> Iterable[Any]:
        if block_configs is not None:
            yield from block_configs.values()

        if block_wrap_configs is None:
            return

        for wrap_conf in block_wrap_configs.values():
            stage_block_configs = getattr(wrap_conf, "block_configs", None)
            if stage_block_configs:
                yield from stage_block_configs.values()

    def _resolve_embed_input_zoom(self, block_conf: Any) -> int:
        embed_confs = getattr(block_conf, "embed_confs", {}) or {}
        if "input_zoom" in embed_confs:
            return int(embed_confs["input_zoom"])

        q_zooms = getattr(block_conf, "q_zooms", None)
        if q_zooms is not None:
            if isinstance(q_zooms, int):
                return (
                    int(min(self.in_zooms))
                    if q_zooms == -1
                    else int(q_zooms)
                )
            return int(min(q_zooms))

        block_in_zooms = getattr(block_conf, "in_zooms", None)
        if block_in_zooms:
            return int(min(block_in_zooms))
        return int(min(self.in_zooms))

    def _build_global_embedders(
        self,
        block_configs: Optional[Mapping[str, Any]],
        block_wrap_configs: Optional[Mapping[str, Any]],
    ) -> nn.ModuleDict:
        global_embedders = nn.ModuleDict()
        global_embed_confs_by_zoom: Dict[str, Dict[str, Any]] = {}

        for block_conf in self._iter_embedding_block_configs(
            block_configs,
            block_wrap_configs,
        ):
            embed_confs = getattr(block_conf, "embed_confs", None)
            if not embed_confs or not embed_confs.get("embed_names"):
                continue

            input_zoom = self._resolve_embed_input_zoom(block_conf)
            zoom_key = str(input_zoom)

            if zoom_key not in global_embed_confs_by_zoom:
                merged_embed_confs = copy.deepcopy(embed_confs)
                merged_embed_confs["embed_names"] = self._normalize_embed_names(embed_confs["embed_names"])
                merged_embed_confs["embed_mode"] = "sum"
                global_embed_confs_by_zoom[zoom_key] = merged_embed_confs
                continue

            merged_embed_confs = global_embed_confs_by_zoom[zoom_key]
            merged_embed_confs["embed_names"] = self._merge_embed_name_lists(
                merged_embed_confs["embed_names"],
                embed_confs["embed_names"],
            )
            for embed_name, embed_conf in embed_confs.get("embed_confs", {}).items():
                if embed_name not in merged_embed_confs["embed_confs"]:
                    merged_embed_confs["embed_confs"][embed_name] = copy.deepcopy(embed_conf)

        for zoom_key, embed_confs in global_embed_confs_by_zoom.items():
            global_embedders[zoom_key] = get_embedder(
                **embed_confs,
                grid_layers=self.grid_layers,
                zoom=int(zoom_key),
            )

        return global_embedders

    @staticmethod
    def _normalize_embed_names(embed_names: Sequence[Any]) -> List[str]:
        if not embed_names:
            return []

        first_entry = embed_names[0]
        if isinstance(first_entry, Sequence) and not isinstance(first_entry, str):
            normalized: List[str] = []
            for group in embed_names:
                for embed_name in group:
                    if embed_name not in normalized:
                        normalized.append(str(embed_name))
            return normalized

        return [str(embed_name) for embed_name in embed_names]

    def _merge_embed_name_lists(
        self,
        base_embed_names: Sequence[Any],
        new_embed_names: Sequence[Any],
    ) -> List[str]:
        merged = self._normalize_embed_names(base_embed_names)
        for embed_name in self._normalize_embed_names(new_embed_names):
            if embed_name not in merged:
                merged.append(embed_name)
        return merged

    def _build_blocks(
        self,
        *,
        block_configs: Mapping[str, Any],
        in_zooms: Sequence[int],
        in_features: Sequence[int],
        block_build_kwargs: Mapping[str, Any],
    ) -> tuple[nn.ModuleDict, List[int], List[int]]:
        blocks = nn.ModuleDict()
        current_in_zooms = list(in_zooms)
        current_in_features = list(in_features)

        for block_key, block_conf in block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            current_block_build_kwargs = dict(block_build_kwargs)
            block_n_groups_variables = list(current_block_build_kwargs.pop("n_groups_variables", self.n_groups_variables))
            block_n_groups_depths = list(current_block_build_kwargs.pop("n_groups_depths", self.n_groups_depths))
            block_shared_indexed_group_variables = list(
                current_block_build_kwargs.pop("shared_indexed_group_variables", self.shared_indexed_group_variables)
            )
            block_shared_indexed_group_depths = list(
                current_block_build_kwargs.pop("shared_indexed_group_depths", self.shared_indexed_group_depths)
            )
            block_shared_indexed_group_space = list(
                current_block_build_kwargs.pop("shared_indexed_group_space", self.shared_indexed_group_space)
            )
            block = create_encoder_decoder_block(
                block_conf,
                current_in_zooms,
                current_in_features,
                block_n_groups_variables,
                self.grid_layers,
                block_n_groups_depths,
                block_shared_indexed_group_variables,
                block_shared_indexed_group_depths,
                block_shared_indexed_group_space,
                **current_block_build_kwargs,
            )

            blocks[block_key] = block
            current_in_features = list(block.out_features)
            current_in_zooms = list(block.out_zooms)

        return blocks, current_in_zooms, current_in_features

    def _prime_global_embedding_cache(
        self,
        emb_groups: Optional[Sequence[Optional[Dict[str, Any]]]],
        sample_configs: Mapping[int, Any],
    ) -> None:
        if emb_groups is None or not self.global_embedders:
            return

        for emb_group in emb_groups:
            if emb_group is None:
                continue

            emb_group[GLOBAL_EMBEDDER_CACHE_KEY] = {}
            for zoom_key, embedder in self.global_embedders.items():
                zoom = int(zoom_key)
                emb_group[GLOBAL_EMBEDDER_CACHE_KEY][zoom_key] = embedder(
                    emb_group,
                    sample_configs=sample_configs,
                    output_zoom=zoom,
                )

    def decode(
        self,
        x_zooms: Dict[int, torch.Tensor],
        sample_configs: Mapping[int, Any],
        out_zoom: Optional[int] = None,
        emb: Optional[Mapping[str, Any]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Decode a single zoom-mapping into a requested zoom.

        :param x_zooms: Mapping from zoom level to tensor of shape ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :param emb: Optional embedding dictionary for decoding.
        :return: Decoded zoom mapping.
        """
        emb = emb or {}
        return self.decoder(x_zooms, emb=emb, sample_configs=sample_configs, out_zoom=out_zoom)

    def forward(
        self,
        x_zooms_groups: Optional[Sequence[Dict[int, torch.Tensor]]] = None,
        mask_zooms_groups: Optional[Sequence[Optional[Dict[int, torch.Tensor]]]] = None,
        emb_groups: Optional[Sequence[Dict[str, Any]]] = None,
        sample_configs: Mapping[int, Any] = {},
        out_zoom: Optional[int] = None,
    ) -> Sequence[Dict[int, torch.Tensor]]:
        """
        Forward pass through the multi-grid transformer.

        :param x_zooms_groups: List of per-group zoom mappings with tensors of shape
            ``(b, v, t, n, d, f)``.
        :param mask_zooms_groups: Optional list of mask mappings aligned with inputs, each
            matching the data tensor shape ``(b, v, t, n, d, f)``.
        :param emb_groups: Optional list of embedding dictionaries aligned with inputs.
        :param sample_configs: Sampling configuration dictionary per zoom.
        :param out_zoom: Optional target zoom level to decode outputs into.
        :return: Output zoom-group mappings.
        """

        x_zooms_groups, mask_zooms_groups, emb_groups, sample_configs = create_missing_zooms(
            x_zooms_groups, self.in_zooms, mask_zooms_groups, emb_groups, sample_configs=sample_configs)
        self._prime_global_embedding_cache(emb_groups, sample_configs)

        context = BlockWrapContext(
            mask_groups=mask_zooms_groups,
            emb_groups=emb_groups,
            sample_configs=sample_configs,
        )

        x_zooms_groups = self._run_execution_stages(x_zooms_groups, context)

        if out_zoom is not None:
            for i, x_zooms in enumerate(x_zooms_groups):
                x_zooms_groups[i] = (
                    self.decoder(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)
                    if x_zooms
                    else {}
                )

        return x_zooms_groups

    def _run_execution_stages(
        self,
        x_zooms_groups: Sequence[Dict[int, torch.Tensor]],
        context: BlockWrapContext,
    ) -> List[Dict[int, torch.Tensor]]:
        if getattr(self, "block_stages", None):
            for stage in self.block_stages.values():
                x_zooms_groups = stage(x_zooms_groups, context)
            return list(x_zooms_groups)

        return self._run_block_sequence_with_wraps(x_zooms_groups, context)

    def _run_block_sequence_with_wraps(
        self,
        x_zooms_groups: Sequence[Dict[int, torch.Tensor]],
        context: BlockWrapContext,
    ) -> List[Dict[int, torch.Tensor]]:
        wrap_states: List[tuple[BlockWrapOperation, Any]] = []

        for operation in self.block_wrap_operations.values():
            x_zooms_groups, state = operation.pre(list(x_zooms_groups), context)
            wrap_states.append((operation, state))

        for block in self.Blocks.values():
            x_zooms_groups = block(
                x_zooms_groups,
                sample_configs=context.sample_configs,
                mask_groups=context.mask_groups,
                emb_groups=context.emb_groups,
            )

        for operation, state in reversed(wrap_states):
            x_zooms_groups = operation.post(x_zooms_groups, state, context)

        return list(x_zooms_groups)
