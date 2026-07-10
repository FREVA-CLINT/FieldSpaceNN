from typing import Any, Dict, List, Optional, Tuple, Union
from einops import rearrange
import copy
from omegaconf import ListConfig

import torch
import torch.nn as nn

from ..base import get_layer, MLP_fac
from ..factorization import broadcast_indexed_tensor, build_indexed_dims
from .field_space_base import (
    Tokenizer,
    LinEmbLayer,
    add_time_overlap_from_neighbor_patches,
    add_depth_overlap_from_neighbor_patches,
)

from ..grids.grid_layer import GridLayer
from ..transformer.transformer_base import safe_scaled_dot_product_attention

from ..embedding.embedder import get_embedder

from ..grids.grid_utils import insert_matching_time_patch

from ...utils.helpers import check_value


class FieldSpaceAttentionConfig:
    def __init__(
        self,
        token_zoom: int,
        groups: Union[List[bool], int] = -1,
        q_zooms: Union[List[int], int] = -1,
        kv_zooms: Union[List[int], int] = -1,
        att_dim: int = 64,
        target_zooms: Optional[List[int]] = None,
        token_len_depth: Union[List[int], int] = [1],
        token_len_time: Union[List[int], int] = 1,
        token_overlap_space: Union[List[int], int, bool] = False,
        token_overlap_time: Union[List[int], int, bool] = False,
        token_overlap_depth: Union[List[int], int, bool] = False,
        token_overlap_mlp_time: Union[List[bool], bool] = False,
        token_overlap_mlp_depth: Union[List[bool], bool] = False,
        rank_variables: Union[List[int], int, None] = None,
        rank_space: Union[List[int], int, None] = None,
        n_rank_space: Union[List[int], int, None] = None,
        rank_time: Union[List[int], int, None] = None,
        rank_depth: Union[List[int], int, None] = None,
        rank_features: Union[List[int], int, None] = None,
        n_times: Union[List[int], int] = 1,
        n_depths: Union[List[int], int, None] = None,
        seq_len_zoom: int = -1,
        seq_len_time: Union[List[int], int] = -1,
        seq_len_depth: Union[List[int], int] = -1,
        seq_overlap_space: bool = False,
        seq_overlap_time: bool = False,
        seq_overlap_depth: bool = False,
        with_var_att: bool = False,
        update: str = 'shift',
        separate_mlp_norm: bool = True,
        mlp_residual_from_attention: bool = False,
        use_variable_emb_layer: bool = True,
        use_variable_layer_norm: bool = True,
        use_variable_qkv: bool = True,
        use_variable_mlp: bool = True,
        use_indexed_emb_layer: Optional[bool] = None,
        use_indexed_layer_norm: Optional[bool] = None,
        use_indexed_qkv: Optional[bool] = None,
        use_indexed_mlp: Optional[bool] = None,
        use_ranks_emb_layer: bool = True,
        use_ranks_qkv: bool = True,
        use_ranks_mlp: bool = True,
        use_variable_att_gammas: bool = False,
        use_variable_mlp_gammas: bool = False,
        use_indexed_att_gammas: Optional[bool] = None,
        use_indexed_mlp_gammas: Optional[bool] = None,
        **kwargs: Any
    ) -> None:
        """
        Store configuration for field-space attention.

        :param token_zoom: Token zoom level.
        :param groups: ``-1`` to instantiate attention for all groups, or a bool list
            indicating which groups get a FieldSpaceAttention block.
        :param q_zooms: Query zoom levels or -1 to default to input zooms.
        :param kv_zooms: Key/value zoom levels or -1 to default to input zooms.
        :param att_dim: Attention feature dimension.
        :param target_zooms: Optional target zooms for updates.
        :param token_len_depth: Token length along depth.
        :param token_len_time: Token length along time.
        :param token_overlap_space: Token overlap along space.
        :param token_overlap_time: Token overlap along time.
        :param token_overlap_depth: Token overlap along depth.
        :param token_overlap_mlp_time: MLP overlap along time.
        :param token_overlap_mlp_depth: MLP overlap along depth.
        :param rank_space: Optional rank for space.
        :param rank_time: Optional rank for time.
        :param rank_depth: Optional rank for depth.
        :param rank_features: Optional rank for features.
        :param rank_variables: Optional rank for features.
        :param seq_len_zoom: Sequence zoom for attention.
        :param seq_len_time: Sequence length along time.
        :param seq_len_depth: Sequence length along depth.
        :param seq_overlap_space: Overlap along space.
        :param seq_overlap_time: Overlap along time.
        :param seq_overlap_depth: Overlap along depth.
        :param with_var_att: Whether to include variable attention.
        :param update: Update mode ("shift" or "shift_scale").
        :param separate_mlp_norm: Whether to separate MLP norm.
        :param mlp_residual_from_attention: Whether the MLP residual uses the
            post-attention tensor instead of the original zoom tensor.
        :param use_variable_emb_layer: Whether embedding layers use variable-specific parameters.
        :param use_variable_layer_norm: Whether embedding-layer layer norms use variable-specific affine params.
        :param use_variable_qkv: Whether Q/KV/attention projection layers use variable-specific parameters.
        :param use_variable_mlp: Whether the MLP branch uses variable-specific parameters.
        :param use_ranks_emb_layer: Whether embedding layers use the configured ranks.
        :param use_ranks_qkv: Whether Q/KV/attention projection layers use the configured ranks.
        :param use_ranks_mlp: Whether the MLP branch uses the configured ranks.
        :param use_variable_att_gammas: Whether attention residual gammas are variable-specific.
        :param use_variable_mlp_gammas: Whether MLP residual gammas are variable-specific.
        :param kwargs: Additional keyword arguments assigned as attributes.
        :return: None.
        """
        self.token_zoom: int
        self.groups: Union[List[bool], int]
        self.q_zooms: Union[List[int], int]
        self.kv_zooms: Union[List[int], int]
        self.att_dim: int
        self.target_zooms: Optional[List[int]]
        self.token_len_depth: Union[List[int], int]
        self.token_len_time: Union[List[int], int]
        self.token_overlap_space: Union[List[int], int, bool]
        self.token_overlap_time: Union[List[int], int, bool]
        self.token_overlap_depth: Union[List[int], int, bool]
        self.token_overlap_mlp_time: Union[List[bool], bool]
        self.token_overlap_mlp_depth: Union[List[bool], bool]
        self.rank_space: Union[List[int], int, None]
        self.n_rank_space: Union[List[int], int, None]
        self.rank_time: Union[List[int], int, None]
        self.rank_depth: Union[List[int], int, None]
        self.rank_features: Union[List[int], int, None]
        self.rank_variables: Union[List[int], int, None]
        self.n_times: Union[List[int], int]
        self.n_depths: Union[List[int], int]
        self.seq_len_zoom: int
        self.seq_len_time: Union[List[int], int]
        self.seq_len_depth: Union[List[int], int]
        self.seq_overlap_space: bool
        self.seq_overlap_time: bool
        self.seq_overlap_depth: bool
        self.with_var_att: bool
        self.update: str
        self.separate_mlp_norm: bool
        self.mlp_residual_from_attention: bool
        self.use_variable_emb_layer: bool
        self.use_variable_layer_norm: bool
        self.use_variable_qkv: bool
        self.use_variable_mlp: bool
        self.use_indexed_emb_layer: bool
        self.use_indexed_layer_norm: bool
        self.use_indexed_qkv: bool
        self.use_indexed_mlp: bool
        self.use_ranks_emb_layer: bool
        self.use_ranks_qkv: bool
        self.use_ranks_mlp: bool
        self.use_variable_att_gammas: bool
        self.use_variable_mlp_gammas: bool
        self.use_indexed_att_gammas: bool
        self.use_indexed_mlp_gammas: bool

        def _resolve_alias(indexed_value: Optional[bool], legacy_value: bool) -> tuple[bool, bool]:
            resolved = legacy_value if indexed_value is None else indexed_value
            return resolved, resolved

        n_depths_is_default = n_depths is None
        n_depths = 1 if n_depths is None else n_depths

        use_indexed_emb_layer, use_variable_emb_layer = _resolve_alias(use_indexed_emb_layer, use_variable_emb_layer)
        use_indexed_layer_norm, use_variable_layer_norm = _resolve_alias(use_indexed_layer_norm, use_variable_layer_norm)
        use_indexed_qkv, use_variable_qkv = _resolve_alias(use_indexed_qkv, use_variable_qkv)
        use_indexed_mlp, use_variable_mlp = _resolve_alias(use_indexed_mlp, use_variable_mlp)
        use_indexed_att_gammas, use_variable_att_gammas = _resolve_alias(use_indexed_att_gammas, use_variable_att_gammas)
        use_indexed_mlp_gammas, use_variable_mlp_gammas = _resolve_alias(use_indexed_mlp_gammas, use_variable_mlp_gammas)

        inputs = copy.deepcopy(locals())

        for input, value in inputs.items():
            if input in {'self', '_resolve_alias'}:
                continue
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)


class FieldSpaceAttentionModule(nn.Module):
  
    def __init__(
        self,
        grid_layers: Dict[str, GridLayer],
        in_zooms: List[int],
        out_zooms: List[int],
        q_zooms: Union[List[int], int],
        kv_zooms: Union[List[int], int],
        token_zoom: int,
        groups: Union[List[bool], int] = -1,
        target_zooms: Optional[List[int]] = None,
        in_features: int = 1,
        n_groups_variables: List[int] = [1],
        n_groups_depths: Optional[List[int]] = None,
        shared_indexed_group_variables: Union[List[bool], bool] = False,
        shared_indexed_group_depths: Union[List[bool], bool] = False,
        shared_indexed_group_space: Union[List[bool], bool] = False,
        token_len_depth: Union[List[int], int] = 1,
        token_len_time: Union[List[int], int] = 1,
        token_overlap_space: Union[List[bool], bool] = False,
        token_overlap_time: Union[List[bool], bool] = False,
        token_overlap_depth: Union[List[bool], bool] = False,
        token_overlap_mlp_time: Union[List[bool], bool] = False,
        token_overlap_mlp_depth: Union[List[bool], bool] = False,
        rank_variables: Union[List[int], int, None] = None,
        rank_space: Union[List[int], int, None] = None,
        n_rank_space: Union[List[int], int, None] = None,
        rank_time: Union[List[int], int, None] = None,
        rank_depth: Union[List[int], int, None] = None,
        rank_features: Union[List[int], int, None] = None,
        n_times: Union[List[int], int] = 1,
        n_depths: Union[List[int], int, None] = None,
        seq_len_zoom: int = -1,
        seq_len_time: Union[List[int], int] = -1,
        seq_len_depth: Union[List[int], int] = -1,
        seq_overlap_space: bool = False,
        seq_overlap_time: bool = False,
        seq_overlap_depth: bool = False,
        with_var_att: bool = False,
        use_mask: bool = False,
        att_dim: Optional[int] = None,
        att_dim_mixed: Optional[int] = 0,
        n_head_channels: int = 16,
        dropout: float = 0,
        update: str = 'shift',
        separate_mlp_norm: bool = True,
        mlp_residual_from_attention: bool = False,
        use_variable_emb_layer: bool = True,
        use_variable_layer_norm: bool = True,
        use_variable_qkv: bool = True,
        use_variable_mlp: bool = True,
        use_indexed_emb_layer: Optional[bool] = None,
        use_indexed_layer_norm: Optional[bool] = None,
        use_indexed_qkv: Optional[bool] = None,
        use_indexed_mlp: Optional[bool] = None,
        use_ranks_emb_layer: bool = True,
        use_ranks_qkv: bool = True,
        use_ranks_mlp: bool = True,
        use_variable_att_gammas: bool = False,
        use_variable_mlp_gammas: bool = False,
        use_indexed_att_gammas: Optional[bool] = None,
        use_indexed_mlp_gammas: Optional[bool] = None,
        embed_confs: Dict[str, Any] = {},
        global_embedders: Optional[nn.ModuleDict] = None,
        fac_mode: str = "Tucker",
        emb_aggregation: str = "shift_scale",
    ) -> None:
        """
        Initialize the field-space attention module.

        :param grid_layers: Mapping from zoom string to GridLayer.
        :param in_zooms: Input zoom levels.
        :param out_zooms: Output zoom levels.
        :param q_zooms: Query zoom levels or -1 to default to input zooms.
        :param kv_zooms: Key/value zoom levels or -1 to default to input zooms.
        :param token_zoom: Token zoom level.
        :param groups: ``-1`` to instantiate attention for all groups, or a bool list
            indicating which groups get a FieldSpaceAttention block.
        :param target_zooms: Optional target zooms for updates.
        :param in_features: Number of input features.
        :param n_groups_variables: Number of variable groups.
        :param token_len_depth: Token length along depth.
        :param token_len_time: Token length along time.
        :param token_overlap_space: Token overlap along space.
        :param token_overlap_time: Token overlap along time.
        :param token_overlap_depth: Token overlap along depth.
        :param token_overlap_mlp_time: MLP overlap along time.
        :param token_overlap_mlp_depth: MLP overlap along depth.
        :param rank_space: Optional rank for space.
        :param rank_time: Optional rank for time.
        :param rank_depth: Optional rank for depth.
        :param rank_features: Optional rank for features.
        :param rank_variables: Optional rank for variables.
        :param seq_len_zoom: Sequence zoom for attention.
        :param seq_len_time: Sequence length along time.
        :param seq_len_depth: Sequence length along depth.
        :param seq_overlap_space: Overlap along space.
        :param seq_overlap_time: Overlap along time.
        :param seq_overlap_depth: Overlap along depth.
        :param with_var_att: Whether to include variable attention.
        :param use_mask: Whether to apply attention masks.
        :param att_dim: Attention feature dimension.
        :param n_head_channels: Head channel size.
        :param dropout: Dropout rate.
        :param update: Update mode ("shift" or "shift_scale").
        :param separate_mlp_norm: Whether to separate MLP norm.
        :param mlp_residual_from_attention: Whether the MLP residual uses the
            post-attention tensor instead of the original zoom tensor.
        :param use_variable_emb_layer: Whether embedding layers use variable-specific parameters.
        :param use_variable_layer_norm: Whether embedding-layer layer norms use variable-specific affine params.
        :param use_variable_qkv: Whether Q/KV/attention projection layers use variable-specific parameters.
        :param use_variable_mlp: Whether the MLP branch uses variable-specific parameters.
        :param use_ranks_emb_layer: Whether embedding layers use the configured ranks.
        :param use_ranks_qkv: Whether Q/KV/attention projection layers use the configured ranks.
        :param use_ranks_mlp: Whether the MLP branch uses the configured ranks.
        :param use_variable_att_gammas: Whether attention residual gammas are variable-specific.
        :param use_variable_mlp_gammas: Whether MLP residual gammas are variable-specific.
        :param embed_confs: Embedding configuration dictionary.
        :param layer_confs: Layer configuration for attention blocks.
        :param layer_confs_emb: Layer configuration for embedding blocks.
        :return: None.
        """
        super().__init__()
        
        # Normalize per-group configs so indexing is consistent across variable groups.
        n_groups = len(n_groups_variables)
        if isinstance(groups, (list, tuple, ListConfig)):
            groups = list(groups)
            if len(groups) != n_groups:
                raise ValueError(
                    f"groups must have length {n_groups}, got {len(groups)}"
                )
            active_groups = [bool(group) for group in groups]
        elif groups == -1:
            active_groups = [True] * n_groups
        else:
            raise ValueError("groups must be -1 or a list of bools")

        if n_groups_depths is None:
            n_groups_depths = [1] * n_groups
        if len(n_groups_depths) != n_groups:
            raise ValueError(f"n_groups_depths must have length {n_groups}, got {len(n_groups_depths)}")
        shared_indexed_group_variables = check_value(shared_indexed_group_variables, n_groups)
        shared_indexed_group_depths = check_value(shared_indexed_group_depths, n_groups)
        shared_indexed_group_space = check_value(shared_indexed_group_space, n_groups)

        def _resolve_alias(indexed_value: Optional[bool], legacy_value: bool) -> bool:
            return legacy_value if indexed_value is None else indexed_value

        token_len_depth = check_value(token_len_depth, n_groups)
        token_len_time = check_value(token_len_time, n_groups)

        token_overlap_space = check_value(token_overlap_space, n_groups)
        token_overlap_time = check_value(token_overlap_time, n_groups)
        token_overlap_depth = check_value(token_overlap_depth, n_groups)
        token_overlap_mlp_time = check_value(token_overlap_mlp_time, n_groups)
        token_overlap_mlp_depth = check_value(token_overlap_mlp_depth, n_groups)

        rank_space = check_value(rank_space, n_groups)
        n_rank_space = check_value(n_rank_space, n_groups)
        rank_time = check_value(rank_time, n_groups)
        rank_depth = check_value(rank_depth, n_groups)
        rank_variables = check_value(rank_variables, n_groups)
        rank_features = check_value(rank_features, n_groups)
        n_times = check_value(n_times, n_groups)
        if n_depths is None:
            n_depths = list(n_groups_depths)
        else:
            n_depths = check_value(n_depths, n_groups)

        use_indexed_emb_layer = _resolve_alias(use_indexed_emb_layer, use_variable_emb_layer)
        use_indexed_layer_norm = _resolve_alias(use_indexed_layer_norm, use_variable_layer_norm)
        use_indexed_qkv = _resolve_alias(use_indexed_qkv, use_variable_qkv)
        use_indexed_mlp = _resolve_alias(use_indexed_mlp, use_variable_mlp)
        use_indexed_att_gammas = _resolve_alias(use_indexed_att_gammas, use_variable_att_gammas)
        use_indexed_mlp_gammas = _resolve_alias(use_indexed_mlp_gammas, use_variable_mlp_gammas)
        

        self.out_zooms: List[int] = copy.deepcopy(out_zooms)
        in_zooms = copy.deepcopy(in_zooms)
        self.use_mask: bool = use_mask

        # Default q/kv zooms to input zooms when not explicitly configured.
        if not isinstance(q_zooms, (List,ListConfig)) and (q_zooms == -1):
            q_zooms = in_zooms
        
        if not isinstance(kv_zooms,(List,ListConfig)) and (kv_zooms == -1):
            kv_zooms = in_zooms

        for k, zoom in enumerate(kv_zooms):
            if zoom not in in_zooms:
                raise ValueError(f"Zoom level {zoom} at index {k} of kv_zooms not found in in_zooms")
        
        # Compute unique set of zooms participating in attention.
        self.qkv_zooms: List[int] = torch.tensor(q_zooms + kv_zooms).unique().tolist()

        seq_zoom = min((min(q_zooms + kv_zooms)), seq_len_zoom)  

        if (min(q_zooms + kv_zooms)) < token_zoom:
            raise ValueError(
                f"Zoom level {min(q_zooms + kv_zooms)} is smaller than token_zoom={token_zoom}. "
                "Configure a top-level refine block wrap operation before this attention block."
            )

        self.blocks: nn.ModuleList = nn.ModuleList()
        self.active_groups: List[bool] = active_groups

        input_zoom_field = embed_confs.get("input_zoom", min(q_zooms))
        zoom_key = str(input_zoom_field)
        embedder_cache_key = None
        if global_embedders is not None and zoom_key in global_embedders:
            shared_embedder = global_embedders[zoom_key]
            embedder_cache_key = zoom_key
        else:
            shared_embedder = get_embedder(**embed_confs, grid_layers=grid_layers, zoom=input_zoom_field)
        block = None
        for k, is_active in enumerate(self.active_groups):
            if not is_active:
                continue
            
            # Each group gets its own attention block with group-specific config.
            block = FieldSpaceAttentionBlock(
                        grid_layers,
                        token_zoom,
                        seq_zoom if seq_zoom > -1 else -1,
                        q_zooms,
                        kv_zooms,
                        att_dim,
                        att_dim_mixed = att_dim_mixed,
                        target_zooms = target_zooms,
                        in_features = in_features,
                        token_len_depth= token_len_depth[k],
                        token_len_time= token_len_time[k],
                        token_overlap_space= token_overlap_space[k],
                        token_overlap_time= token_overlap_time[k],
                        token_overlap_depth= token_overlap_depth[k],
                        token_overlap_mlp_time= token_overlap_mlp_time[k],
                        token_overlap_mlp_depth= token_overlap_mlp_depth[k],
                        shared_indexed_variables=shared_indexed_group_variables[k],
                        shared_indexed_depths=shared_indexed_group_depths[k],
                        shared_indexed_space=shared_indexed_group_space[k],
                        rank_space = rank_space[k],
                        n_rank_space = n_rank_space[k],
                        rank_time = rank_time[k],
                        rank_depth = rank_depth[k],
                        rank_features = rank_features[k],
                        rank_variables = rank_variables[k],
                        n_times = n_times[k],
                        n_depths = n_depths[k],
                        seq_len_time= seq_len_time,
                        seq_len_depth= seq_len_depth,
                        seq_overlap_space = seq_overlap_space,
                        seq_overlap_time = seq_overlap_time,
                        seq_overlap_depth = seq_overlap_depth,
                        with_var_att = with_var_att,
                        n_head_channels = n_head_channels,
                        dropout=dropout,
                        embed_confs=embed_confs,
                        embedder=shared_embedder,
                        embedder_cache_key=embedder_cache_key,
                        n_variables=n_groups_variables[k],
                        fac_mode=fac_mode,
                        emb_aggregation=emb_aggregation,
                        update=update,
                        separate_mlp_norm=separate_mlp_norm,
                        mlp_residual_from_attention=mlp_residual_from_attention,
                        use_variable_emb_layer=use_variable_emb_layer,
                        use_variable_layer_norm=use_variable_layer_norm,
                        use_variable_qkv=use_variable_qkv,
                        use_variable_mlp=use_variable_mlp,
                        use_indexed_emb_layer=use_indexed_emb_layer,
                        use_indexed_layer_norm=use_indexed_layer_norm,
                        use_indexed_qkv=use_indexed_qkv,
                        use_indexed_mlp=use_indexed_mlp,
                        use_ranks_emb_layer=use_ranks_emb_layer,
                        use_ranks_qkv=use_ranks_qkv,
                        use_ranks_mlp=use_ranks_mlp,
                        use_variable_att_gammas=use_variable_att_gammas,
                        use_variable_mlp_gammas=use_variable_mlp_gammas,
                        use_indexed_att_gammas=use_indexed_att_gammas,
                        use_indexed_mlp_gammas=use_indexed_mlp_gammas,
                        )
            self.blocks.append(block)

        self.block: Optional[FieldSpaceAttentionBlock] = block
        self.concat_dim = -2 if with_var_att else 0
        self.capture_attention: bool = False
        self.last_attention: Optional[Dict[str, Any]] = None

    def set_attention_capture(self, enabled: bool = True, clear: bool = True) -> None:
        self.capture_attention = bool(enabled)
        if clear or not enabled:
            self.last_attention = None

    def get_last_attention(self, clear: bool = False) -> Optional[Dict[str, Any]]:
        attention = self.last_attention
        if clear:
            self.last_attention = None
        return attention
    
    def forward(
        self,
        x_zooms_groups: List[Dict[int, torch.Tensor]],
        emb_groups: List[Optional[Dict[str, Any]]],
        mask_groups: List[Dict[int, torch.Tensor]] = {},
        sample_configs: Dict[int, Dict[str, Any]] = {}
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Run field-space attention across zoom groups.

        :param x_zooms_groups: List of zoom-to-tensor mappings with tensors shaped like
            ``(b, v, t, n, d, f)``.
        :param emb_groups: List of embedding dictionaries per group.
        :param mask_groups: Optional mask dictionaries per group, with tensors shaped like
            ``(b, v, t, n, d, 1)`` or broadcastable to it.
        :param sample_configs: Sampling configuration per zoom.
        :return: Updated zoom groups with tensors shaped like ``(b, v, t, n, d, f)``.
        """

        capture_attention = bool(getattr(self, "capture_attention", False))
        self.last_attention = None
        x_ress, qs, Ks, Vs, masks, shapes, seq_lens = [], [], [], [], [], [], []
        active_group_indices = [k for k, is_active in enumerate(self.active_groups) if is_active]
        for block, k in zip(self.blocks, active_group_indices):
            # Build per-group Q/K/V tensors and tracking metadata.
            x_res, q, K, V, mask, shape = block.create_QKV(x_zooms_groups[k], emb=emb_groups[k], mask_zooms=mask_groups[k] if self.use_mask else {}, sample_configs=sample_configs)
            x_ress.append(x_res)
            qs.append(q)
            Ks.append(K)
            Vs.append(V)
            masks.append(mask)
            shapes.append(shape)
            seq_lens.append(q.shape[self.concat_dim])
        
        if qs:
            # Concatenate across groups for a single attention call.
            q = torch.concat(qs, dim=self.concat_dim)
            K = torch.concat(Ks, dim=self.concat_dim)
            V = torch.concat(Vs, dim=self.concat_dim)
            mask = torch.concat(masks, dim=self.concat_dim) if self.use_mask else None

            # Shared attention across all groups.
            if capture_attention:
                att_out, attention_weights = safe_scaled_dot_product_attention(
                    q,
                    K,
                    V,
                    mask=mask,
                    return_attention=True,
                )
                self.last_attention = {
                    "weights": attention_weights.detach(),
                    "active_group_indices": tuple(active_group_indices),
                    "group_seq_lens": tuple(int(seq_len) for seq_len in seq_lens),
                    "group_shapes": [dict(shape) for shape in shapes],
                    "group_block_metadata": [
                        {
                            "token_zoom": int(getattr(block, "token_zoom", -1)),
                            "seq_zoom": int(getattr(block, "seq_zoom", -1)),
                            "q_zooms": tuple(int(zoom) for zoom in getattr(block, "q_zooms", ())),
                            "kv_zooms": tuple(int(zoom) for zoom in getattr(block, "kv_zooms", ())),
                            "with_var_att": bool(getattr(block, "with_var_att", False)),
                        }
                        for block in self.blocks
                    ],
                    "concat_dim": self.concat_dim,
                    "q_shape": tuple(q.shape),
                    "k_shape": tuple(K.shape),
                    "v_shape": tuple(V.shape),
                    "mask_shape": None if mask is None else tuple(mask.shape),
                }
            else:
                att_out = safe_scaled_dot_product_attention(q, K, V, mask=mask)

            # Split attention outputs back to per-group chunks.
            att_outs = att_out.split(seq_lens, dim=self.concat_dim)

            for block, k, x_res, att_out_k, shape in zip(self.blocks, active_group_indices, x_ress, att_outs, shapes):
                # Apply per-group MLP updates and merge into zoom tensors.
                x_zooms_groups[k] = block.forward_mlp(
                    x_zooms_groups[k],
                    x_res,
                    att_out_k,
                    shape,
                    emb=emb_groups[k],
                    sample_configs=sample_configs,
                )
        
        for k, x_zooms in enumerate(x_zooms_groups):
            x_zooms_out = {}

            for zoom in self.out_zooms:
                # Keep only requested output zooms.
                x_zooms_out[zoom] = x_zooms[zoom]

            x_zooms_groups[k] = x_zooms_out

        return x_zooms_groups


class FieldSpaceAttentionBlock(nn.Module):
  
    def __init__(
        self,
        grid_layers: Dict[str, GridLayer],
        token_zoom: int,
        seq_zoom: int,
        q_zooms: List[int],
        kv_zooms: List[int],
        att_dim: int,
        att_dim_mixed: int = 0,
        target_zooms: Optional[List[int]] = None,
        in_features: int = 1,
        token_len_depth: int = 1,
        token_len_time: int = 1,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: bool = False,
        token_overlap_mlp_time: bool = False,
        token_overlap_mlp_depth: bool = False,
        shared_indexed_variables: bool = False,
        shared_indexed_depths: bool = False,
        shared_indexed_space: bool = False,
        rank_space: Optional[int] = None,
        n_rank_space: Optional[int] = None,
        rank_time: Optional[int] = None,
        rank_depth: Optional[int] = None,
        rank_features: Optional[int] = None,
        rank_variables: Optional[int] = None,
        n_times: int = 1,
        n_depths: int = 1,
        dropout: float = 0.0,
        n_head_channels: int = 32,
        embed_confs: Dict[str, Any] = {},
        embedder: Optional[nn.Module] = None,
        embedder_cache_key: Optional[str] = None,
        seq_len_time: int = -1,
        seq_len_depth: int = -1,
        seq_overlap_space: bool = False,
        seq_overlap_time: bool = False,
        seq_overlap_depth: bool = False,
        with_var_att: bool = False,
        n_variables: int = 1,
        fac_mode: str = "Tucker",
        emb_aggregation: str = "shift_scale",
        update: str = 'shift',
        layer_norm: bool = True,
        separate_mlp_norm: bool = False,
        mlp_residual_from_attention: bool = False,
        use_variable_emb_layer: bool = True,
        use_variable_layer_norm: bool = True,
        use_variable_qkv: bool = True,
        use_variable_mlp: bool = True,
        use_indexed_emb_layer: Optional[bool] = None,
        use_indexed_layer_norm: Optional[bool] = None,
        use_indexed_qkv: Optional[bool] = None,
        use_indexed_mlp: Optional[bool] = None,
        use_ranks_emb_layer: bool = True,
        use_ranks_qkv: bool = True,
        use_ranks_mlp: bool = True,
        use_variable_att_gammas: bool = False,
        use_variable_mlp_gammas: bool = False,
        use_indexed_att_gammas: Optional[bool] = None,
        use_indexed_mlp_gammas: Optional[bool] = None,
    ) -> None:
        """
        Initialize a field-space attention block.

        :param grid_layers: Mapping from zoom string to GridLayer.
        :param token_zoom: Token zoom level.
        :param seq_zoom: Sequence zoom level for attention.
        :param q_zooms: Query zoom levels.
        :param kv_zooms: Key/value zoom levels.
        :param att_dim: Attention feature dimension.
        :param target_zooms: Optional target zooms for updates.
        :param in_features: Number of input features.
        :param token_len_depth: Token length along depth.
        :param token_len_time: Token length along time.
        :param token_overlap_space: Token overlap along space.
        :param token_overlap_time: Token overlap along time.
        :param token_overlap_depth: Token overlap along depth.
        :param token_overlap_mlp_time: MLP overlap along time.
        :param token_overlap_mlp_depth: MLP overlap along depth.
        :param rank_space: Optional rank for space.
        :param rank_time: Optional rank for time.
        :param rank_depth: Optional rank for depth.
        :param rank_features: Optional rank for features.
        :param dropout: Dropout rate.
        :param n_head_channels: Head channel size.
        :param embed_confs: Embedding configuration dictionary.
        :param seq_len_time: Sequence length along time.
        :param seq_len_depth: Sequence length along depth.
        :param seq_overlap_space: Overlap along space.
        :param seq_overlap_time: Overlap along time.
        :param seq_overlap_depth: Overlap along depth.
        :param with_var_att: Whether to include variable attention.
        :param layer_confs: Layer configuration for attention blocks.
        :param layer_confs_emb: Layer configuration for embedding blocks.
        :param update: Update mode ("shift" or "shift_scale").
        :param layer_norm: Whether to apply layer norm in embedding layers.
        :param separate_mlp_norm: Whether to separate MLP norm.
        :param mlp_residual_from_attention: Whether the MLP residual uses the
            post-attention tensor instead of the original zoom tensor.
        :param use_variable_emb_layer: Whether embedding layers use variable-specific parameters.
        :param use_variable_layer_norm: Whether embedding-layer layer norms use variable-specific affine params.
        :param use_variable_qkv: Whether Q/KV/attention projection layers use variable-specific parameters.
        :param use_variable_mlp: Whether the MLP branch uses variable-specific parameters.
        :param use_ranks_emb_layer: Whether embedding layers use the configured ranks.
        :param use_ranks_qkv: Whether Q/KV/attention projection layers use the configured ranks.
        :param use_ranks_mlp: Whether the MLP branch uses the configured ranks.
        :param use_variable_att_gammas: Whether attention residual gammas are variable-specific.
        :param use_variable_mlp_gammas: Whether MLP residual gammas are variable-specific.
        :return: None.
        """
               
        super().__init__()

        target_zooms = q_zooms if target_zooms is None else target_zooms
        self.seq_overlap_time: bool = seq_overlap_time
        self.seq_overlap_depth: bool = seq_overlap_depth
        self.seq_overlap_space: bool = seq_overlap_space if seq_zoom > -1 else False
        self.token_len_time: int = token_len_time
        self.token_len_depth: int = token_len_depth
        self.token_overlap_depth: bool = token_overlap_depth
        self.token_overlap_time: bool = token_overlap_time

        # Resolve grid layers used for tokenization and attention sequencing.
        grid_layer_field = grid_layers[str(token_zoom)] if token_zoom >-1 else grid_layers[str(0)]
        grid_layer_att = grid_layers[str(seq_zoom)] if seq_zoom >-1 else -1

        global_update = token_zoom == -1

        if global_update:
            token_overlap_space = 0

        self.att_dim: int = att_dim

        # Build rank configuration used by factorized layers.
        ranks = [rank_time, rank_space, rank_depth, rank_features, rank_features]
        shared_ranks = [None] * len(ranks)

        emb_ranks_default = embed_confs.get("ranks", [*ranks, None])
        shared_emb_ranks = [None] * len(emb_ranks_default)

        def _resolve_alias(indexed_value: Optional[bool], legacy_value: bool) -> bool:
            return legacy_value if indexed_value is None else indexed_value

        use_indexed_emb_layer = _resolve_alias(use_indexed_emb_layer, use_variable_emb_layer)
        use_indexed_layer_norm = _resolve_alias(use_indexed_layer_norm, use_variable_layer_norm)
        use_indexed_qkv = _resolve_alias(use_indexed_qkv, use_variable_qkv)
        use_indexed_mlp = _resolve_alias(use_indexed_mlp, use_variable_mlp)
        use_indexed_att_gammas = _resolve_alias(use_indexed_att_gammas, use_variable_att_gammas)
        use_indexed_mlp_gammas = _resolve_alias(use_indexed_mlp_gammas, use_variable_mlp_gammas)

        n_variables_emb = n_variables if use_variable_emb_layer else 1
        n_variables_norm = n_variables if use_variable_layer_norm else 1
        n_variables_qkv = n_variables if use_variable_qkv else 1
        n_variables_mlp = n_variables if use_variable_mlp else 1

        ranks_emb = ranks if use_ranks_emb_layer else shared_ranks
        ranks_qkv = ranks if use_ranks_qkv else shared_ranks
        ranks_mlp = ranks if use_ranks_mlp else shared_ranks
        rank_variables_qkv = rank_variables if use_ranks_qkv else None

        self.scale_shift: bool = update == 'shift_scale'
        
        global_att = isinstance(grid_layer_att, int) and grid_layer_att == -1

        self.n_head_channels: int = n_head_channels
        self.grid_layer_field: GridLayer = grid_layer_field
        self.grid_layer_att: Union[GridLayer, int] = grid_layer_att

        self.emb_layers: nn.ModuleDict = nn.ModuleDict()
        self.mlp_emb_layers: nn.ModuleDict = nn.ModuleDict()
        self.q_layers: nn.ModuleDict = nn.ModuleDict()
        self.kv_layers: nn.ModuleDict = nn.ModuleDict()
        self.mlps: nn.ModuleDict = nn.ModuleDict()
        self.out_layers: nn.ModuleDict = nn.ModuleDict()

        self.dropout_att: nn.Module = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.dropout_mlp: nn.Module = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.q_zooms: List[int] = q_zooms
        self.kv_zooms: List[int] = kv_zooms

        # Output update size depends on whether we emit shift+scale or shift-only.
        update_dim = in_features
        update_dim = 2 * in_features if update == 'shift_scale' else update_dim
        
        if len(self.q_zooms) == len(self.kv_zooms):
            self.self_att: bool = True
            self.self_att = ((torch.tensor(q_zooms) - torch.tensor(kv_zooms)) == 0).all()
        else:
            self.self_att = False

        self.token_zoom: int = grid_layer_field.zoom
        self.seq_zoom: int = seq_zoom
        self.with_var_att: bool = with_var_att

        self.q_projection_layers: nn.ModuleDict = nn.ModuleDict()
        self.kv_projection_layers: nn.ModuleDict = nn.ModuleDict()
        self.gammas: nn.ParameterDict = nn.ParameterDict()

        # Tokenizers define how each zoom is chunked into tokens for attention.
        tokenizer_update = Tokenizer(target_zooms, 
                                    token_zoom,
                                    grid_layers=grid_layers,
                                    overlap_thickness=int(token_overlap_space),
                                    token_len_time=token_len_time,
                                    token_len_depth=token_len_depth)
        
        self.tokenizer: Tokenizer = Tokenizer(q_zooms, 
                                    token_zoom,
                                    grid_layers=grid_layers,
                                    overlap_thickness=int(token_overlap_space),
                                    token_len_time=token_len_time,
                                    token_len_depth=token_len_depth)
        if not self.self_att:
            self.kv_tokenizer: Tokenizer = Tokenizer(kv_zooms, 
                                     token_zoom,
                                     grid_layers=grid_layers, 
                                     overlap_thickness=int(token_overlap_space),
                                     token_len_time=token_len_time,
                                     token_len_depth=token_len_depth)
        else:
            self.kv_tokenizer = self.tokenizer

        _, n_out_features_update = tokenizer_update.get_features()
        self.n_out_features_update: Dict[int, int] = n_out_features_update
        n_in_features_zooms_q, n_out_features_zooms_q = self.tokenizer.get_features()
        self.n_in_features_zooms_q: Dict[int, int] = n_in_features_zooms_q
        self.n_out_features_zooms_q: Dict[int, int] = n_out_features_zooms_q
        n_in_features_zooms_kv, n_out_features_zooms_kv = self.kv_tokenizer.get_features()
        self.n_in_features_zooms_kv: Dict[int, int] = n_in_features_zooms_kv
        self.n_out_features_zooms_kv: Dict[int, int] = n_out_features_zooms_kv
        
        # Token shapes used for Q/KV projections and updates.
        self.token_size_space: List[int] = [token_len_time, sum(self.n_in_features_zooms_q.values()), token_len_depth, in_features]
        self.token_size_space_kv: List[int] = [token_len_time, sum(self.n_in_features_zooms_kv.values()), token_len_depth, in_features]
        self.token_size_update: List[int] = [token_len_time, sum(self.n_out_features_update.values()), token_len_depth, in_features]

        token_size_in_overlap = [token_len_time + 2 * token_overlap_time, sum(self.n_in_features_zooms_q.values()), token_len_depth + 2 * token_overlap_depth, in_features]
        token_size_in_mlp_overlap = [token_len_time + 2 * token_overlap_mlp_time, sum(self.n_in_features_zooms_q.values()), token_len_depth + 2 * token_overlap_mlp_depth, in_features]
        token_size_in_kv_overlap = [token_len_time + 2 * token_overlap_time, sum(self.n_in_features_zooms_kv.values()), token_len_depth + 2 * token_overlap_depth, in_features]

        self.separate_mlp_norm: bool = separate_mlp_norm
        self.mlp_residual_from_attention: bool = mlp_residual_from_attention

        # Optional embedding path for conditioning.
        input_zoom_field = embed_confs.get("input_zoom", min(q_zooms))
        if embedder is None:
            embedder = get_embedder(**embed_confs, grid_layers=grid_layers, zoom=input_zoom_field)

        emb_tokenizer = Tokenizer(
            input_zooms=[input_zoom_field] if embedder and embedder.has_space() else [],
            token_zoom=token_zoom,
            token_len_time=token_len_time if embedder and embedder.has_time() else 1,
            token_len_depth=token_len_depth if embedder and embedder.has_depth() else 1,
            overlap_thickness=int(embed_confs.get("token_overlap_space", False)),
            grid_layers=grid_layers
        ) 

        emb_ranks = emb_ranks_default if use_ranks_emb_layer else shared_emb_ranks

        emb_tokenizer_out_features = copy.deepcopy(self.token_size_space)
        emb_tokenizer_out_features[1] = self.token_size_space[1] if embedder and embedder.has_space() else 1

        def _build_branch_indexed_dims(
            *,
            n_variables_local: int = 1,
            rank_variables_local: Optional[int] = None,
            include_indexing: bool = True,
        ) -> Dict[str, Dict[str, Any]]:
            if not include_indexing:
                return {}

            indexed_n_depths = max(1, int(n_depths) // max(1, int(token_len_depth))) if int(n_depths) > 1 else 1
            indexed_n_space = 12 * 4**int(token_zoom) if n_rank_space is not None and int(n_rank_space) > 0 and int(token_zoom) >= 0 else 1
            indexed_rank_space = int(n_rank_space) if indexed_n_space > 1 else None

            return build_indexed_dims(
                n_variables=int(n_variables_local),
                rank_variables=rank_variables_local,
                same_values_variables=shared_indexed_variables,
                n_times=int(n_times) if int(n_times) > 1 else 1,
                n_space=indexed_n_space,
                rank_space=indexed_rank_space,
                same_values_space=shared_indexed_space,
                n_depths=indexed_n_depths,
                same_values_depths=shared_indexed_depths,
            )

        indexed_dims_emb = _build_branch_indexed_dims(
            n_variables_local=n_variables if use_variable_emb_layer else 1,
            rank_variables_local=rank_variables if use_ranks_emb_layer else None,
            include_indexing=use_indexed_emb_layer,
        )
        indexed_dims_norm = _build_branch_indexed_dims(
            n_variables_local=n_variables if use_variable_layer_norm else 1,
            rank_variables_local=None,
            include_indexing=use_indexed_layer_norm,
        )
        indexed_dims_emb_kv = _build_branch_indexed_dims(
            n_variables_local=n_variables if use_variable_emb_layer else 1,
            rank_variables_local=rank_variables if use_ranks_emb_layer else None,
            include_indexing=use_indexed_emb_layer,
        )
        indexed_dims_norm_kv = _build_branch_indexed_dims(
            n_variables_local=n_variables if use_variable_layer_norm else 1,
            rank_variables_local=None,
            include_indexing=use_indexed_layer_norm,
        )
        indexed_dims_qkv = _build_branch_indexed_dims(
            n_variables_local=n_variables if use_variable_qkv else 1,
            rank_variables_local=rank_variables_qkv,
            include_indexing=use_indexed_qkv,
        )
        indexed_dims_kv = _build_branch_indexed_dims(
            n_variables_local=n_variables if use_variable_qkv else 1,
            rank_variables_local=rank_variables_qkv,
            include_indexing=use_indexed_qkv,
        )
        indexed_dims_mlp = _build_branch_indexed_dims(
            n_variables_local=n_variables if use_variable_mlp else 1,
            rank_variables_local=None,
            include_indexing=use_indexed_mlp,
        )
        self.indexed_dims_att_gammas = _build_branch_indexed_dims(
            n_variables_local=n_variables,
            rank_variables_local=None,
            include_indexing=use_indexed_att_gammas,
        )
        self.indexed_dims_mlp_gammas = _build_branch_indexed_dims(
            n_variables_local=n_variables,
            rank_variables_local=None,
            include_indexing=use_indexed_mlp_gammas,
        )

        # Embed Q with optional positional/field embedding layer.
        self.emb_layer_q_field: LinEmbLayer = LinEmbLayer(
            emb_tokenizer_out_features,
            emb_tokenizer_out_features,
            ranks=ranks_emb,
            n_variables=n_variables_emb,
            n_variable_norm=n_variables_norm,
            indexed_dims=indexed_dims_emb,
            indexed_dims_norm=indexed_dims_norm,
            fac_mode=fac_mode,
            identity_if_equal=True,
            embedder=embedder,
            field_tokenizer= emb_tokenizer,
            output_zoom=max(self.q_zooms),
            layer_norm=True,
            emb_aggregation=emb_aggregation,
            emb_ranks=emb_ranks,
            embedder_cache_key=embedder_cache_key,
        )
        
        # Optional separate normalization for MLP path.
        if separate_mlp_norm:
            self.emb_layer_mlp: Optional[LinEmbLayer] = LinEmbLayer(
                self.token_size_space,
                self.token_size_space,
                ranks=ranks_emb,
                n_variables=n_variables_emb,
                n_variable_norm=n_variables_norm,
                indexed_dims=indexed_dims_emb,
                indexed_dims_norm=indexed_dims_norm,
                fac_mode=fac_mode,
                identity_if_equal=True,
                embedder=embedder,
                field_tokenizer= emb_tokenizer,
                output_zoom=max(self.q_zooms),
                layer_norm=layer_norm,
                emb_aggregation=emb_aggregation,
                emb_ranks=emb_ranks,
                embedder_cache_key=embedder_cache_key,
            )
        else:
            self.emb_layer_mlp = None

        # Only build KV embedder when doing cross-attention.
        if not self.self_att:
            self.emb_layer_kv: Optional[LinEmbLayer] = LinEmbLayer(
                self.token_size_space_kv,
                self.token_size_space_kv,
                ranks=ranks_emb,
                n_variables=n_variables_emb,
                n_variable_norm=n_variables_norm,
                indexed_dims=indexed_dims_emb_kv,
                indexed_dims_norm=indexed_dims_norm_kv,
                fac_mode=fac_mode,
                identity_if_equal=True,
                embedder=embedder,
                field_tokenizer= emb_tokenizer,
                output_zoom=max(self.q_zooms),
                layer_norm=layer_norm,
                emb_aggregation=emb_aggregation,
                emb_ranks=emb_ranks,
                embedder_cache_key=embedder_cache_key,
            )
        else:
            self.emb_layer_kv = None

        out_dim_q = [1, 1 , 1, att_dim] 
        out_dim_kv = [1, 1, 1, 2 * att_dim]

        update_dims = [*self.token_size_space[:-1], update_dim]
        update_dims_mlp = [*self.token_size_update[:-1], update_dim]

        # Linear projections into attention space.
        self.q_projection_layer = get_layer(token_size_in_overlap, out_dim_q, ranks=ranks_qkv, n_variables=n_variables_qkv, indexed_dims=indexed_dims_qkv, fac_mode=fac_mode, rank_variables=rank_variables_qkv, bias=False)
        self.kv_projection_layer = get_layer(token_size_in_kv_overlap, out_dim_kv, ranks=ranks_qkv, n_variables=n_variables_qkv, indexed_dims=indexed_dims_kv, fac_mode=fac_mode, rank_variables=rank_variables_qkv, bias=True)
        self.out_layer_att = get_layer([1,1,1, att_dim+att_dim_mixed], update_dims, ranks=ranks_qkv, n_variables=n_variables_qkv, indexed_dims=indexed_dims_qkv, fac_mode=fac_mode, rank_variables=rank_variables_qkv)
        
        self.att_dim_mixed = att_dim_mixed
        if att_dim_mixed > 0:
            assert n_variables>1, "n_variables need to be fixed and >1 for att_dim_mixed > 0" 
            in_size_q = token_size_in_overlap[:-1] + [token_size_in_overlap[-1] * n_variables]
            in_size_kv = token_size_in_overlap[:-1] + [token_size_in_overlap[-1] * n_variables]
            self.q_projection_layer_mixed = get_layer(in_size_q, [1, 1 , 1, att_dim_mixed], ranks=ranks_qkv, n_variables=1, indexed_dims=indexed_dims_qkv, fac_mode=fac_mode, rank_variables=rank_variables_qkv)
            self.kv_projection_layer_mixed = get_layer(in_size_kv, [1, 1 , 1, att_dim_mixed*2], ranks=ranks_qkv, n_variables=1, indexed_dims=indexed_dims_kv, fac_mode=fac_mode, rank_variables=rank_variables_qkv)
            self.mixed_pattern: str = 'b v T N D t n d f -> b 1 T N D t n d (v f)'

        # Learned residual scaling for attention and MLP updates.
        self.use_variable_att_gammas: bool = use_variable_att_gammas
        self.use_variable_mlp_gammas: bool = use_variable_mlp_gammas
        self.use_indexed_att_gammas: bool = use_indexed_att_gammas
        self.use_indexed_mlp_gammas: bool = use_indexed_mlp_gammas

        gamma_indexed_shape_att = [spec["n_features"] for spec in self.indexed_dims_att_gammas.values()]
        gamma_shape_att = [*gamma_indexed_shape_att, *self.token_size_space] if use_indexed_att_gammas else self.token_size_space
        self.gamma_res = nn.Parameter(torch.ones(gamma_shape_att) * 1e-12, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(gamma_shape_att) * 1e-12, requires_grad=True)

        self.mlp = MLP_fac(
            token_size_in_mlp_overlap,
            update_dims_mlp,
            hidden_dim=[1,1,1,att_dim+att_dim_mixed],
            dropout=dropout,
            ranks=ranks_mlp,
            n_variables=n_variables_mlp,
            indexed_dims=indexed_dims_mlp,
            fac_mode=fac_mode,
            gamma=False,
        )
        gamma_indexed_shape_mlp = [spec["n_features"] for spec in self.indexed_dims_mlp_gammas.values()]
        gamma_shape_mlp = [len(target_zooms), *gamma_indexed_shape_mlp] if use_indexed_mlp_gammas else [len(target_zooms)]
        self.gamma_res_mlp = nn.Parameter(torch.ones(gamma_shape_mlp) * 1e-12, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(gamma_shape_mlp) * 1e-12, requires_grad=True)

        self.pattern_tokens: str = 'b v (T t) N n (D d) f ->  b v T N D t n d f'
        self.pattern_tokens_reverse: str = 'b v T N D t n d f ->  b v (T t) (N n) (D d) f'
        self.pattern_tokens_fold: str = 'b v T N D t n d f ->  b v T N D (t n d f)'

        self.pattern_tokens_nh_space: str = 'b v T N NH D (t n d f) -> b v T N D t (n NH) d f'

        self.att_pattern_chunks: str = 'b v (T t) (N n) (D d) 1 1 1 f ->  b v T N D t n d f'
        self.att_pattern_chunks_w_nh: str = 'b v (T t) N n (D d) 1 1 1 f ->  b v T N D t n d f'
        # Shapes for token chunking and attention packing.
        self.rearrange_dict: Dict[str, int] = {}
        if global_att:
            self.rearrange_dict.update({'N': 1})
            self.seq_overlap_space = False
        else:
            self.rearrange_dict.update({'n': 4**(grid_layer_field.zoom-grid_layer_att.zoom)})
        
        if seq_len_time ==-1:
            self.rearrange_dict.update({'T': 1})
            self.seq_overlap_time = False
        else:
            self.rearrange_dict.update({'t': seq_len_time})

        if seq_len_depth==-1:
            self.rearrange_dict.update({'D': 1})
            self.seq_overlap_depth = False
        else:
            self.rearrange_dict.update({'d': seq_len_depth})

        self.rearrange_dict_nh: Dict[str, int] = self.rearrange_dict.copy()
        if seq_zoom > -1:
            self.rearrange_dict_nh['n'] = self.grid_layer_att.adjc.shape[-1] * 4**(self.token_zoom - seq_zoom)
        
        self.att_pattern: str
        self.mask_pattern: str
        self.att_pattern_reverse: str
        if with_var_att:
            # Variable-aware attention packs variable dimension into sequence.
            self.att_pattern: str = 'b v T N D t n d (NH H) -> (b T N D) NH (v t n d) H'
            self.mask_pattern: str = 'b v T N D t n d 1 -> (b T N D) 1 1 (v t n d)'
            self.att_pattern_reverse: str = '(b T N D) NH (v t n d) H -> b v (T t) (N n) (D d) 1 1 1 (NH H)'

        else:
            # Standard attention packs only token dims into sequence.
            self.att_pattern = 'b v T N D t n d (NH H) -> (b v T N D) NH (t n d) H'
            self.mask_pattern = 'b v T N D t n d 1 -> (b v T N D) 1 1 (v t n d)'
            self.att_pattern_reverse = '(b v T N D) NH (t n d) H -> b v (T t) (N n) (D d) 1 1 1 (NH H)'

        self.capture_attention: bool = False
        self.last_attention: Optional[Dict[str, Any]] = None

    def set_attention_capture(self, enabled: bool = True, clear: bool = True) -> None:
        self.capture_attention = bool(enabled)
        if clear or not enabled:
            self.last_attention = None

    def get_last_attention(self, clear: bool = False) -> Optional[Dict[str, Any]]:
        attention = self.last_attention
        if clear:
            self.last_attention = None
        return attention

    def get_ms_features(self, zooms: List[int]) -> Dict[int, int]:
        """
        Compute multiscale feature sizes per zoom.

        :param zooms: List of zoom levels.
        :return: Mapping from zoom to feature size.
        """
        features = {}
        for zoom in zooms:
            if self.token_zoom == 0:
                features[zoom] = max([12*4**(zoom - self.token_zoom),1])
            else: 
                features[zoom] = max([4**(zoom - self.token_zoom),1])
        return features
    

    def get_time_depth_overlaps(
        self,
        x: torch.Tensor,
        overlap_time: bool = False,
        overlap_depth: bool = False,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply time/depth overlap padding to tokenized tensors.

        :param x: Input tensor of shape ``(b, v, T, N, D, t, n, d, f)``.
        :param overlap_time: Whether to add time overlap.
        :param overlap_depth: Whether to add depth overlap.
        :param mask: Optional mask tensor (unused).
        :return: Tensor with overlap padding applied.
        """
        # Time/depth overlap is used to include neighbor tokens for smoother transitions.
        if overlap_time:
            x = add_time_overlap_from_neighbor_patches(x, overlap=1, pad_mode= "edge")
        
        if overlap_depth:
            x = add_depth_overlap_from_neighbor_patches(x, overlap=1, pad_mode= "edge")

        return x
    
    
    def select_emb(self, emb: Optional[Dict[str, Any]], sample_configs: Optional[Dict[str, Any]] = None):
        """
        Select embedding entries for the active zooms.

        :param emb: Embedding dictionary or None.
        :param sample_configs: Optional sampling configuration dictionary.
        :return: Filtered embedding dictionary or None.
        """
        if sample_configs is None:
            sample_configs = {}

        if emb is None:
            return None

        # Shallow copy to avoid mutating the caller's embeddings.
        emb_cpy = dict(emb)
        for emb_key in ("TimeEmbedder", "TimeProgressEmbedder"):
            if emb_key not in emb_cpy or not isinstance(emb_cpy[emb_key], dict):
                continue
            emb_cpy[emb_key] = {max(self.q_zooms): emb_cpy[emb_key][max(self.q_zooms)]}

        return emb_cpy

    def _get_att_gamma(
        self,
        gamma: torch.Tensor,
        reference: torch.Tensor,
        emb: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Select and broadcast attention residual gammas.

        :param gamma: Shared gamma of shape ``(t, n, d, f)`` or variable-specific gamma of
            shape ``(n_variables, t, n, d, f)``.
        :param reference: Reference tensor defining the target broadcast shape.
        :param emb: Optional embedding dict containing variable indices.
        :return: Gamma broadcastable to ``(b, v, T, N, D, t, n, d, f)``.
        """
        if not self.use_indexed_att_gammas:
            return gamma

        return broadcast_indexed_tensor(gamma, self.indexed_dims_att_gammas, reference, emb=emb)

    def _get_mlp_gamma(
        self,
        gamma: torch.Tensor,
        idx: int,
        reference: torch.Tensor,
        emb: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Select and broadcast MLP residual gammas.

        :param gamma: Shared gamma of shape ``(n_zooms,)`` or variable-specific gamma of
            shape ``(n_zooms, n_variables)``.
        :param idx: Target zoom index.
        :param reference: Reference tensor defining the target broadcast shape.
        :param emb: Optional embedding dict containing variable indices.
        :return: Gamma broadcastable to ``(b, v, t, n, d, f)``.
        """
        if not self.use_indexed_mlp_gammas:
            return gamma[idx]

        return broadcast_indexed_tensor(gamma[idx], self.indexed_dims_mlp_gammas, reference, emb=emb)
    
    def create_QKV(
        self,
        x_zooms: Dict[int, torch.Tensor],
        emb: Optional[Dict[str, Any]] = None,
        sample_configs: Dict[int, Dict[str, Any]] = {},
        mask_zooms: Dict[int, torch.Tensor] = {}
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, int]]:
        """
        Create Q/K/V tensors and masks from zoomed inputs.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration per zoom.
        :param mask_zooms: Optional mask tensors per zoom shaped like ``(b, v, t, n, d, 1)``
            or broadcastable to it.
        :return: Tuple of (x_base, q, K, V, mask, shape). `x_base` is tokenized to
            ``(b, v, T, N, D, t, n, d, f)``. `q`, `K`, `V` are packed attention tensors
            shaped like ``(b*v*T*N*D, NH, t*n*d, H)``. `mask` (if present) is shaped like
            ``(b*v*T*N*D, 1, 1, t*n*d)``.
        """
        zoom_field = self.grid_layer_field.zoom

        # Tokenize input zoom tensors for attention.
        x = self.tokenizer(x_zooms, sample_configs)

        emb_tokenized = emb#self.select_emb(emb)

        # Q path may include embedding projection.
        if self.emb_layer_q_field is not None:
            q = self.emb_layer_q_field(x, emb=emb_tokenized, sample_configs=sample_configs)

        x_base = q if not self.separate_mlp_norm else x

        q = self.get_time_depth_overlaps(q, overlap_time=self.token_overlap_time, overlap_depth=self.token_overlap_depth)

        # KV tokens come from a dedicated tokenizer for cross-attention.
        if not self.self_att:
            kv = self.kv_tokenizer(x_zooms, sample_configs)
            kv = self.emb_layer_kv(kv, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])
            kv = self.get_time_depth_overlaps(kv, overlap_time=self.token_overlap_time, overlap_depth=self.token_overlap_depth)
        else:
            kv = q

        if self.att_dim_mixed > 0:
            q_mixed = rearrange(q, self.mixed_pattern)
            kv_mixed = rearrange(kv, self.mixed_pattern)

            q_mixed: torch.Tensor = self.q_projection_layer_mixed(q_mixed, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])
            kv_mixed: torch.Tensor = self.kv_projection_layer_mixed(kv_mixed, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])

        # Project to attention feature space.
        q = self.q_projection_layer(q, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])
        kv = self.kv_projection_layer(kv, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])

        if self.att_dim_mixed > 0:
            q = torch.concat((q, q_mixed.expand(-1, q.shape[1], -1, -1, -1, -1, -1, -1, -1)), dim=-1)
            kv = torch.concat((kv, kv_mixed.expand(-1, kv.shape[1], -1, -1, -1, -1, -1, -1, -1)), dim=-1)

        zoom_field = self.grid_layer_field.zoom

        # Chunk tokens into attention-friendly layout.
        q = rearrange(q, self.att_pattern_chunks, **self.rearrange_dict)

        mask = mask_zooms[zoom_field] if zoom_field in mask_zooms.keys() else None
        # Optional spatial neighborhood expansion for KV.
        if self.seq_overlap_space:
            kv, mask = self.grid_layer_att.get_nh(kv, input_zoom=zoom_field, sample_configs=sample_configs[zoom_field], mask=mask)
            kv = rearrange(kv, self.att_pattern_chunks_w_nh, **self.rearrange_dict_nh)
        else:
            kv = rearrange(kv, self.att_pattern_chunks, **self.rearrange_dict)

        # Apply time/depth overlap to KV and mask if configured.
        kv = self.get_time_depth_overlaps(kv, overlap_time=self.seq_overlap_time, overlap_depth=self.seq_overlap_depth)
        
        if mask is not None:
            mask = self.get_time_depth_overlaps(mask, overlap_time=self.seq_overlap_time, overlap_depth=self.seq_overlap_depth)

        K, V = kv.chunk(2, dim=-1)

        b, v, T, N, D, t, n, d, f = q.shape
        # Pack heads and token dims for scaled dot-product attention.
        q = rearrange(q, self.att_pattern, H=self.n_head_channels)
        K = rearrange(K, self.att_pattern, H=self.n_head_channels)
        V = rearrange(V, self.att_pattern, H=self.n_head_channels)

        mask = rearrange(mask, self.mask_pattern) if mask is not None else None

        shape = {'b': b, 'v': v, 'T': T, 'N': N, 'D': D, 't': t, 'n': n, 'd': d}

        return x_base, q, K, V, mask, shape
    
    def attend(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sample_configs: Dict[int, Dict[str, Any]] = {}
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Apply attention for given Q and KV tensors.

        :param q: Query tensor.
        :param kv: Key/value tensor.
        :param mask: Optional attention mask.
        :param sample_configs: Sampling configuration per zoom.
        :return: Tuple of (attention_output, shape_metadata). `attention_output` is shaped
            like ``(b*v*T*N*D, NH, t*n*d, H)``.
        """
        zoom_field = self.grid_layer_field.zoom

        q = rearrange(q, self.att_pattern_chunks, **self.rearrange_dict)

        # Match attention layout to create Q/K/V blocks.
        if self.seq_overlap_space:
            kv, mask = self.grid_layer_att.get_nh(kv, input_zoom=zoom_field, sample_configs=sample_configs[zoom_field], mask=mask)
            kv = rearrange(kv, self.att_pattern_chunks_w_nh, **self.rearrange_dict_nh)
        else:
            kv = rearrange(kv, self.att_pattern_chunks, **self.rearrange_dict)

        kv = self.get_time_depth_overlaps(kv, overlap_time=self.seq_overlap_time, overlap_depth=self.seq_overlap_depth)
        
        if mask is not None:
            mask = self.get_time_depth_overlaps(mask, overlap_time=self.seq_overlap_time, overlap_depth=self.seq_overlap_depth)

        K, V = kv.chunk(2, dim=-1)

        b, v, T, N, D, t, n, d, f = q.shape
        q = rearrange(q, self.att_pattern, H=self.n_head_channels)
        K = rearrange(K, self.att_pattern, H=self.n_head_channels)
        V = rearrange(V, self.att_pattern, H=self.n_head_channels)

        mask = rearrange(mask, self.mask_pattern) if mask is not None else None

        # Scaled dot-product attention over packed tokens.
        att_out = safe_scaled_dot_product_attention(q, K, V, mask=mask)

        # Restore attention output to token layout.
        att_out = rearrange(att_out, self.att_pattern_reverse, b=b, v=v, T=T, N=N, D=D, t=t, n=n, d=d)

        shape = {'b': b, 'v': v, 'T': T, 'N': N, 'D': D, 't': t, 'n': n, 'd': d}
        return att_out, shape

    def forward_mlp(
        self,
        x_zooms: Dict[int, torch.Tensor],
        x_base: torch.Tensor,
        att_out: torch.Tensor,
        shape: Dict[str, int],
        emb: Optional[Dict[str, Any]] = None,
        sample_configs: Dict[int, Dict[str, Any]] = {}
    ) -> Dict[int, torch.Tensor]:
        """
        Apply MLP updates to attention outputs and merge into zoomed tensors.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param x_base: Base tensor used for residual updates.
        :param att_out: Attention output tensor shaped like ``(b*v*T*N*D, NH, t*n*d, H)``.
        :param shape: Shape metadata for rearranging.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration per zoom.
        :return: Updated zoom tensors.
        """
        emb_tokenized = emb

        att_out = rearrange(att_out, self.att_pattern_reverse, **shape)

        zoom_field = self.grid_layer_field.zoom

        # Project attention output back to update dimension.
        att_out = self.out_layer_att(att_out, emb=emb_tokenized, sample_configs=sample_configs)
        gamma_res_att = self._get_att_gamma(self.gamma_res, x_base, emb_tokenized)
        gamma_att = self._get_att_gamma(self.gamma, x_base, emb_tokenized)
        if self.scale_shift:
            scale, shift = self.dropout_att(att_out).chunk(2, dim=-1)
            # Apply scale/shift residual update.
            x = x_base * (1 + gamma_res_att * self.dropout_att(scale)) + gamma_att * self.dropout_att(shift)
        else:
            # Simple residual update when only shift is used.
            x = (1 + gamma_res_att) * x_base + gamma_att * self.dropout_att(att_out)

        residual_bases = None
        if self.mlp_residual_from_attention:
            residual_bases = {}
            x_residual_splits = x.split(tuple(self.n_out_features_update.values()), dim=-3)
            for k, (zoom, n) in enumerate(self.n_out_features_update.items()):
                residual_base = rearrange(x_residual_splits[k], self.pattern_tokens_reverse, n=n)
                residual_bases[zoom] = insert_matching_time_patch(
                    x_zooms[zoom],
                    residual_base,
                    zoom,
                    max(self.q_zooms),
                    sample_configs,
                )

        if self.separate_mlp_norm and self.emb_layer_mlp is not None:
            x = self.emb_layer_mlp(x, emb=emb_tokenized, sample_configs=sample_configs)

        # MLP update path operating on tokenized representation.
        x = self.mlp(x, emb=emb_tokenized, sample_configs=sample_configs[int(zoom_field)])

        # Split per-zoom outputs and fold them back into zoom tensors.
        x = x.split(tuple(self.n_out_features_update.values()), dim=-3)

        for k, (zoom, n) in enumerate(self.n_out_features_update.items()):
            if x_zooms and x is not None:
                x_out = rearrange(x[k], self.pattern_tokens_reverse, n=n)
                gamma_res_mlp = self._get_mlp_gamma(self.gamma_res_mlp, k, x_zooms[zoom], emb_tokenized)
                gamma_mlp = self._get_mlp_gamma(self.gamma_mlp, k, x_zooms[zoom], emb_tokenized)
                residual_base = residual_bases[zoom] if residual_bases is not None else x_zooms[zoom]

                if self.scale_shift:
                    scale, shift = x_out.chunk(2, dim=-1)
                    shift = insert_matching_time_patch(x_zooms[zoom], shift, zoom, max(self.q_zooms), sample_configs)
                    scale = insert_matching_time_patch(x_zooms[zoom], scale, zoom, max(self.q_zooms), sample_configs)
                    x_zooms[zoom] = residual_base * (1 + gamma_res_mlp * scale) + gamma_mlp * shift
                else:
                    x_out = insert_matching_time_patch(x_zooms[zoom], x_out, zoom, max(self.q_zooms), sample_configs)
                    # Simple residual update at each zoom.
                    x_zooms[zoom] = (1 + gamma_res_mlp) * residual_base + gamma_mlp * x_out

        return x_zooms

    def forward(
        self,
        x_zooms: Dict[int, torch.Tensor] = {},
        mask_zooms: Dict[int, torch.Tensor] = {},
        emb: Optional[Dict[str, Any]] = None,
        sample_configs: Dict[int, Dict[str, Any]] = {}
    ) -> Dict[int, torch.Tensor]:
        """
        Run the full attention block on zoomed inputs.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param mask_zooms: Optional masks per zoom.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration per zoom.
        :return: Updated zoom tensors shaped like ``(b, v, t, n, d, f)``.
        """
        capture_attention = bool(getattr(self, "capture_attention", False))
        self.last_attention = None
        x_base, q, K, V, mask, shape = self.create_QKV(
            x_zooms,
            emb=emb,
            sample_configs=sample_configs,
            mask_zooms=mask_zooms,
        )
        if capture_attention:
            att_out, attention_weights = safe_scaled_dot_product_attention(
                q,
                K,
                V,
                mask=mask,
                return_attention=True,
            )
            self.last_attention = {
                "weights": attention_weights.detach(),
                "shape": dict(shape),
                "q_shape": tuple(q.shape),
                "k_shape": tuple(K.shape),
                "v_shape": tuple(V.shape),
                "mask_shape": None if mask is None else tuple(mask.shape),
            }
        else:
            att_out = safe_scaled_dot_product_attention(q, K, V, mask=mask)
        return self.forward_mlp(
            x_zooms,
            x_base,
            att_out,
            shape,
            emb=emb,
            sample_configs=sample_configs,
        )
