from typing import Any, Dict, List, Optional, Tuple, Union
from einops import rearrange
import copy
from omegaconf import ListConfig

import torch
import torch.nn as nn

from ..base import get_layer, MLP_fac
from .field_space_base import (
    refine_zoom,
    coarsen_zoom,
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
        rank_space: Union[List[int], int, None] = None,
        rank_time: Union[List[int], int, None] = None,
        rank_depth: Union[List[int], int, None] = None,
        rank_features: Union[List[int], int, None] = None,
        seq_len_zoom: int = -1,
        seq_len_time: Union[List[int], int] = -1,
        seq_len_depth: Union[List[int], int] = -1,
        seq_overlap_space: bool = False,
        seq_overlap_time: bool = False,
        seq_overlap_depth: bool = False,
        with_var_att: bool = False,
        shift: Optional[bool] = None,
        multi_shift: bool = False,
        update: str = 'shift',
        refine_zooms: Dict[int, int] = {},
        separate_mlp_norm: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Store configuration for field-space attention.

        :param token_zoom: Token zoom level.
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
        :param shift: Whether to apply shift.
        :param multi_shift: Whether to shift at multiple zooms.
        :param update: Update mode ("shift" or "shift_scale").
        :param refine_zooms: Mapping of zoom refinements.
        :param separate_mlp_norm: Whether to separate MLP norm.
        :param kwargs: Additional keyword arguments assigned as attributes.
        :return: None.
        """
        self.token_zoom: int
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
        self.rank_time: Union[List[int], int, None]
        self.rank_depth: Union[List[int], int, None]
        self.rank_features: Union[List[int], int, None]
        self.rank_variables: Union[List[int], int, None]
        self.seq_len_zoom: int
        self.seq_len_time: Union[List[int], int]
        self.seq_len_depth: Union[List[int], int]
        self.seq_overlap_space: bool
        self.seq_overlap_time: bool
        self.seq_overlap_depth: bool
        self.with_var_att: bool
        self.shift: Optional[bool]
        self.multi_shift: bool
        self.update: str
        self.refine_zooms: Dict[int, int]
        self.separate_mlp_norm: bool

        inputs = copy.deepcopy(locals())

        for input, value in inputs.items():
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
        target_zooms: Optional[List[int]] = None,
        in_features: int = 1,
        n_groups_variables: List[int] = [1],
        token_len_depth: Union[List[int], int] = 1,
        token_len_time: Union[List[int], int] = 1,
        token_overlap_space: Union[List[bool], bool] = False,
        token_overlap_time: Union[List[bool], bool] = False,
        token_overlap_depth: Union[List[bool], bool] = False,
        token_overlap_mlp_time: Union[List[bool], bool] = False,
        token_overlap_mlp_depth: Union[List[bool], bool] = False,
        rank_variables: Union[List[int], int, None] = None,
        rank_space: Union[List[int], int, None] = None,
        rank_time: Union[List[int], int, None] = None,
        rank_depth: Union[List[int], int, None] = None,
        rank_features: Union[List[int], int, None] = None,
        seq_len_zoom: int = -1,
        seq_len_time: Union[List[int], int] = -1,
        seq_len_depth: Union[List[int], int] = -1,
        seq_overlap_space: bool = False,
        seq_overlap_time: bool = False,
        seq_overlap_depth: bool = False,
        with_var_att: bool = False,
        use_mask: bool = False,
        att_dim: Optional[int] = None,
        n_head_channels: int = 16,
        refine_zooms: Dict[int, int] = {},
        shift: bool = False,
        multi_shift: bool = False,
        dropout: float = 0,
        update: str = 'shift',
        separate_mlp_norm: bool = True,
        embed_confs: Dict[str, Any] = {},
        layer_confs: Union[List[Dict[str, Any]], Dict[str, Any]] = {},
        layer_confs_emb: Union[List[Dict[str, Any]], Dict[str, Any]] = {}
    ) -> None:
        """
        Initialize the field-space attention module.

        :param grid_layers: Mapping from zoom string to GridLayer.
        :param in_zooms: Input zoom levels.
        :param out_zooms: Output zoom levels.
        :param q_zooms: Query zoom levels or -1 to default to input zooms.
        :param kv_zooms: Key/value zoom levels or -1 to default to input zooms.
        :param token_zoom: Token zoom level.
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
        :param refine_zooms: Mapping of zoom refinements.
        :param shift: Whether to apply shift.
        :param multi_shift: Whether to shift at multiple zooms.
        :param dropout: Dropout rate.
        :param update: Update mode ("shift" or "shift_scale").
        :param separate_mlp_norm: Whether to separate MLP norm.
        :param embed_confs: Embedding configuration dictionary.
        :param layer_confs: Layer configuration for attention blocks.
        :param layer_confs_emb: Layer configuration for embedding blocks.
        :return: None.
        """
        super().__init__()
        
        # Normalize per-group configs so indexing is consistent across variable groups.
        n_groups = len(n_groups_variables)
        token_len_depth = check_value(token_len_depth, n_groups)
        token_len_time = check_value(token_len_time, n_groups)

        token_overlap_space = check_value(token_overlap_space, n_groups)
        token_overlap_time = check_value(token_overlap_time, n_groups)
        token_overlap_depth = check_value(token_overlap_depth, n_groups)
        token_overlap_mlp_time = check_value(token_overlap_mlp_time, n_groups)
        token_overlap_mlp_depth = check_value(token_overlap_mlp_depth, n_groups)

        layer_confs = check_value(layer_confs, n_groups)

        layer_confs_emb = check_value(layer_confs_emb, n_groups)

        rank_space = check_value(rank_space, n_groups)
        rank_time = check_value(rank_time, n_groups)
        rank_depth = check_value(rank_depth, n_groups)
        rank_variables = check_value(rank_variables, n_groups)
        rank_features = check_value(rank_features, n_groups)
        

        self.out_zooms: List[int] = copy.deepcopy(out_zooms)
        in_zooms = copy.deepcopy(in_zooms)
        self.use_mask: bool = use_mask

        # Default q/kv zooms to input zooms when not explicitly configured.
        if not isinstance(q_zooms, (List,ListConfig)) and (q_zooms == -1):
            q_zooms = in_zooms
        
        if not isinstance(kv_zooms,(List,ListConfig)) and (kv_zooms == -1):
            kv_zooms = in_zooms

        # Apply refinement mapping to all configured zoom lists.
        for k,zoom in enumerate(q_zooms):
            if zoom in refine_zooms.keys():
                q_zooms[k] = refine_zooms[zoom]
        
        for k,zoom in enumerate(kv_zooms):
            if zoom in refine_zooms.keys():
                kv_zooms[k] = refine_zooms[zoom]

        for k,zoom in enumerate(in_zooms):
            if zoom in refine_zooms.keys():
                in_zooms[k] = refine_zooms[zoom]

        for k, zoom in enumerate(kv_zooms):
            if zoom not in in_zooms:
                raise ValueError(f"Zoom level {zoom} at index {k} of kv_zooms not found in in_zooms")
        
        # Compute unique set of zooms participating in attention.
        self.qkv_zooms: List[int] = torch.tensor(q_zooms + kv_zooms).unique().tolist()

        seq_zoom = min((min(q_zooms + kv_zooms)), seq_len_zoom)  

        if (min(q_zooms + kv_zooms)) < token_zoom:
            raise ValueError(f"Zoom level {min(q_zooms + kv_zooms)} need to be refined. please indicate refine_zooms={refine_zooms}")

        self.blocks: nn.ModuleList = nn.ModuleList()

        for k in range(n_groups):
            
            # Each group gets its own attention block with group-specific config.
            layer_confs[k]['n_variables'] = n_groups_variables[k]
            block = FieldSpaceAttentionBlock(
                        grid_layers,
                        token_zoom,
                        seq_zoom if seq_zoom > -1 else -1,
                        q_zooms,
                        kv_zooms,
                        att_dim,
                        target_zooms = target_zooms,
                        in_features = in_features,
                        token_len_depth= token_len_depth[k],
                        token_len_time= token_len_time[k],
                        token_overlap_space= token_overlap_space[k],
                        token_overlap_time= token_overlap_time[k],
                        token_overlap_depth= token_overlap_depth[k],
                        token_overlap_mlp_time= token_overlap_mlp_time[k],
                        token_overlap_mlp_depth= token_overlap_mlp_depth[k],
                        rank_space = rank_space[k],
                        rank_time = rank_time[k],
                        rank_depth = rank_depth[k],
                        rank_features = rank_features[k],
                        rank_variables = rank_variables[k],
                        seq_len_time= seq_len_time,
                        seq_len_depth= seq_len_depth,
                        seq_overlap_space = seq_overlap_space,
                        seq_overlap_time = seq_overlap_time,
                        seq_overlap_depth = seq_overlap_depth,
                        with_var_att = with_var_att,
                        n_head_channels = n_head_channels,
                        dropout=dropout,
                        embed_confs=embed_confs,
                        layer_confs=layer_confs[k],
                        layer_confs_emb=layer_confs_emb[k],
                        update=update,
                        separate_mlp_norm=separate_mlp_norm
                        )
            self.blocks.append(block)

        self.grid_layers: Dict[str, GridLayer] = grid_layers
        self.multi_shift: bool = multi_shift
        self.token_zoom: int = token_zoom
        self.shift: bool = shift
        self.direction: str = 'east'

        # Build inverse mapping for coarse operations.
        self.refine_zooms: Dict[int, int] = refine_zooms
        self.coarse_zooms: Dict[int, int] = invert_dict(refine_zooms)

        self.block: "FieldSpaceAttentionBlock" = block


    def refine_groups(self, x_zooms_groups: List[Dict[int, torch.Tensor]]) -> List[Dict[int, torch.Tensor]]:
        """
        Refine all zooms in the provided groups.

        :param x_zooms_groups: List of zoom-to-tensor mappings with tensors shaped like
            ``(b, v, t, n, d, f)``.
        :return: Refined zoom groups with tensors shaped like ``(b, v, t, n, d, f)``.
        """
        for k, x_zooms in enumerate(x_zooms_groups):
            for in_zoom, out_zoom in self.refine_zooms.items():
                # Refine in-place to ensure downstream zooms exist.
                x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
            x_zooms_groups[k] = x_zooms

        return x_zooms_groups
        
    def coarse_groups(self, x_zooms_groups: List[Dict[int, torch.Tensor]]) -> List[Dict[int, torch.Tensor]]:
        """
        Coarsen all zooms in the provided groups.

        :param x_zooms_groups: List of zoom-to-tensor mappings with tensors shaped like
            ``(b, v, t, n, d, f)``.
        :return: Coarsened zoom groups with tensors shaped like ``(b, v, t, n, d, f)``.
        """
        for k, x_zooms in enumerate(x_zooms_groups):
            for in_zoom, out_zoom in self.coarse_zooms.items():
                # Coarsen in-place to create matching zooms for outputs.
                x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
            x_zooms_groups[k] = x_zooms
        
        return x_zooms_groups

    def shift_groups(
        self,
        x_zooms_groups: List[Dict[int, torch.Tensor]],
        sample_configs: Dict[int, Dict[str, Any]] = {},
        reverse: bool = False
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Apply a directional shift to each zoom group.

        :param x_zooms_groups: List of zoom-to-tensor mappings with tensors shaped like
            ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration per zoom.
        :param reverse: Whether to apply the reverse shift.
        :return: Shifted zoom groups with tensors shaped like ``(b, v, t, n, d, f)``.
        """
        for k, x_zooms in enumerate(x_zooms_groups):
            for zoom in self.qkv_zooms:
                # Use a shared shift grid layer or per-zoom shift based on config.
                grid_layer = self.grid_layers[str(self.token_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
                x_zooms[zoom] = grid_layer.apply_shift(x_zooms[zoom], self.direction, **sample_configs[zoom], reverse=reverse)[0]
        x_zooms_groups[k] = x_zooms

        return x_zooms_groups
    
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
        
        # Ensure all zooms required by attention are present.
        x_zooms_groups = self.refine_groups(x_zooms_groups)

        if self.shift:
            # Optional pre-attention shift for token mixing.
            x_zooms_groups = self.shift_groups(x_zooms_groups, sample_configs=sample_configs)
        
        x_ress, qs, Ks, Vs, masks, shapes, seq_lens = [], [], [], [], [], [], []
        for k, block in enumerate(self.blocks):
            # Build per-group Q/K/V tensors and tracking metadata.
            x_res, q, K, V, mask, shape = block.create_QKV(x_zooms_groups[k], emb=emb_groups[k], mask_zooms=mask_groups[k] if self.use_mask else {}, sample_configs=sample_configs)
            x_ress.append(x_res)
            qs.append(q)
            Ks.append(K)
            Vs.append(V)
            masks.append(mask)
            shapes.append(shape)
            seq_lens.append(q.shape[-2])
        
        # Concatenate across groups for a single attention call.
        q = torch.concat(qs, dim=-2)
        K = torch.concat(Ks, dim=-2)
        V = torch.concat(Vs, dim=-2)
        mask = torch.concat(masks, dim=-2) if self.use_mask else None

        # Shared attention across all groups.
        att_out = safe_scaled_dot_product_attention(q, K, V, mask=mask)

        # Split attention outputs back to per-group chunks.
        att_outs = att_out.split(seq_lens, dim=-2)

        for k, att_out in enumerate(att_outs):
            # Apply per-group MLP updates and merge into zoom tensors.
            x_zooms_groups[k] = self.blocks[k].forward_mlp(x_zooms_groups[k], x_ress[k], att_outs[k], shapes[k], emb=emb_groups[k], sample_configs=sample_configs)

        if self.shift:
            # Undo the pre-attention shift.
            x_zooms_groups = self.shift_groups(x_zooms_groups, sample_configs=sample_configs, reverse=True)
        
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
        target_zooms: Optional[List[int]] = None,
        in_features: int = 1,
        token_len_depth: int = 1,
        token_len_time: int = 1,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: bool = False,
        token_overlap_mlp_time: bool = False,
        token_overlap_mlp_depth: bool = False,
        rank_space: Optional[int] = None,
        rank_time: Optional[int] = None,
        rank_depth: Optional[int] = None,
        rank_features: Optional[int] = None,
        rank_variables: Optional[int] = None,
        dropout: float = 0.0,
        n_head_channels: int = 32,
        embed_confs: Dict[str, Any] = {},
        seq_len_time: int = -1,
        seq_len_depth: int = -1,
        seq_overlap_space: bool = False,
        seq_overlap_time: bool = False,
        seq_overlap_depth: bool = False,
        with_var_att: bool = False,
        layer_confs: Dict[str, Any] = {},
        layer_confs_emb: Dict[str, Any] = {},
        update: str = 'shift',
        layer_norm: bool = True,
        separate_mlp_norm: bool = False
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

        # Build rank configuration used by linear embedding layers.
        layer_confs_ = layer_confs.copy()
        layer_confs_['ranks'] = [rank_time,rank_space,rank_depth,rank_features,rank_features]
        layer_confs_['rank_variables'] = rank_variables

        layer_confs_emb_ = layer_confs_emb.copy()
        layer_confs_emb_['ranks'] = [rank_time,rank_space,rank_depth,rank_features,rank_features]
        layer_confs_emb_['rank_variables'] = rank_variables

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

        # Optional embedding path for conditioning.
        input_zoom_field = embed_confs.get("input_zoom", min(q_zooms))
        embedder = get_embedder(**embed_confs, grid_layers=grid_layers, zoom=input_zoom_field)

        emb_tokenizer = Tokenizer(
            input_zooms=[input_zoom_field] if embedder and embedder.has_space() else [],
            token_zoom=token_zoom,
            token_len_time=token_len_time if embedder and embedder.has_time() else 1,
            token_len_depth=token_len_depth if embedder and embedder.has_depth() else 1,
            overlap_thickness=int(embed_confs.get("token_overlap_space", False)),
            grid_layers=grid_layers
        ) 

        layer_confs_emb_field = layer_confs_emb_.copy()
        layer_confs_emb_field['ranks'] = embed_confs.get("ranks", [*layer_confs_['ranks'], None]) 

        emb_tokenizer_out_features = copy.deepcopy(self.token_size_space)
        emb_tokenizer_out_features[1] = self.token_size_space[1] if embedder and embedder.has_space() else 1

        # Embed Q with optional positional/field embedding layer.
        self.emb_layer_q_field: LinEmbLayer = LinEmbLayer(
            emb_tokenizer_out_features,
            emb_tokenizer_out_features,
            layer_confs=layer_confs_,
            identity_if_equal=True,
            embedder=embedder,
            field_tokenizer= emb_tokenizer,
            output_zoom=max(self.q_zooms),
            layer_norm=True,
            layer_confs_emb=layer_confs_emb_field
        )
        
        # Optional separate normalization for MLP path.
        if separate_mlp_norm:
            self.emb_layer_mlp: Optional[LinEmbLayer] = LinEmbLayer(
                self.token_size_space,
                self.token_size_space,
                layer_confs=layer_confs_,
                identity_if_equal=True,
                embedder=embedder,
                field_tokenizer= emb_tokenizer,
                output_zoom=max(self.q_zooms),
                layer_norm=layer_norm,
                layer_confs_emb=layer_confs_emb_,
            )
        else:
            self.emb_layer_mlp = None

        # Only build KV embedder when doing cross-attention.
        if not self.self_att:
            self.emb_layer_kv: Optional[LinEmbLayer] = LinEmbLayer(
                self.token_size_space_kv,
                self.token_size_space_kv,
                layer_confs=layer_confs_,
                identity_if_equal=True,
                embedder=embedder,
                field_tokenizer= emb_tokenizer,
                output_zoom=max(self.q_zooms),
                layer_norm=layer_norm,
                layer_confs_emb=layer_confs_emb_,
            )
        else:
            self.emb_layer_kv = None

        out_dim_q = [1, 1 , 1, att_dim] 
        out_dim_kv = [1, 1, 1, 2 * att_dim]

        update_dims = [*self.token_size_space[:-1], update_dim]
        update_dims_mlp = [*self.token_size_update[:-1], update_dim]

        # Linear projections into attention space.
        self.q_projection_layer: nn.Module = get_layer(token_size_in_overlap, out_dim_q, layer_confs=layer_confs_, bias=False)
        self.kv_projection_layer: nn.Module = get_layer(token_size_in_kv_overlap, out_dim_kv, layer_confs=layer_confs_, bias=True)
        self.out_layer_att: nn.Module = get_layer(out_dim_q, update_dims, layer_confs=layer_confs_)

        # Learned residual scaling for attention and MLP updates.
        self.gamma_res = nn.Parameter(torch.ones(self.token_size_space) * 1e-7, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(self.token_size_space) * 1e-7, requires_grad=True)

        self.mlp = MLP_fac(token_size_in_mlp_overlap, update_dims_mlp, hidden_dim=out_dim_q, dropout=dropout, layer_confs=layer_confs_, gamma=False)
        self.gamma_res_mlp = nn.Parameter(torch.ones(len(target_zooms)) * 1e-7, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(len(target_zooms)) * 1e-7, requires_grad=True)

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
        emb_cpy['TimeEmbedder'] = {max(self.q_zooms): emb_cpy['TimeEmbedder'][max(self.q_zooms)]}

        return emb_cpy
    
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

        # Project to attention feature space.
        q = self.q_projection_layer(q, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])
        kv = self.kv_projection_layer(kv, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])

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
        if self.scale_shift:
            scale, shift = self.dropout_att(att_out).chunk(2, dim=-1)
            # Apply scale/shift residual update.
            x = x_base * (1 + self.gamma_res * self.dropout_att(scale)) + self.gamma * self.dropout_att(shift)
        else:
            # Simple residual update when only shift is used.
            x = self.gamma_res * x_base + self.gamma * self.dropout_att(att_out)

        if self.separate_mlp_norm and self.emb_layer_mlp is not None:
            x = self.emb_layer_mlp(x, emb=emb_tokenized, sample_configs=sample_configs)

        # MLP update path operating on tokenized representation.
        x = self.mlp(x, emb=emb_tokenized, sample_configs=sample_configs[int(zoom_field)])

        # Split per-zoom outputs and fold them back into zoom tensors.
        x = x.split(tuple(self.n_out_features_update.values()), dim=-3)

        for k, (zoom, n) in enumerate(self.n_out_features_update.items()):
            if x_zooms and x is not None:
                x_out = rearrange(x[k], self.pattern_tokens_reverse, n=n)

                if self.scale_shift:
                    scale, shift = x_out.chunk(2, dim=-1)
                    shift = insert_matching_time_patch(x_zooms[zoom], shift, zoom, max(self.q_zooms), sample_configs)
                    scale = insert_matching_time_patch(x_zooms[zoom], scale, zoom, max(self.q_zooms), sample_configs)
                    x_zooms[zoom] = x_zooms[zoom] * (1 + self.gamma_res_mlp[k] * scale) + self.gamma_mlp[k] * shift
                else:
                    x_out = insert_matching_time_patch(x_zooms[zoom], x_out, zoom, max(self.q_zooms), sample_configs)
                    # Simple residual update at each zoom.
                    x_zooms[zoom] = (1 + self.gamma_res_mlp[k]) * x_zooms[zoom] + self.gamma_mlp[k] * x_out

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
        x_base, q, kv = self.create_QKV(x_zooms, emb=emb, sample_configs=sample_configs)
        att_out = self.attend(q, kv, mask=mask_zooms[self.grid_layer_field.zoom], sample_configs=sample_configs)
        return self.forward_mlp(x_zooms, x_base, att_out, emb=emb, sample_configs=sample_configs)

    
def invert_dict(d: Dict[int, int]) -> Dict[int, int]:
    """
    Invert a dictionary mapping.

    :param d: Mapping from key to value.
    :return: Inverted mapping from value to key.
    """
    inverted_d = {}
    for key, value in d.items():
        inverted_d[value] = key
    return inverted_d
