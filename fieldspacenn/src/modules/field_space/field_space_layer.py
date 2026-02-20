from typing import Any, Dict, List, Optional, Union
import math

import string

from einops import rearrange
import torch
import torch.nn as nn

import copy
from ..base import get_layer, MLP_fac
from ...utils.helpers import check_value
from .field_space_base import Tokenizer, add_time_overlap_from_neighbor_patches, add_depth_overlap_from_neighbor_patches

_AXIS_POOL = list("g") + list(string.ascii_lowercase.replace("g","")) + list(string.ascii_uppercase)


class FieldSpaceLayerConfig:
    def __init__(
        self,
        in_zooms: List[int],
        target_zooms: List[int],
        field_zoom: int,
        out_zooms: Optional[List[int]] = None,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: bool = False,
        rank_space: Optional[int] = None,
        rank_time: Optional[int] = None,
        rank_depth: Optional[int] = None,
        in_token_len_time: int = 1,
        in_token_len_depth: int = 1,
        out_token_len_time: int = 1,
        out_token_len_depth: int = 1,
        n_groups_variables: List[int] = [1],
        mult: int = 2,
        hidden_dim: int = None,
        type: str = 'linear',
        **kwargs: Any
    ) -> None:
        """
        Store configuration for field-space layers.

        :param in_zooms: Input zoom levels.
        :param target_zooms: Target zoom levels.
        :param field_zoom: Zoom level used for tokenization.
        :param out_zooms: Optional output zoom levels.
        :param token_overlap_space: Whether to overlap space tokens.
        :param token_overlap_time: Whether to overlap time tokens.
        :param token_overlap_depth: Whether to overlap depth tokens.
        :param rank_space: Optional rank for space.
        :param rank_time: Optional rank for time.
        :param rank_depth: Optional rank for depth.
        :param in_token_len_time: Input token length along time.
        :param in_token_len_depth: Input token length along depth.
        :param out_token_len_time: Output token length along time.
        :param out_token_len_depth: Output token length along depth.
        :param n_groups_variables: Number of variable groups.
        :param mult: MLP multiplier when using non-linear type.
        :param hidden_dim: Optional explicit hidden dimension for MLP.
        :param type: Layer type ("linear" or "mlp").
        :param kwargs: Additional keyword arguments assigned as attributes.
        :return: None.
        """
        self.in_zooms: List[int]
        self.target_zooms: List[int]
        self.field_zoom: int
        self.out_zooms: Optional[List[int]]
        self.token_overlap_space: bool
        self.token_overlap_time: bool
        self.token_overlap_depth: bool
        self.rank_space: Optional[int]
        self.rank_time: Optional[int]
        self.rank_depth: Optional[int]
        self.in_token_len_time: int
        self.in_token_len_depth: int
        self.out_token_len_time: int
        self.out_token_len_depth: int
        self.n_groups_variables: List[int]
        self.mult: int
        self.hidden_dim: int
        self.type: str

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)

class FieldSpaceLayerModule(nn.Module):
    def __init__(self,
                 grid_layers: Dict[str, Any],
                 x_zooms: List[int],
                 in_zooms: List[int],
                 target_zooms: List[int],
                 field_zoom: int,
                 n_groups_variables: List[int] = [1],
                 **kwargs: Any):
        """
        Initialize a field-space layer module with per-group blocks.

        :param grid_layers: Mapping from zoom string to grid layer.
        :param x_zooms: Zoom levels present in inputs.
        :param in_zooms: Input zoom levels.
        :param target_zooms: Target zoom levels.
        :param field_zoom: Zoom level used for tokenization.
        :param n_groups_variables: Number of variable groups.
        :param kwargs: Additional keyword arguments for block construction.
        :return: None.
        """
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()
        n_groups = len(n_groups_variables)

        # Ensure layer_confs is a list for each group
        layer_confs = kwargs.get('layer_confs', {})
        if not isinstance(layer_confs, list):
            layer_confs = [copy.deepcopy(layer_confs) for _ in range(n_groups)]
        
        # Handle other kwargs that might be group-specific
        in_features = check_value(kwargs.get('in_features', 1), n_groups)
        target_features = check_value(kwargs.get('target_features', [1]), n_groups)

        for i in range(n_groups):
            block_kwargs = kwargs.copy()
            block_kwargs['layer_confs'] = layer_confs[i]
            block_kwargs['in_features'] = in_features[i]
            block_kwargs['target_features'] = target_features[i]

            block = FieldSpaceLayerBlock(
                grid_layers=grid_layers,
                x_zooms=x_zooms,
                in_zooms=in_zooms,
                target_zooms=target_zooms,
                field_zoom=field_zoom,
                **block_kwargs
            )
            self.out_zooms: Optional[List[int]] = block.out_zooms
            self.out_features: List[int] = block.out_features
            self.blocks.append(block)

    def forward(
        self,
        x_zooms_groups: List[Dict[int, torch.Tensor]],
        emb_groups: Optional[List[Optional[Dict[str, Any]]]] = None,
        sample_configs: Dict[str, Any] = {},
        **kwargs: Any
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Apply field-space blocks to each group.

        :param x_zooms_groups: List of zoom-to-tensor mappings shaped like
            ``(b, v, t, n, d, f)``.
        :param emb_groups: Optional list of embedding dictionaries per group.
        :param sample_configs: Sampling configuration dictionary.
        :param kwargs: Additional keyword arguments forwarded to blocks.
        :return: List of output zoom mappings with tensors shaped like ``(b, v, t, n, d, f)``.
        """
        if emb_groups is None:
            emb_groups = [None] * len(x_zooms_groups)

        output_groups = []
        for i, block in enumerate(self.blocks):
            output_groups.append(block(
                x_zooms=x_zooms_groups[i],
                emb=emb_groups[i],
                sample_configs=sample_configs,
                **kwargs
            ))
        return output_groups


class FieldSpaceLayerBlock(nn.Module):
  
    def __init__(
        self,
        grid_layers: Dict[str, Any],
        x_zooms: List[int],
        in_zooms: List[int],
        target_zooms: List[int],
        field_zoom: int,
        out_zooms: Optional[List[int]] = None,
        in_features: Union[List[int], int] = [1],
        target_features: Union[List[int], int] = [1],
        type: str = 'linear',
        in_token_len_time: int = 1,
        in_token_len_depth: int = 1,
        out_token_len_time: int = 1,
        out_token_len_depth: int = 1,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: bool = False,
        rank_space: Optional[int] = None,
        rank_time: Optional[int] = None,
        rank_depth: Optional[int] = None,
        mult: int = 2,
        hidden_dim: int = None,
        layer_confs: Dict[str, Any] = {}
    ) -> None:
        """
        Initialize a field-space layer block.

        :param grid_layers: Mapping from zoom string to grid layer.
        :param x_zooms: Zoom levels present in inputs.
        :param in_zooms: Input zoom levels.
        :param target_zooms: Target zoom levels.
        :param field_zoom: Zoom level used for tokenization.
        :param out_zooms: Optional output zoom levels.
        :param in_features: Input feature counts per zoom.
        :param target_features: Target feature counts per zoom.
        :param type: Layer type ("linear" or "mlp").
        :param in_token_len_time: Input token length along time.
        :param in_token_len_depth: Input token length along depth.
        :param out_token_len_time: Output token length along time.
        :param out_token_len_depth: Output token length along depth.
        :param token_overlap_space: Whether to overlap space tokens.
        :param token_overlap_time: Whether to overlap time tokens.
        :param token_overlap_depth: Whether to overlap depth tokens.
        :param rank_space: Optional rank for space.
        :param rank_time: Optional rank for time.
        :param rank_depth: Optional rank for depth.
        :param mult: MLP multiplier when using non-linear type.
        :param hidden_dim: Optional explicit hidden dimension for MLP.
        :param layer_confs: Layer configuration dictionary.
        :return: None.
        """

        super().__init__()

        if isinstance(in_features, int):
            in_features = [in_features] * len(x_zooms)
        if isinstance(target_features, int):
            target_features = [target_features] * len(target_zooms)
        self.token_overlap_space: bool = token_overlap_space
        self.token_overlap_time: bool = token_overlap_time
        self.token_overlap_depth: bool = token_overlap_depth

        self.out_zooms: Optional[List[int]] = out_zooms
        self.in_zooms: List[int] = in_zooms
        self.field_zoom: int = field_zoom
        self.n_channels_in: Dict[int, int] = {}

        self.in_features_dict: Dict[int, int] = dict(zip(x_zooms, in_features))
        self.target_features_dict: Dict[int, int] = dict(zip(target_zooms, target_features))

        self.out_features: List[int] = [
            self.target_features_dict[zoom] if zoom in self.target_features_dict.keys() else self.in_features_dict[zoom]
            for zoom in out_zooms
        ]

        self.tokenizer: Tokenizer = Tokenizer(in_zooms, 
                                   field_zoom,
                                   grid_layers=grid_layers,
                                   overlap_thickness=int(self.token_overlap_space),
                                   token_len_time=in_token_len_time,
                                   token_len_depth=in_token_len_depth)

        tokenizer_out = Tokenizer(target_zooms, 
                                  field_zoom,
                                  grid_layers=grid_layers,
                                  token_len_time=out_token_len_time,
                                  token_len_depth=out_token_len_depth)

        n_in_features_zooms, _ = self.tokenizer.get_features()
        n_out_features_zooms, _ = tokenizer_out.get_features()
        self.n_in_features_zooms: Dict[int, int] = n_in_features_zooms
        self.n_out_features_zooms: Dict[int, int] = n_out_features_zooms

        for z,f in self.n_in_features_zooms.items():
            self.n_in_features_zooms[z] = f * self.in_features_dict[z]
        
        for z,f in self.n_out_features_zooms.items():
            self.n_out_features_zooms[z] = f * self.target_features_dict[z]

        in_features_space = sum(self.n_in_features_zooms.values())
        out_features_space = sum(self.n_out_features_zooms.values())

        in_features_full = [
            in_token_len_time + 2 * int(self.token_overlap_time),
            in_features_space,
            in_token_len_depth + 2 * int(self.token_overlap_depth),
            1
        ]
        out_features_full = [out_token_len_time, out_features_space, out_token_len_depth, 1]

        layer_confs_ = layer_confs.copy()
        layer_confs_['ranks'] = [rank_time, rank_space, rank_depth, None, None]

        if type == 'linear':
            self.layer: nn.Module = get_layer(in_features_full, out_features_full, layer_confs=layer_confs_)
        else:
            self.layer = MLP_fac(in_features_full, out_features_full, mult=mult, hidden_dim=hidden_dim, layer_confs=layer_confs_)

        self.pattern_tokens_reverse: str = 'b v T N D t (n f) d 1 -> b v (T t) (N n) (D d) f'


    def update_time_embedder(self, emb: Dict[str, Any]) -> None:
        """
        Normalize time embeddings to the max input zoom.

        :param emb: Embedding dictionary containing "TimeEmbedder".
        :return: None.
        """
        for zoom in self.in_zooms:
            emb['TimeEmbedder'][zoom] = emb['TimeEmbedder'][max(self.in_zooms) if max(self.in_zooms) in emb['TimeEmbedder'].keys() else max(emb['TimeEmbedder'].keys())]

    def get_time_depth_overlaps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time/depth overlap padding to tokenized tensors.

        :param x: Input tensor of shape ``(b, v, T, N, D, t, n, d, f)``.
        :return: Tensor with overlap padding applied.
        """
        if self.token_overlap_time:
            x = add_time_overlap_from_neighbor_patches(x, overlap=1, pad_mode="edge")

        if self.token_overlap_depth:
            x = add_depth_overlap_from_neighbor_patches(x, overlap=1, pad_mode="edge")

        return x

    def forward(
        self,
        x_zooms: Dict[int, torch.Tensor],
        emb: Optional[Dict[str, Any]] = None,
        sample_configs: Dict[str, Any] = {},
        **kwargs: Any
    ) -> Dict[int, torch.Tensor]:
        """
        Apply the field-space layer to zoomed tensors.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :param kwargs: Additional keyword arguments (unused).
        :return: Updated zoom tensors shaped like ``(b, v, t, n, d, f)``.
        """
        nv = x_zooms[list(self.n_in_features_zooms.keys())[0]].shape[1]

        x = self.tokenizer(x_zooms, sample_configs=sample_configs)

        if emb:
            self.update_time_embedder(emb)

        x = self.get_time_depth_overlaps(x)

        x = self.layer(x, emb=emb, sample_configs=sample_configs[self.field_zoom])

        x = x.split(tuple(self.n_out_features_zooms.values()), dim=-3)
        
        for k, (zoom, n) in enumerate(self.n_out_features_zooms.items()):
            x_zooms[zoom] = rearrange(x[k], self.pattern_tokens_reverse, f=self.target_features_dict[zoom], v=nv)
        
        if self.out_zooms is None:
            return x_zooms
        else:
            x_zooms_out = {}
            for zoom in self.out_zooms:
                x_zooms_out[zoom] = x_zooms[zoom]
            return x_zooms_out
