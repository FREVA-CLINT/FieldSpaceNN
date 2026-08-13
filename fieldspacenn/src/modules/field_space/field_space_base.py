from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import string

from einops import rearrange
import torch
import torch.nn as nn

from ..base import get_layer, IdentityLayer, MLP_fac, LayerNorm
from ..grids.grid_layer import GridLayer
from ..grids.grid_utils import insert_matching_time_patch, get_matching_time_patch, decode_zooms

_AXIS_POOL = list("g") + list(string.ascii_lowercase.replace("g","")) + list(string.ascii_uppercase)
GLOBAL_EMBEDDER_CACHE_KEY = "_global_embedder_cache"
_TIME_EMBEDDING_KEYS = (
    "TimeEmbedder",
    "TimeProgressEmbedder",
    "TimeIndexEmbedder",
)


def align_time_embeddings_to_tokens(
    emb: Optional[Dict[str, Any]],
    *,
    zoom: int,
    token_len_time: int,
    field_time_steps: int,
) -> Optional[Dict[str, Any]]:
    """Select the final timestep of every temporal token for time-aware inputs."""
    if emb is None or token_len_time == 1:
        return emb
    if token_len_time < 1:
        raise ValueError(f"token_len_time must be positive, got {token_len_time}")
    if field_time_steps % token_len_time != 0:
        raise ValueError(
            f"Field time length {field_time_steps} is not divisible by "
            f"token_len_time={token_len_time} at zoom {zoom}"
        )

    aligned_emb = dict(emb)
    aligned_any = False
    for emb_key in _TIME_EMBEDDING_KEYS:
        if emb_key not in emb:
            continue

        zoom_values = emb[emb_key]
        if not isinstance(zoom_values, Mapping):
            raise ValueError(
                f"{emb_key} must map zoom levels to tensors when "
                f"token_len_time={token_len_time}"
            )

        zoom_key: Union[int, str]
        if zoom in zoom_values:
            zoom_key = zoom
        elif str(zoom) in zoom_values:
            zoom_key = str(zoom)
        else:
            raise ValueError(f"{emb_key} has no entry for active zoom {zoom}")

        values = zoom_values[zoom_key]
        if not torch.is_tensor(values) or values.ndim < 2:
            shape = None if not torch.is_tensor(values) else tuple(values.shape)
            raise ValueError(
                f"{emb_key}[{zoom}] must be a tensor with batch and time axes; "
                f"got {type(values).__name__} with shape {shape}"
            )
        if values.shape[1] != field_time_steps:
            raise ValueError(
                f"{emb_key}[{zoom}] has time length {values.shape[1]}, expected "
                f"{field_time_steps} to match the field before temporal tokenization"
            )

        aligned_zoom_values = dict(zoom_values)
        aligned_zoom_values[zoom_key] = values[
            :, token_len_time - 1 : field_time_steps : token_len_time, ...
        ].clone()
        aligned_emb[emb_key] = aligned_zoom_values
        aligned_any = True

    if aligned_any:
        aligned_emb.pop(GLOBAL_EMBEDDER_CACHE_KEY, None)
    return aligned_emb


def add_depth_overlap_from_neighbor_patches(
    x: torch.Tensor,
    overlap: int = 1,
    pad_mode: str = "zeros",  

) -> torch.Tensor:
    """
    Add depth overlap between neighboring token patches.

    :param x: Input tensor of shape ``(b, v, T, N, D, t, n, d, f)``.
    :param overlap: Number of depth tokens to overlap.
    :param pad_mode: Padding mode ("zeros" or "edge").
    :return: Tensor with depth overlap applied.
    """
  
    o = overlap
    if o == 0:
        return x

    b, v, T, N, D, t, n, d, f = x.shape
    assert o <= d, f"overlap={o} must be <= d={d}"

    out = x.new_empty(b, v, T, N, D, t, n, d + 2 * o, f)

    # center
    out[..., o:o + d, :] = x

    if D > 1:
        out[:, :, :, :, 1:, :, :, :o] = x[:, :, :, :, :-1, :, :, d - o : d]
        out[:, :, :, :, :-1, :, :, o + d :] = x[:, :, :, :, 1:, :, :, :o]

    # boundaries
    if pad_mode == "zeros":
        out[:, :, :, :, 0,  :, :, :o] = 0
        out[:, :, :, :, -1, :, :, o + d :] = 0

    elif pad_mode == "edge":
        left_edge  = x[:, :, :, :, 0,  :, :, :1].expand(b, v, T, N, t, n, o, f)
        right_edge = x[:, :, :, :, -1, :, :, -1:].expand(b, v, T, N, t, n, o, f)
        out[:, :, :, :, 0,  :, :, :o] = left_edge
        out[:, :, :, :, -1, :, :, o + d :] = right_edge

    else:
        raise ValueError("pad_mode must be 'zeros' or 'edge'")

    return out

def add_time_overlap_from_neighbor_patches(
    x: torch.Tensor,
    overlap: int = 1,
    pad_mode: str = "zeros",  

) -> torch.Tensor:
    """
    Add time overlap between neighboring token patches.

    :param x: Input tensor of shape ``(b, v, T, N, D, t, n, d, f)``.
    :param overlap: Number of time tokens to overlap.
    :param pad_mode: Padding mode ("zeros" or "edge").
    :return: Tensor with time overlap applied.
    """
  
    o = overlap
    if o == 0:
        return x

    b, v, T, N, D, t, n, d, f = x.shape
    assert o <= t, f"overlap={o} must be <= t={t}"

    out = x.new_empty(b, v, T, N, D, t + 2 * o, n, d, f)

    # center
    out[..., o:o + t,:,:,:] = x

    if T > 1:
        out[:, :, 1:, :, :, :o] = x[:, :, :-1, :, :, t - o : t]
        out[:, :, :-1, :, :, o + t :] = x[:, :, 1:, :, :, :o]

    # boundaries
    if pad_mode == "zeros":
        out[:, :, 0, :, :,  :o] = 0
        out[:, :, -1, :, :, o + t :] = 0

    elif pad_mode == "edge":
        left_edge  = x[:, :, 0, :, :,  :1].expand(b, v, N, D, o, n, d, f)
        right_edge = x[:, :, -1, :, :, -1:].expand(b, v, N, D, o, n, d, f)
        out[:, :, 0, :, :,  :o] = left_edge
        out[:, :, -1, :, :, o + t :] = right_edge

    else:
        raise ValueError("pad_mode must be 'zeros' or 'edge'")

    return out



class ConservativeLayerConfig:
    """Configure how strongly each zoom contributes its local mean to its parent."""

    def __init__(
        self,
        mean_strengths: Optional[Mapping[int, float]] = None,
    ) -> None:
        """
        Store per-child-zoom mean strengths.

        A value for zoom ``z`` scales the local mean added from ``z`` to the next
        lower configured zoom. Unspecified zooms retain the original strength of
        ``1.0``.

        :param mean_strengths: Optional mapping from child zoom to mean strength.
        :return: None.
        """
        self.mean_strengths = {
            int(zoom): float(strength)
            for zoom, strength in (mean_strengths or {}).items()
        }



class Tokenizer(nn.Module):
  
    def __init__(
        self,
        input_zooms: List[int] = [],
        token_zoom: int = -1,
        overlap_thickness: int = 0,
        grid_layers: Dict[str, GridLayer] = {},
        token_len_time: int = 1,
        token_len_depth: int = 1
    ) -> None:
        """
        Initialize a tokenizer that groups grid points into tokens.

        :param input_zooms: Input zoom levels.
        :param token_zoom: Output token zoom level.
        :param overlap_thickness: Overlap thickness for neighborhood tokens.
        :param grid_layers: Mapping from zoom string to GridLayer.
        :param token_len_time: Token length along time.
        :param token_len_depth: Token length along depth.
        :return: None.
        """
               
        super().__init__()

        if token_zoom==-1:
            overlap_thickness = 0

        self.overlap_thickness: int = overlap_thickness
        self.token_zoom: int = token_zoom
        self.input_zooms: List[int] = input_zooms

        self.grid_layers_overlap: nn.ModuleDict = nn.ModuleDict()
        self.features_zoom_w_overlap: List[int] = []
        self.features_zoom: List[int] = []
        for input_zoom in input_zooms:

            n_patch = 4**(input_zoom - self.token_zoom) if token_zoom > -1 else 12*4**(input_zoom)
            if overlap_thickness > 0:
                grid_layer = grid_layers[str(input_zoom + (overlap_thickness - 1))]
                self.grid_layers_overlap[str(input_zoom)] = grid_layer

                n_tot = grid_layer.get_number_of_points_in_patch(token_zoom)
            else:
                n_tot = n_patch

            self.features_zoom_w_overlap.append(n_tot)
            self.features_zoom.append(n_patch)

        if overlap_thickness> 0 and len(input_zooms)>0:
            self.token_fcn = self.get_token_w_overlap
        else:
            self.token_fcn = self.get_token
        
        if len(input_zooms) > 0:
            self.token_size: List[int] = [token_len_time, sum(self.features_zoom_w_overlap), token_len_depth]
        else:
            self.token_size = [token_len_time, 1, token_len_depth]

        self.pattern_tokens: str = 'b v (T t) N n (D d) f ->  b v T N D t n d f'

    def get_features(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        Return token feature sizes with and without overlap.

        :return: Tuple of (features_with_overlap, features_without_overlap).
        """
        return dict(zip(self.input_zooms, self.features_zoom_w_overlap)), dict(zip(self.input_zooms, self.features_zoom))
    
    def get_patch_features_zoom(self, input_zoom: int, overlap_thickness: int) -> int:
        """
        Compute number of patch features for a zoom with overlap.

        :param input_zoom: Input zoom level.
        :param overlap_thickness: Overlap thickness.
        :return: Number of features in the patch.
        """
        n_overlap = 4*overlap_thickness * 2**(input_zoom - self.token_zoom) + 4*overlap_thickness**2
        n_patch = 4**(input_zoom - self.token_zoom)

        return n_patch + n_overlap

    def get_token(self, x_zooms: Dict[int, torch.Tensor], sample_configs: Dict[str, Any] = {}, **kwargs: Any) -> torch.Tensor:
        """
        Tokenize inputs without overlap.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary.
        :param kwargs: Additional keyword arguments (unused).
        :return: Tokenized tensor of shape ``(b, v, T, N, D, t, n, d, f)``.
        """
        return combine_zooms(x_zooms, out_zoom=self.token_zoom, zooms=self.input_zooms, sample_configs=sample_configs)

    def get_token_w_overlap(
        self,
        x_zooms: Dict[int, torch.Tensor],
        sample_configs: Dict[str, Any] = {},
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Tokenize inputs with spatial overlap from neighbor patches.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary.
        :param mask: Optional mask tensor.
        :return: Tokenized tensor of shape ``(b, v, T, N, D, t, n, d, f)``.
        """
    
        x_out = []
        for zoom in self.input_zooms:
            x = x_zooms[zoom]
            x, mask = self.grid_layers_overlap[str(zoom)].get_nh(x, zoom, **sample_configs[zoom], mask=mask, zoom_patch_out=self.token_zoom)

            x = get_matching_time_patch(x, zoom, max(self.input_zooms), sample_configs)

            x_out.append(x)

        return torch.concat(x_out, dim=-3)
    
    def forward(
        self,
        x_zooms: Union[Dict[int, torch.Tensor], torch.Tensor],
        sample_configs: Dict[str, Any],
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert zoomed tensors into token sequences.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``,
            or a single tensor with the same shape.
        :param sample_configs: Sampling configuration dictionary.
        :param mask: Optional mask tensor.
        :return: Tokenized tensor of shape ``(b, v, T, N, D, t, n, d, f)``.
        """
        
        if self.token_size[1] > 1 or isinstance(x_zooms, Dict):
            if not isinstance(x_zooms, Dict):
                x_zooms = {self.input_zooms[0]: x_zooms}
            x = self.token_fcn(x_zooms, sample_configs=sample_configs, mask=mask)
        else:
            x = x_zooms.unsqueeze(dim=-3)

        x = rearrange(x, self.pattern_tokens, t=self.token_size[0], n=self.token_size[1], d=self.token_size[2])
        return x


class EmbLayer(nn.Module):
    def __init__(
        self,
        out_features: Union[List[int], int],
        embedder: Any,
        in_features: Optional[Union[List[int], int]] = None,
        emb_aggregation: str = "shift_scale",
        emb_ranks: Optional[List[Optional[int]]] = None,
        n_variables: int = 1,
        indexed_dims: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
        fac_mode: str = "Tucker",
        spatial_dim_count: int = 1,
        field_tokenizer: Optional[Tokenizer] = None,
        output_zoom: Optional[int] = None,
        embedder_cache_key: Optional[str] = None,
    ) -> None:
        """
        Initialize an embedding aggregation layer.

        :param out_features: Output feature sizes.
        :param embedder: Embedder instance used to generate conditioning.
        :param in_features: Optional input feature sizes.
        :param layer_confs_emb: Embedding layer configuration.
        :param spatial_dim_count: Number of spatial dimensions.
        :param field_tokenizer: Optional tokenizer for embedding inputs.
        :param output_zoom: Output zoom for embedding alignment.
        :return: None.
        """
         
        super().__init__()

        aggregation = emb_aggregation
        self.embedder = embedder
        self.field_tokenizer: Optional[Tokenizer] = field_tokenizer
        self.spatial_dim_count: int = spatial_dim_count
        self.output_zoom: Optional[int] = output_zoom
        self.embedder_cache_key: Optional[str] = embedder_cache_key

        if not isinstance(out_features, list):
            out_features_ = [out_features]
        else:
            out_features_ = out_features
        
        self.out_features: List[int] = out_features_

        if in_features is None:
            in_features = [1] * (len(out_features_) - 1)   

        self.get_emb_fcn = self.get_emb
        if field_tokenizer is not None:
            in_features = field_tokenizer.token_size
            self.get_emb_fcn = self.get_emb_and_tokenize

        if aggregation == 'shift_scale':
            ranks = emb_ranks if emb_ranks is not None else ([None] * (len(in_features) + 2))
            self.embedding_layer = get_layer(
                [*in_features, self.embedder.get_out_channels, 1],
                [*out_features_, 2],
                ranks=ranks,
                n_variables=n_variables,
                indexed_dims=indexed_dims,
                fac_mode=fac_mode,
            )
            self.forward_fcn = self.forward_w_shift_scale

        elif aggregation == 'shift_scale_gamma':
            ranks = emb_ranks if emb_ranks is not None else ([None] * (len(in_features) + 2))
            self.embedding_layer = get_layer(
                [*in_features, self.embedder.get_out_channels, 1],
                [*out_features_, 2],
                ranks=ranks,
                n_variables=n_variables,
                indexed_dims=indexed_dims,
                fac_mode=fac_mode,
            )
            self.forward_fcn = self.forward_w_shift_scale_gamma
            self.gamma_shift = nn.Parameter(torch.zeros(out_features_) * 1e-12, requires_grad=True)
            self.gamma_scale = nn.Parameter(torch.zeros(out_features_) * 1e-12, requires_grad=True)

        elif aggregation == 'shift_scale_mlp':
            ranks = emb_ranks if emb_ranks is not None else ([None] * (len(in_features) + 2))
            self.embedding_layer = MLP_fac(
                [*in_features, self.embedder.get_out_channels, 1],
                [*out_features_, 2],
                mult=1,
                ranks=ranks,
                n_variables=n_variables,
                indexed_dims=indexed_dims,
                fac_mode=fac_mode
            ) 
            self.forward_fcn = self.forward_w_shift_scale
        
        elif aggregation == 'shift_scale_mlp_gamma':
            ranks = emb_ranks if emb_ranks is not None else ([None] * (len(in_features) + 2))
            self.embedding_layer = MLP_fac(
                [*in_features, self.embedder.get_out_channels, 1],
                [*out_features_, 2],
                mult=1,
                ranks=ranks,
                n_variables=n_variables,
                indexed_dims=indexed_dims,
                fac_mode=fac_mode
            ) 
            self.forward_fcn = self.forward_w_shift_scale_gamma
            self.gamma_shift = nn.Parameter(torch.zeros(out_features_) * 1e-12, requires_grad=True)
            self.gamma_scale = nn.Parameter(torch.zeros(out_features_) * 1e-12, requires_grad=True)

        elif aggregation == 'shift':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], ranks=emb_ranks, n_variables=n_variables, indexed_dims=indexed_dims, fac_mode=fac_mode)
            self.forward_fcn = self.forward_w_shift
        
        elif aggregation == 'scale':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], ranks=emb_ranks, n_variables=n_variables, indexed_dims=indexed_dims, fac_mode=fac_mode)
            self.forward_fcn = self.forward_w_scale

        elif aggregation == 'concat':
            self.embedding_layer = get_layer([*in_features, self.embedder.get_out_channels], [*out_features_], ranks=emb_ranks, n_variables=n_variables, indexed_dims=indexed_dims, fac_mode=fac_mode)
            self.forward_fcn = self.forward_w_concat

        self.aggregation: str = aggregation
    
    def get_emb(self, emb: Dict[str, Any], sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Compute embedding for the given inputs.

        :param emb: Embedding dictionary with tensors shaped like ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary.
        :return: Embedded tensor shaped like ``(b, v, t, n, d, c)``.
        """
        cache = None if emb is None else emb.get(GLOBAL_EMBEDDER_CACHE_KEY)
        if self.embedder_cache_key is not None and isinstance(cache, Mapping) and self.embedder_cache_key in cache:
            return cache[self.embedder_cache_key]

        emb_ = self.embedder(emb, sample_configs, output_zoom=self.output_zoom)
        if self.embedder_cache_key is not None and emb is not None:
            if not isinstance(cache, dict):
                cache = {}
                emb[GLOBAL_EMBEDDER_CACHE_KEY] = cache
            cache[self.embedder_cache_key] = emb_
        return emb_
    
    def get_emb_and_tokenize(self, emb: Dict[str, Any], sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Compute embeddings and tokenize them if a field tokenizer is provided.

        :param emb: Embedding dictionary with tensors shaped like ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary.
        :return: Tokenized embedding tensor of shape ``(b, v, T, N, D, t, n, d, c)``.
        """
        emb = self.get_emb(emb, sample_configs=sample_configs)
        emb_tokenized = self.field_tokenizer(emb, sample_configs=sample_configs)
        return emb_tokenized
    
    def forward_w_shift(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Apply a shift-only embedding update.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :return: Updated tensor of shape ``(b, v, t, n, d, f)``.
        """
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = shift.shape[-1]
        shift = shift.view(*shift.shape[:3], -1, n)

        x = x + shift

        return x
    
    def forward_w_scale(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Apply a scale-only embedding update.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :return: Updated tensor of shape ``(b, v, t, n, d, f)``.
        """
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        scale = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = scale.shape[-1]
        scale = scale.view(*scale.shape[:3], -1, n)

        x = x * (1 + scale)

        return x

    def forward_w_concat(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Concatenate embedding features with input tensor.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :return: Concatenated tensor with expanded feature dimension.
        """
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        e = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb)

        n = e.shape[-1]
        e = e.view(*e.shape[:3], -1, n)

        x = torch.concat((x, e), dim=-1)

        return x
    
    def forward_w_shift_scale(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Apply scale and shift embedding update.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :return: Updated tensor of shape ``(b, v, t, n, d, f)``.
        """
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        scale, shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb).chunk(2, dim=-1)

        scale = scale.squeeze(dim=-1)
        shift = shift.squeeze(dim=-1)

        x = x * (scale + 1) + shift

        return x

    def forward_w_shift_scale_gamma(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Apply scale and shift embedding update.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :return: Updated tensor of shape ``(b, v, t, n, d, f)``.
        """
        
        emb_ = self.get_emb_fcn(emb, sample_configs)
        scale, shift = self.embedding_layer(emb_, sample_configs=sample_configs, emb=emb).chunk(2, dim=-1)

        scale = scale.squeeze(dim=-1)
        shift = shift.squeeze(dim=-1)

        x = x * (scale * self.gamma_scale + 1) + shift * self.gamma_shift

        return x

    def forward(self, x: torch.Tensor, emb: Dict[str, Any], sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Apply the configured embedding aggregation.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :return: Updated tensor of shape ``(b, v, t, n, d, f)``.
        """
        return self.forward_fcn(x, emb=emb, sample_configs=sample_configs)



class LinEmbLayer(nn.Module):
    def __init__(
        self,
        in_features: Optional[Union[List[int], int]],
        out_features: Union[List[int], int],
        layer_norm: bool = False,
        identity_if_equal: bool = False,
        ranks: Optional[List[Optional[int]]] = None,
        emb_ranks: Optional[List[Optional[int]]] = None,
        n_variables: int = 1,
        n_variable_norm: int = 1,
        indexed_dims: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
        indexed_dims_norm: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
        fac_mode: str = "Tucker",
        emb_aggregation: str = "shift_scale",
        embedder: Optional[Any] = None,
        field_tokenizer: Optional[Tokenizer] = None,
        output_zoom: Optional[int] = None,
        spatial_dim_count: int = 1,
        embedder_cache_key: Optional[str] = None,
    ) -> None:
        """
        Initialize a linear embedding layer with optional conditioning.

        :param in_features: Input feature sizes.
        :param out_features: Output feature sizes.
        :param layer_norm: Whether to apply layer normalization.
        :param identity_if_equal: Use identity when input/output sizes match.
        :param layer_confs: Layer configuration dictionary.
        :param layer_confs_emb: Embedding layer configuration dictionary.
        :param embedder: Optional embedder instance.
        :param field_tokenizer: Optional tokenizer for embedding inputs.
        :param output_zoom: Output zoom for embedding alignment.
        :param spatial_dim_count: Number of spatial dimensions.
        :return: None.
        """
         
        super().__init__()

        in_features = out_features if in_features is None else in_features

        self.embedder = embedder
        self.spatial_dim_count: int = spatial_dim_count

        if not isinstance(out_features, list):
            out_features_ = [out_features]
        else:
            out_features_ = out_features
        
        self.out_features = out_features
        
        if not isinstance(in_features, list):
            in_features_ = [in_features]
        else:
            in_features_ = in_features

        if self.embedder is not None:
            
            self.embedding_layer: nn.Module = EmbLayer(out_features, 
                                            embedder=embedder,
                                            emb_aggregation=emb_aggregation,
                                            emb_ranks=emb_ranks,
                                            n_variables=n_variables,
                                            indexed_dims=indexed_dims,
                                            fac_mode=fac_mode,
                                            spatial_dim_count=spatial_dim_count,
                                            field_tokenizer = field_tokenizer,
                                            output_zoom=output_zoom,
                                            embedder_cache_key=embedder_cache_key)
           
            concat = emb_aggregation == 'concat'

            self.out_features = self.embedding_layer.out_features + out_features if concat else out_features

        else:
            self.embedding_layer = IdentityLayer()

        if layer_norm:
            self.layer_norm = LayerNorm(
                out_features_,
                elementwise_affine=True,
                n_variables=n_variable_norm,
                indexed_dims=indexed_dims_norm,
            )
        else:
            self.layer_norm = IdentityLayer()

        if identity_if_equal and (torch.tensor(in_features_)-torch.tensor(out_features_)==0).all():
            self.layer: nn.Module = IdentityLayer()
        else:
            self.layer = get_layer(
                in_features_,
                out_features_,
                ranks=ranks,
                n_variables=n_variables,
                indexed_dims=indexed_dims,
                fac_mode=fac_mode,
            )


    def forward(
        self,
        x: torch.Tensor,
        emb: Dict[str, Any] = {},
        sample_configs: Dict[str, Any] = {},
        x_stats: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Apply linear projection, normalization, and embedding conditioning.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Sampling configuration dictionary.
        :param x_stats: Optional statistics tensor for normalization.
        :param kwargs: Additional keyword arguments (unused).
        :return: Updated tensor of shape ``(b, v, t, n, d, f)``.
        """
        
        x = self.layer(x, emb=emb, sample_configs=sample_configs)

        x = self.layer_norm(x, emb=emb, x_stats=x_stats)

        x = self.embedding_layer(x, emb=emb, sample_configs=sample_configs)

        return x
    


class DiffDecoder(nn.Module):
    def __init__(self):
        """
        Initialize a diffusion decoder wrapper.

        :return: None.
        """
        super().__init__()

    def forward(
        self,
        x_zooms: Dict[int, torch.Tensor],
        sample_configs: Dict[str, Any],
        out_zoom: Optional[int] = None,
        **kwargs: Any
    ) -> Dict[int, torch.Tensor]:
        """
        Decode zoomed tensors to a target zoom if provided.

        :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary.
        :param out_zoom: Optional output zoom level.
        :param kwargs: Additional keyword arguments (unused).
        :return: Decoded zoom tensors shaped like ``(b, v, t, n, d, f)``.
        """

        if out_zoom is None:
            return x_zooms
        
        return decode_zooms(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)


class ConservativeLayer(nn.Module):
  
    def __init__(self,
                 in_zooms: List[int],
                 first_feature_only: bool = False,
                 mean_strengths: Optional[Mapping[int, float]] = None,
                ) -> None: 
        """
        Initialize a conservative layer that preserves coarse averages.

        :param in_zooms: Input zoom levels.
        :param first_feature_only: Whether to apply conservation to first feature only.
        :param mean_strengths: Optional mapping from child zoom to the factor applied
            to its local mean before adding it to the next lower configured zoom.
            Unspecified child zooms use ``1.0``.
        :return: None.
        """
      
        super().__init__()

        self.ffo: bool = first_feature_only

        self.proj_layers: nn.ModuleDict = nn.ModuleDict()
        self.out_zooms: List[int] = [int(zoom) for zoom in in_zooms]
        
        zooms_sorted = sorted(self.out_zooms, reverse=True)
        
        self.cons_dict = dict(zip(zooms_sorted[:-1],zooms_sorted[1:]))
        self.cons_dict[zooms_sorted[-1]] = zooms_sorted[-1]

        configured_strengths = {
            int(zoom): float(strength)
            for zoom, strength in (mean_strengths or {}).items()
        }
        child_zooms = set(zooms_sorted[:-1])
        invalid_zooms = sorted(set(configured_strengths) - child_zooms)
        if invalid_zooms:
            raise ValueError(
                "mean_strengths can only contain zooms with a lower configured "
                f"parent; got {invalid_zooms} for in_zooms={self.out_zooms}."
            )
        self.mean_strengths: Dict[int, float] = {
            zoom: configured_strengths.get(zoom, 1.0)
            for zoom in child_zooms
        }

        self.in_zooms: List[int] = self.out_zooms
    

    def forward(
        self,
        x_zooms_groups: List[Dict[int, torch.Tensor]],
        sample_configs: Dict[str, Any] = {},
        **kwargs: Any
    ) -> List[Dict[int, torch.Tensor]]:
        """
        Apply conservative updates across zoom levels.

        :param x_zooms_groups: List of zoom-to-tensor mappings with tensors shaped like
            ``(b, v, t, n, d, f)``.
        :param sample_configs: Sampling configuration dictionary.
        :param kwargs: Additional keyword arguments (unused).
        :return: Updated zoom groups with tensors shaped like ``(b, v, t, n, d, f)``.
        """
        
        for k, x_zooms in enumerate(x_zooms_groups):
            for zoom in sorted(x_zooms.keys()):
                
                x = x_zooms[zoom]
                zoom_level_cons = zoom - self.cons_dict[zoom]

                if zoom_level_cons > 0:
                    x = x.view(*x.shape[:3], -1, 4**zoom_level_cons, *x.shape[-2:]) 

                    mean = x.mean(dim=-3)
                    x = (x-mean.unsqueeze(dim=-3)).view(*x.shape[:3], -1, *x.shape[-2:])

                    mean_strength = self.mean_strengths[zoom]
                    x_patch = (
                        get_matching_time_patch(
                            x_zooms[self.cons_dict[zoom]],
                            self.cons_dict[zoom],
                            zoom,
                            sample_configs,
                        )
                        + mean_strength * mean
                    )

                    x_zooms[self.cons_dict[zoom]] = insert_matching_time_patch(x_zooms[self.cons_dict[zoom]], x_patch, self.cons_dict[zoom], zoom, sample_configs)

                    x_zooms[zoom] = x
            x_zooms_groups[k] = x_zooms
        return x_zooms_groups



def combine_zooms(
    x_zooms: Dict[int, torch.Tensor],
    out_zoom: int,
    zooms: Optional[List[int]] = None,
    sample_configs: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Combine multiple zoom tensors into a single tokenized representation.

    :param x_zooms: Mapping from zoom to tensors shaped like ``(b, v, t, n, d, f)``.
    :param out_zoom: Output zoom level for tokenization.
    :param zooms: Optional subset of zoom levels.
    :param sample_configs: Optional sampling configuration dictionary.
    :return: Combined tensor shaped like ``(b, v, t, N, d, f)`` with concatenated zooms.
    """
    zooms = list(x_zooms.keys()) if zooms is None else zooms
    x_out = []
    for zoom in zooms:
        x = x_zooms[zoom]

        x = get_matching_time_patch(x, zoom, max(zooms), sample_configs)

        if zoom < out_zoom:
            x = refine_zoom(x, zoom, out_zoom).unsqueeze(dim=-3)
        elif out_zoom==-1:
            x = x.view(*x.shape[:3],1, -1,*x.shape[-2:])
        else:
            x = x.view(*x.shape[:3],-1, 4**(zoom - out_zoom),*x.shape[-2:])
        x_out.append(x)
    return torch.concat(x_out, dim=-3)



def refine_zoom(x: torch.Tensor, in_zoom: int, out_zoom: int) -> torch.Tensor:
    """
    Refine a zoom tensor to a higher resolution.

    :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
    :param in_zoom: Input zoom level.
    :param out_zoom: Output zoom level.
    :return: Refined tensor of shape ``(b, v, t, n', d, f)``.
    """
    x = x.view(*x.shape[:3],-1, 1, *x.shape[-2:])
    x = x.expand(-1,-1,-1, -1,4**(out_zoom - in_zoom),-1,-1).reshape(*x.shape[:3],-1, *x.shape[-2:])
    return x



def coarsen_zoom(x: torch.Tensor, in_zoom: int, out_zoom: int) -> torch.Tensor:
    """
    Coarsen a zoom tensor to a lower resolution.

    :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
    :param in_zoom: Input zoom level.
    :param out_zoom: Output zoom level.
    :return: Coarsened tensor of shape ``(b, v, t, n', d, f)``.
    """
    x = x.view(*x.shape[:3],-1, 4**(in_zoom - out_zoom), *x.shape[-2:]).mean(dim=-3)
    return x
