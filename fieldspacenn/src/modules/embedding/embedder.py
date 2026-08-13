import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import math
from omegaconf import ListConfig
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch import ModuleDict
from einops import rearrange

from .embedding_layers import SinusoidalLayer, TimeScaleLayer, RandomFourierLayer, get_mg_embeddings
from ...modules.grids.grid_layer import GridLayer
from ...utils.helpers import expand_tensor

from ..base import MLP_fac


class BaseEmbedder(nn.Module):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param in_channels: Number of input features.
    :param embed_dim: Dimensionality of the embedding output.
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int) -> None:
        """
        Initialize the base embedder.

        :param name: Embedder name.
        :param in_channels: Number of input features.
        :param embed_dim: Dimensionality of the embedding output.
        :return: None.
        """
        super().__init__()
        self.name: str = name
        self.in_channels: int = in_channels
        self.embed_dim: int = embed_dim
        self.embedding_fn: Optional[Callable[..., torch.Tensor]] = None

        self.keep_dims: List[str] = []

    def forward(
        self,
        emb: torch.Tensor,
        output_zoom: Optional[int] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Perform the forward pass to embed the tensor.

        :param emb: Input tensor to embed, typically shaped like
            ``(b, v, t, n, d, f)`` or a subset of those dimensions.
        :param output_zoom: Optional output zoom level.
        :return: Embedded tensor with the last dimension expanded to ``embed_dim``.
        """
        # Apply the embedder to the input tensor
        return self.embedding_fn(emb)


class ZoomBaseEmbedder(nn.Module):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param in_channels: Number of input features.
    :param embed_dim: Dimensionality of the embedding output.
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int, zoom: int) -> None:
        """
        Initialize a zoom-aware embedder.

        :param name: Embedder name.
        :param in_channels: Number of input features.
        :param embed_dim: Dimensionality of the embedding output.
        :param zoom: Zoom level this embedder operates on.
        :return: None.
        """
        super().__init__()
        self.name: str = name
        self.in_channels: int = in_channels
        self.embed_dim: int = embed_dim
        self.embedding_fn: Optional[Callable[..., torch.Tensor]] = None
        self.zoom: int = zoom

        self.keep_dims: List[str] = []

    def forward(
        self,
        emb: Dict[int, torch.Tensor],
        output_zoom: Optional[int] = None,
        sample_configs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Perform the forward pass to embed the tensor.

        :param emb: Mapping of zoom to tensors shaped like ``(b, v, t, n, d, f)``.
        :param output_zoom: Optional output zoom level.
        :param sample_configs: Optional sampling configuration dictionary.
        :return: Embedded tensor for the configured zoom with the last dimension
            expanded to ``embed_dim``.
        """
        # Apply the embedder to the input tensor
        emb = emb[output_zoom] if output_zoom is not None and output_zoom in emb else emb[self.zoom]
        
        return self.embedding_fn(emb)


class TimeEmbedder(ZoomBaseEmbedder):
    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        time_scales: Sequence[float],
        time_min: float,
        time_max: float,
        zoom: int,
        use_linear: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Time2Vec module with fixed periodic components based on user-defined time scales.

        :param name: Embedder name.
        :param in_channels: Number of input features.
        :param embed_dim: Dimensionality of the embedding output.
        :param time_scales: List of time scales (e.g., [24, 168, 720, 8760]).
        :param time_min: Minimum time value for scaling.
        :param time_max: Maximum time value for scaling.
        :param zoom: Zoom level this embedder operates on.
        :param use_linear: Whether to include a linear component.
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim, zoom)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims: List[str] = ["b", "t", "c"]

        self.zoom: int = zoom

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn: nn.Module = torch.nn.Sequential(
            TimeScaleLayer(in_features=self.in_channels, n_neurons=self.embed_dim, time_scales=time_scales, time_min=time_min, time_max=time_max, use_linear=use_linear),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )


class TimeIndexEmbedder(ZoomBaseEmbedder):
    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        zoom: int,
        init_value: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Learn a time-position embedding indexed by the time axis length.

        :param name: Embedder name.
        :param in_channels: Maximum supported number of time indices.
        :param embed_dim: Dimensionality of the embedding output.
        :param zoom: Zoom level this embedder operates on.
        :param init_value: Optional constant initialization value.
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim, zoom)

        self.keep_dims: List[str] = ["b", "t", "c"]
        self.zoom: int = zoom
        self.embedding_fn: nn.Module = nn.Embedding(self.in_channels, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)

    def forward(
        self,
        emb: Dict[int, torch.Tensor],
        output_zoom: Optional[int] = None,
        sample_configs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        emb_zoom = emb[output_zoom] if output_zoom is not None and output_zoom in emb else emb[self.zoom]

        batch_size = emb_zoom.shape[0]
        n_time = emb_zoom.shape[1]
        if n_time > self.in_channels:
            raise ValueError(
                f"`TimeIndexEmbedder` supports at most {self.in_channels} time indices, but received {n_time}."
            )

        time_indices = torch.arange(n_time, device=emb_zoom.device, dtype=torch.long).view(1, -1).expand(batch_size, -1)
        return self.embedding_fn(time_indices)


class TimeProgressEmbedder(ZoomBaseEmbedder):
    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        zoom: int,
        grid_layers: Optional[Dict[str, GridLayer]] = None,
        wave_length: float = 1.0,
        wave_length_2: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Embed raw day/year phase fractions, optionally varying per spatial token.

        :param name: Embedder name.
        :param in_channels: Number of phase channels (expected: 2).
        :param embed_dim: Dimensionality of the embedding output.
        :param zoom: Zoom level this embedder operates on.
        :param wave_length: Primary wavelength for random Fourier features.
        :param wave_length_2: Optional secondary wavelength.
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim, zoom)

        self.keep_dims: List[str] = ["b", "t", "s", "c"]
        self.zoom: int = zoom
        self.grid_layer: Optional[GridLayer] = None if grid_layers is None else grid_layers[str(zoom)]

        longitude_fraction = torch.empty(0, dtype=torch.float32)
        if self.grid_layer is not None:
            lon_rad = self.grid_layer.coordinates[..., 0].reshape(-1).to(torch.float32)
            longitude_fraction = lon_rad / (2 * torch.pi)
        self.register_buffer("longitude_fraction", longitude_fraction, persistent=False)
        self.longitude_fraction: torch.Tensor

        self.embedding_fn: nn.Module = nn.Sequential(
            RandomFourierLayer(
                in_features=self.in_channels,
                n_neurons=self.embed_dim,
                wave_length=wave_length,
                wave_length_2=wave_length_2,
            ),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def _get_patch_longitude_fraction(
        self,
        zoom: int,
        sample_configs: Optional[Dict[str, Any]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.longitude_fraction.numel() == 0:
            return None

        lon_fraction = self.longitude_fraction.to(device=device, dtype=dtype)
        if self.grid_layer is not None and sample_configs is not None and zoom in sample_configs:
            idx = self.grid_layer.get_idx_of_patch(
                **sample_configs[zoom],
                return_local=False,
            )
            lon_fraction = lon_fraction[idx]
            if lon_fraction.ndim == 1:
                lon_fraction = lon_fraction.view(1, 1, -1, 1)
            else:
                lon_fraction = lon_fraction.view(lon_fraction.shape[0], 1, lon_fraction.shape[1], 1)
        else:
            lon_fraction = lon_fraction.view(1, 1, -1, 1)

        return lon_fraction

    def forward(
        self,
        emb: Dict[int, torch.Tensor],
        output_zoom: Optional[int] = None,
        sample_configs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        zoom_key = output_zoom if output_zoom is not None and output_zoom in emb else self.zoom
        if not zoom_key in emb.keys():
            zoom_key = int((torch.tensor(list(emb.keys()),dtype=torch.int) - output_zoom).abs().argmin())
            zoom_key = list(emb.keys())[zoom_key]
        emb_zoom = emb[zoom_key]

        if emb_zoom.ndim == 3:
            emb_zoom = emb_zoom.unsqueeze(2)

        lon_fraction = self._get_patch_longitude_fraction(
            zoom=zoom_key,
            sample_configs=sample_configs,
            device=emb_zoom.device,
            dtype=emb_zoom.dtype,
        )
        if lon_fraction is not None:
            if emb_zoom.shape[2] == 1 and lon_fraction.shape[2] != 1:
                emb_zoom = emb_zoom.expand(-1, -1, lon_fraction.shape[2], -1)
            elif emb_zoom.shape[2] != lon_fraction.shape[2]:
                raise ValueError(
                    f"`TimeProgressEmbedder` expected {lon_fraction.shape[2]} spatial positions at zoom {self.zoom}, "
                    f"but received {emb_zoom.shape[2]}."
                )

            emb_zoom = emb_zoom.clone()
            emb_zoom[..., 0:1] = torch.remainder(emb_zoom[..., 0:1] + lon_fraction, 1.0)

        return self.embedding_fn(emb_zoom)


class CoordinateEmbedder(ZoomBaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        wave_length: float = 1.0,
        wave_length_2: Optional[float] = None,
        zoom: Optional[int] = None,
        zoom_max: Optional[int] = None,
        ranks: Optional[List[Optional[int]]] = None,
        n_variables: int = 1,
        fac_mode: str = "Tucker",
        **kwargs: Any
    ) -> None:
        """
        Initialize the coordinate embedder.

        :param name: Embedder name.
        :param in_channels: Number of input coordinate features.
        :param embed_dim: Dimensionality of the embedding output.
        :param wave_length: Wavelength for the random Fourier features.
        :param wave_length_2: Optional secondary wavelength.
        :param zoom: Zoom level this embedder operates on.
        :param zoom_max: Maximum zoom used for downscaling coordinates.
        :param layer_confs: Configuration for the MLP.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims: List[str] = ["b", "s", "c"]

        self.zoom: Optional[int] = zoom
        self.zoom_max: Optional[int] = zoom_max

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn: nn.Module = RandomFourierLayer(
            in_features=self.in_channels,
            n_neurons=self.embed_dim,
            wave_length=wave_length,
            wave_length_2=wave_length_2,
        )
        self.mlp = MLP_fac(self.embed_dim, self.embed_dim, mult=1, dropout=0, ranks=ranks, n_variables=n_variables, fac_mode=fac_mode, gamma=False)

    def forward(self, coordinates_emb: Tuple[torch.Tensor, torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """
        Embed coordinate tensors with optional variable conditioning.

        :param coordinates_emb: Tuple of (coordinates, variable_indices). Coordinates are
            shaped like ``(b, s, c)`` and variable indices like ``(b, v)``.
        :param kwargs: Additional keyword arguments (e.g., `sample_configs`).
        :return: Embedded coordinates of shape ``(b, s, embed_dim)``.
        """
        coordinates, var_indices = coordinates_emb

        sample_configs = kwargs.get('sample_configs', {'zoom': self.zoom})
        zoom_diff = int(self.zoom_max - min(sample_configs['zoom_lvl'], self.zoom))

        coordinates= coordinates.view(coordinates.shape[0],-1, 4**zoom_diff, coordinates.shape[-1])[:,:,0]
        coord_emb = self.embedding_fn(coordinates)
        coord_emb = self.mlp(coord_emb, sample_configs=sample_configs, emb={'VariableEmbedder': var_indices})
        return coord_emb
    

class MGEmbedder(BaseEmbedder):


    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        grid_layers: Optional[Dict[str, GridLayer]] = None,
        zoom: Optional[int] = None,
        n_variables: int = 1,
        init_method: str = 'spherical_harmonics',
        **kwargs: Any
    ) -> None:
        """
        Initialize a multigrid (MG) embedding for a specific zoom.

        :param name: Embedder name.
        :param in_channels: Number of input features (overridden to 2 internally).
        :param embed_dim: Dimensionality of the embedding output.
        :param grid_layers: Mapping of zoom strings to grid layers.
        :param zoom: Zoom level this embedder operates on.
        :param n_variables: Number of variables encoded in the embedding.
        :param init_method: Initialization method for MG embeddings.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """

        in_channels = 2

        super().__init__(name, in_channels, embed_dim)

        mg_emb_confs = {}
        mg_emb_confs['zooms'] = [zoom]
        mg_emb_confs['features'] = [embed_dim]
        mg_emb_confs['n_variables'] = [n_variables]
        mg_emb_confs['init_methods'] = [init_method]

        self.mg_embedding: Any = get_mg_embeddings(mg_emb_confs, grid_layers)[str(zoom)]
        self.zoom: Optional[int] = zoom

        self.grid_layer: GridLayer = grid_layers[str(zoom)]
        # keep batch, spatial, variable and channel dimensions
        self.keep_dims: List[str] = ["b", "v", "s", "c"]

        self.get_emb_fcn: Callable[[torch.Tensor], torch.Tensor]
        if n_variables == 1:
            self.get_emb_fcn = self.get_embeddings
        else:
            self.get_emb_fcn = self.get_embeddings_from_var_idx
        

    def get_embeddings(self, var_indices: torch.Tensor) -> torch.Tensor:
        """
        Get shared embeddings for all variables.

        :param var_indices: Variable indices tensor of shape ``(b, v)``.
        :return: Embedding tensor of shape ``(v, embed_dim)`` or broadcastable.
        """
        return self.mg_embedding[var_indices*0]
    
    def get_embeddings_from_var_idx(self, var_indices: torch.Tensor) -> torch.Tensor:
        """
        Get variable-specific embeddings.

        :param var_indices: Variable indices tensor of shape ``(b, v)``.
        :return: Embedding tensor indexed by variable id.
        """
        return self.mg_embedding[var_indices]
    

    def get_patch(self, embs: torch.Tensor, sample_configs: Dict[str, Any] = {}) -> torch.Tensor:
        """
        Extract embeddings for the spatial patch defined by the sample configuration.

        :param embs: Embedding tensor of shape ``(b, v, s, c)`` or ``(v, s, c)``.
        :param sample_configs: Sampling configuration dictionary.
        :return: Patch embeddings aligned with the sample patch.
        """
    
        idx = self.grid_layer.get_idx_of_patch(**sample_configs[self.zoom], return_local=False)

        idx = idx.view(idx.shape[0],1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:2], idx.shape[-2], embs.shape[-1]))

        return embs
    
    
    def forward(self, var_indices: torch.Tensor, sample_configs: Dict[str, Any] = {}, **kwargs: Any) -> torch.Tensor:
        """
        Embed variable indices and extract the current spatial patch.

        :param var_indices: Variable indices tensor of shape ``(b, v)``.
        :param sample_configs: Sampling configuration dictionary.
        :param kwargs: Additional keyword arguments (unused).
        :return: Patch embeddings of shape ``(b, v, t, s, c)``.
        """

        get_emb_fcn = self.get_emb_fcn

        embs = get_emb_fcn(var_indices)
        embs = self.get_patch(embs, sample_configs=sample_configs)

        return embs


class DensityEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        wave_length: float = 1.0,
        wave_length_2: Optional[float] = None,
        zoom: Optional[int] = None,
        ranks: Optional[List[Optional[int]]] = None,
        n_variables: int = 1,
        fac_mode: str = "Tucker",
        **kwargs: Any
    ) -> None:
        """
        Initialize the density embedder.

        :param name: Embedder name.
        :param in_channels: Number of input features.
        :param embed_dim: Dimensionality of the embedding output.
        :param wave_length: Wavelength for the random Fourier features.
        :param wave_length_2: Optional secondary wavelength.
        :param zoom: Zoom level this embedder operates on.
        :param layer_confs: Configuration for the MLP.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims: List[str] = ["b", "v", "t", "s", "c"]

        self.zoom: Optional[int] = zoom

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn: nn.Module = RandomFourierLayer(
            in_features=self.in_channels,
            n_neurons=self.embed_dim,
            wave_length=wave_length,
            wave_length_2=wave_length_2,
        )
        self.mlp = MLP_fac(self.embed_dim, self.embed_dim, mult=1, dropout=0, ranks=ranks, n_variables=n_variables, fac_mode=fac_mode, gamma=False)
    
    def forward(self, density_emb: Tuple[Dict[int, torch.Tensor], torch.Tensor], **kwargs: Any) -> torch.Tensor:
        """
        Embed density fields with optional variable conditioning.

        :param density_emb: Tuple of (density_by_zoom, variable_indices). Density tensors
            are shaped like ``(b, v, t, s, c)`` per zoom.
        :param kwargs: Additional keyword arguments (e.g., `sample_configs`).
        :return: Embedded density tensor of shape ``(b, v, t, s, embed_dim)``.
        """
        density, var_indices = density_emb
        sample_configs = kwargs.get('sample_configs', {})

        if self.zoom in density.keys():
            density_ = density[self.zoom]
        else:
            zooms = torch.tensor(list(density.keys()),device=list(density.values())[0].device)
            sorted_zooms, indices = torch.sort(zooms)
            is_higher = sorted_zooms > self.zoom
            if is_higher.any():
                zoom = sorted_zooms[is_higher][0].item()
            else:
         
                is_lower = sorted_zooms <= self.zoom
                if is_lower.any():
                    zoom = sorted_zooms[is_lower][-1].item()
                else:
                    raise ValueError("No suitable zoom level found.")
            
            density_ = density_[zoom].clone()
    
            density_ = density_.view(*density_.shape[:3],-1, 4**(zoom-self.zoom), density.shape[-1]).mean(dim=-2)
        
        density_emb = self.embedding_fn(density_)

        density_emb = self.mlp(density_emb, sample_configs=sample_configs, emb={'VariableEmbedder': var_indices})
        return density_emb

class VariableEmbedder(BaseEmbedder):

    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        init_value: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the variable embedder.

        :param name: Embedder name.
        :param in_channels: Number of variables.
        :param embed_dim: Dimensionality of the embedding output.
        :param init_value: Optional constant to initialize the embedding table.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims: List[str] = ["b", "v", "c"]

        self.embedding_fn: nn.Module = nn.Embedding(self.in_channels, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)



class GroupDepthEmbedder(BaseEmbedder):

    def __init__(
        self,
        name: str,
        in_channels: Optional[int],
        embed_dim: int,
        in_features: Sequence[Optional[int]],
        init_value: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize per-group depth embeddings.

        :param name: Embedder name.
        :param in_channels: Unused legacy argument kept for config compatibility.
        :param embed_dim: Dimensionality of the embedding output.
        :param in_features: Number of depth entries available for each group.
        :param init_value: Optional constant initialization value.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, 0 if in_channels is None else in_channels, embed_dim)

        self.keep_dims: List[str] = ["b", "d", "c"]
        self.in_features: List[Optional[int]] = [None if feat is None else int(feat) for feat in in_features]
        self.embedding_fn: nn.ModuleList = nn.ModuleList(
            [
                nn.Embedding(group_in_features, self.embed_dim)
                if group_in_features is not None and group_in_features > 0
                else nn.Identity()
                for group_in_features in self.in_features
            ]
        )

        if init_value is not None:
            for embedder in self.embedding_fn:
                if isinstance(embedder, nn.Embedding):
                    embedder.weight.data.fill_(init_value)

    def forward(
        self,
        emb: Tuple[Union[int, torch.Tensor], torch.Tensor],
        output_zoom: Optional[int] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Embed depth indices for a specific variable group.

        :param emb: Tuple of ``(group_id, depth_ids)`` where ``group_id`` selects the
            group-specific embedding table and ``depth_ids`` indexes the depth entries.
        :param output_zoom: Unused.
        :param kwargs: Additional keyword arguments (unused).
        :return: Depth embedding tensor of shape ``(d, embed_dim)`` or compatible.
        """
        group_id, depth_ids = emb
        if isinstance(group_id, torch.Tensor):
            if group_id.numel()>1:
                group_id = group_id[0]
            group_id = int(group_id.item())
        else:
            group_id = int(group_id)

        if group_id >= len(self.embedding_fn):
            raise IndexError(f"group_id {group_id} out of range for {len(self.embedding_fn)} group depth embedders")

        group_in_features = self.in_features[group_id]
        if group_in_features is None or group_in_features <= 0:
            return torch.zeros(*depth_ids.shape, 1, self.embed_dim, device=depth_ids.device)

        return self.embedding_fn[group_id](depth_ids)


class PressureLevelEmbedder(BaseEmbedder):

    def __init__(
        self,
        name: str,
        in_channels: Optional[int],
        embed_dim: int,
        reference_pressure_hpa: Optional[float] = 1100,
        top_pressure_hpa: Optional[float] = 50,
        use_surface_embedding: Optional[bool] = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize per-group depth embeddings.

        :param name: Embedder name.
        :param in_channels: Unused legacy argument kept for config compatibility.
        :param embed_dim: Dimensionality of the embedding output.
        :param init_value: Optional constant initialization value.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, 0 if in_channels is None else in_channels, embed_dim)

        self.keep_dims: List[str] = ["b", "d", "c"]
        self.in_features: List[Optional[int]] = [None if feat is None else int(feat) for feat in in_channels]

        self.top_pressure_hpa = top_pressure_hpa
        self.reference_pressure_hpa = reference_pressure_hpa

        self.embedding_fn: nn.ModuleList = nn.ModuleList()
        self.forward_fcns = []

        self.use_surface_embedding = use_surface_embedding 
        if any(features == 1 for features in in_channels) and use_surface_embedding:
            self.use_surface_embedding = True
            self.surface_embedding = nn.Parameter(torch.empty(1, self.embed_dim))
            nn.init.normal_(self.surface_embedding) 

        if any(features > 1 for features in in_channels):
            self.pressure_level_embedder: nn.Module = nn.Sequential(
            RandomFourierLayer(
                in_features=1,
                n_neurons=self.embed_dim,
                wave_length=1
            ),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
            
    def normalize_pressure(
        self,
        pressure_hpa: torch.Tensor,
    ) -> torch.Tensor:
        pressure_hpa = self.reference_pressure_hpa - pressure_hpa.clamp_min(1e-4)

        log_pressure_range = math.log(self.reference_pressure_hpa / self.top_pressure_hpa)
        log_reference_pressure = math.log(self.reference_pressure_hpa)

        log_height_coordinate = (
            log_reference_pressure - torch.log(pressure_hpa)
        )

        return log_height_coordinate / log_pressure_range
        
    def forward(
        self,
        emb: Tuple[Union[int, torch.Tensor], torch.Tensor],
        output_zoom: Optional[int] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Embed depth indices for a specific variable group.

        :param emb: Tuple of ``(group_id, depth_ids)`` where ``group_id`` selects the
            group-specific embedding table and ``depth_ids`` indexes the depth entries.
        :param output_zoom: Unused.
        :param kwargs: Additional keyword arguments (unused).
        :return: Depth embedding tensor of shape ``(d, embed_dim)`` or compatible.
        """
        if emb.dim() == 1 and self.use_surface_embedding:
            return self.surface_embedding.expand(emb.shape[0],-1).view(emb.shape[0],1,-1)
        
        else:
            return self.pressure_level_embedder(
                self.normalize_pressure(emb).unsqueeze(dim=-1)
                )



class StaticVariableFieldReshaper(nn.Module):

    def __init__(self) -> None:
        """
        Initialize the static variable reshaper.

        :return: None.
        """
        super().__init__()

        self.variables_as_features: str = 'b v t n f d-> b t n (f v d)'

    def forward(self, static_variables: torch.Tensor) -> torch.Tensor:
        """
        Collapse variable and depth dimensions into the feature dimension.

        :param static_variables: Tensor of shape ``(b, v, t, n, d, f)``,
            arranged internally as ``b v t n f d``.
        :return: Reshaped tensor of shape ``(b, t, n, f*v*d)``.
        """
        return rearrange(static_variables, self.variables_as_features)


class StaticVariableEmbedder(ZoomBaseEmbedder):

    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        zoom: int,
        grid_layers: Optional[Dict[str, GridLayer]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the static variable embedder.

        :param name: Embedder name.
        :param in_channels: Number of input features.
        :param embed_dim: Dimensionality of the embedding output.
        :param zoom: Zoom level this embedder operates on.
        :param grid_layers: Optional grid layers used to gather spatial neighborhoods.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim, zoom)

        self.keep_dims: List[str] = ["b", "t", "s", "c"]

        self.zoom: int = zoom
        self.grid_layers: Dict[str, GridLayer] = {} if grid_layers is None else grid_layers
        self.nh_size: int = 1
        if str(self.zoom) in self.grid_layers:
            self.nh_size = int(self.grid_layers[str(self.zoom)].adjc.shape[-1])

        self.embedding_fn: nn.Module = nn.Sequential(
            nn.Linear(self.in_channels * self.nh_size, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(
        self,
        emb: Dict[int, torch.Tensor],
        output_zoom: Optional[int] = None,
        sample_configs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Embed static variables after augmenting each cell with its local neighborhood.

        :param emb: Mapping of zoom to tensors shaped like ``(b, t, n, f)``.
        :param output_zoom: Optional output zoom level.
        :param sample_configs: Optional sampling configuration dictionary.
        :param kwargs: Additional keyword arguments (unused).
        :return: Embedded tensor of shape ``(b, t, n, embed_dim)``.
        """
        zoom = output_zoom if output_zoom is not None and output_zoom in emb else self.zoom
        if zoom not in emb:
            target_zoom = self.zoom if output_zoom is None else output_zoom
            available_zooms = list(emb.keys())
            zoom = min(available_zooms, key=lambda zoom_key: abs(int(zoom_key) - int(target_zoom)))

        emb_zoom = emb[zoom]

        if emb_zoom.ndim == 3:
            emb_zoom = emb_zoom.unsqueeze(1)

        grid_layer = self.grid_layers[str(zoom)]
        if grid_layer is None:
            return self.embedding_fn(emb_zoom)

        if sample_configs is None:
            sample_config = {}
            if emb_zoom.shape[-2] != grid_layer.adjc.shape[0]:
                raise ValueError(
                    "`StaticVariableEmbedder` requires `sample_configs` for patch-local inputs "
                    "when neighborhood features are enabled."
                )
        else:
            sample_config = sample_configs.get(zoom, {})

        emb_zoom_nh, _ = grid_layer.get_nh(emb_zoom.unsqueeze(1), **sample_config)
        emb_zoom_nh = rearrange(emb_zoom_nh, "b 1 t s nh f -> b t s (nh f)")

        return self.embedding_fn(emb_zoom_nh)


class ForcingEmbedder(ZoomBaseEmbedder):

    def __init__(
        self,
        name: str,
        in_channels: Optional[int],
        embed_dim: int,
        zoom: int,
        forcing_names: Sequence[str],
        hidden_dim: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """
        Encode named time-dependent one-dimensional forcing profiles.

        Each forcing has an independent convolutional encoder, allowing profile
        dimensions and lengths to differ. Adaptive pooling reduces every profile
        to a fixed-size vector before the forcing vectors are fused.

        :param name: Embedder name.
        :param in_channels: Unused compatibility argument.
        :param embed_dim: Dimensionality of the embedding output.
        :param zoom: Zoom level whose time window is used.
        :param forcing_names: Ordered forcing variable names.
        :param hidden_dim: Channels used by each profile encoder.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        forcing_names = list(forcing_names)
        if not forcing_names:
            raise ValueError("ForcingEmbedder requires at least one forcing name.")
        if len(set(forcing_names)) != len(forcing_names):
            raise ValueError(f"Forcing names must be unique, got {forcing_names}.")

        super().__init__(name, len(forcing_names), embed_dim, zoom)

        self.keep_dims: List[str] = ["b", "t", "c"]
        self.forcing_names: List[str] = forcing_names
        self.hidden_dim: int = hidden_dim if hidden_dim is not None else embed_dim

        self.profile_encoders: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, self.hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            for _ in self.forcing_names
        ])
        self.fusion: nn.Module = nn.Sequential(
            nn.Linear(len(self.forcing_names) * self.hidden_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(
        self,
        emb: Dict[int, Dict[str, torch.Tensor]],
        output_zoom: Optional[int] = None,
        sample_configs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Embed forcing profiles into a per-timestep conditioning tensor.

        :param emb: Zoom mapping containing named tensors shaped ``(b, t, n)``.
        :param output_zoom: Optional output zoom used to align the time window.
        :param sample_configs: Sampling configuration by zoom.
        :return: Embedded forcing tensor shaped ``(b, t, embed_dim)``.
        """
        zoom = output_zoom if output_zoom is not None and output_zoom in emb else self.zoom
        if zoom not in emb:
            raise KeyError(
                f"ForcingEmbedder zoom {zoom} is missing. Available zooms: {list(emb.keys())}."
            )

        forcing_data = emb[zoom]
        expected_names = set(self.forcing_names)
        input_names = set(forcing_data.keys())
        if input_names != expected_names:
            missing = sorted(expected_names - input_names)
            unexpected = sorted(input_names - expected_names)
            raise ValueError(
                f"Forcing inputs do not match the configured names. "
                f"Missing: {missing}; unexpected: {unexpected}."
            )

        ts_start = 0
        ts_end = 0
        if output_zoom is not None and output_zoom != zoom:
            if sample_configs is None:
                raise ValueError("sample_configs is required when aligning forcing time windows.")
            ts_start = (
                sample_configs[zoom]['n_past_ts']
                - sample_configs[output_zoom]['n_past_ts']
            )
            ts_end = (
                sample_configs[zoom]['n_future_ts']
                - sample_configs[output_zoom]['n_future_ts']
            )
            if ts_start < 0 or ts_end < 0:
                raise ValueError(
                    f"Cannot expand forcing time window from zoom {zoom} "
                    f"to zoom {output_zoom}."
                )

        encoded_forcings = []
        output_shape = None
        for forcing_name, encoder in zip(self.forcing_names, self.profile_encoders):
            forcing = forcing_data[forcing_name]
            if forcing.ndim != 3:
                raise ValueError(
                    f"Forcing '{forcing_name}' must have shape (b, t, n), "
                    f"got {tuple(forcing.shape)}."
                )

            stop = forcing.shape[1] - ts_end if ts_end > 0 else forcing.shape[1]
            if ts_start >= stop:
                raise ValueError(
                    f"Time alignment produced an empty window for forcing '{forcing_name}'."
                )
            forcing = forcing[:, ts_start:stop]

            current_shape = forcing.shape[:2]
            if output_shape is None:
                output_shape = current_shape
            elif current_shape != output_shape:
                raise ValueError(
                    f"Forcing '{forcing_name}' has batch/time shape {tuple(current_shape)}, "
                    f"expected {tuple(output_shape)}."
                )

            if not torch.is_floating_point(forcing):
                forcing = forcing.float()
            forcing = rearrange(forcing, 'b t n -> (b t) 1 n')
            encoded = encoder(forcing).squeeze(-1)
            encoded_forcings.append(encoded)

        forcing_embedding = self.fusion(torch.cat(encoded_forcings, dim=-1))
        return rearrange(
            forcing_embedding,
            '(b t) c -> b t c',
            b=output_shape[0],
            t=output_shape[1],
        )


class MaskEmbedder(BaseEmbedder):

    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        init_value: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the mask embedder.

        :param name: Embedder name.
        :param in_channels: Number of input features (unused; mask is binary).
        :param embed_dim: Dimensionality of the embedding output.
        :param init_value: Optional constant to initialize the embedding table.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims: List[str] = ["b", "v", "t", "s", "c"]

        self.embedding_fn: nn.Module = nn.Embedding(2, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)

class GridEmbedder(BaseEmbedder):

    def __init__(
        self,
        name: str,
        in_channels: int,
        embed_dim: int,
        init_value: Optional[float] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the grid index embedder.

        :param name: Embedder name.
        :param in_channels: Number of grid indices.
        :param embed_dim: Dimensionality of the embedding output.
        :param init_value: Optional constant to initialize the embedding table.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims: List[str] = ["v", "c"]

        self.embedding_fn: nn.Module = nn.Embedding(self.in_channels, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)


class DiffusionStepEmbedder(BaseEmbedder):
    """
    A neural network module that encodes diffusion steps.

    This class takes as input a sequence of diffusion steps, applies sinusoidal embeddings,
    and then processes these embeddings through a simple feedforward network.
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int, **kwargs: Any) -> None:
        """
        Initializes the DiffusionStepEmbedder module.

        :param name: Embedder name.
        :param in_channels: Number of input channels for the embedding.
        :param embed_dim: Number of output channels for the final embedding.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim)
        # keep batch and channel dimensions
        self.keep_dims: List[str] = ["b", "t", "c"]

        # Define a feedforward network with SiLU activation
        self.embedding_fn: nn.Module = nn.Sequential(
            SinusoidalLayer(in_channels),
            nn.Linear(self.in_channels, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )


# Embedder manager to handle shared or non-shared instances
class EmbedderManager:
    _instance: Optional["EmbedderManager"] = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "EmbedderManager":
        if not cls._instance:
            cls._instance = super(EmbedderManager, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the singleton embedder manager.

        :return: None.
        """
        if not self._initialized:
            self.shared_embedders: Dict[str, BaseEmbedder] = {}
            self._initialized = True

    @staticmethod
    def _normalize_cache_value(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, Mapping):
            return tuple(
                sorted((str(key), EmbedderManager._normalize_cache_value(val)) for key, val in value.items())
            )
        if isinstance(value, (list, tuple, ListConfig)):
            return tuple(EmbedderManager._normalize_cache_value(item) for item in value)
        if isinstance(value, nn.Module):
            return ("module", value.__class__.__name__, id(value))
        return repr(value)

    def _get_shared_cache_key(
        self,
        name: str,
        in_channels: Optional[int],
        embed_dim: Optional[int],
        kwargs: Dict[str, Any],
    ) -> str:
        normalized_kwargs = self._normalize_cache_value(kwargs)
        return repr((name, in_channels, embed_dim, normalized_kwargs))

    def get_embedder(
        self,
        name: str,
        in_channels: Optional[int] = None,
        embed_dim: Optional[int] = None,
        shared: bool = True,
        **kwargs: Any
    ) -> BaseEmbedder:
        """
        Retrieve an embedder instance, optionally sharing it across calls.

        :param name: Embedder class name.
        :param in_channels: Number of input channels.
        :param embed_dim: Embedding dimensionality.
        :param shared: Whether to reuse a shared instance.
        :param kwargs: Additional keyword arguments forwarded to the embedder.
        :return: Embedder instance.
        """
        current_module = sys.modules[__name__]

        # Use getattr to get the class from the current module
        embedder_class = getattr(current_module, name)
        if shared:
            cache_key = self._get_shared_cache_key(name, in_channels, embed_dim, kwargs)
            if cache_key not in self.shared_embedders.keys():
                self.shared_embedders[cache_key] = embedder_class(name, in_channels, embed_dim, **kwargs)
            return self.shared_embedders[cache_key]
        else:
            # Create a new instance each time
            return embedder_class(name, in_channels, embed_dim, **kwargs)


class EmbedderSequential(nn.Module):
    def __init__(self, embedders: ModuleDict, mode: str = 'sum', spatial_dim_count: int = 2) -> None:
        """
        Initialize a sequential embedder combiner.

        :param embedders: Mapping of embedder name to embedder instance.
        :param mode: Combination mode ("average", "sum", or "concat").
        :param spatial_dim_count: Number of spatial dimensions represented by "s".
        :return: None.
        """
        super(EmbedderSequential, self).__init__()
        self.embedders: ModuleDict = embedders
        assert mode in ['average', 'sum', 'concat'], "Mode must be 'average', 'sum', or 'concat'."
        self.mode: str = mode
        self.spatial_dim_count: int = spatial_dim_count
        self.activation: nn.Module = nn.Identity()

    @staticmethod
    def _get_variable_embedder_size(inputs: Dict[str, Any]) -> Optional[int]:
        """
        Infer the runtime variable axis size from the variable embedder input.

        :param inputs: Embedding input mapping passed to the sequential embedder.
        :return: Number of runtime variables if available, otherwise None.
        """
        var_input = inputs.get("VariableEmbedder", inputs.get("variables_sampled"))
        if not isinstance(var_input, torch.Tensor):
            return None
        if var_input.ndim == 0:
            return 1
        if var_input.ndim == 1:
            return int(var_input.shape[0])
        return int(var_input.shape[1])

    def get_embedding_dims(self) -> List[str]:
        """
        Collect all dimension labels used by the active embedders.

        :return: List of dimension labels.
        """
        dims = []
        for embedder in self.embedders.values():
            dims = dims + [dim for dim in embedder.keep_dims]
        return dims
    
    def has_time(self) -> bool:
        """
        Check whether any embedder outputs a time dimension.

        :return: True if "t" appears in the embedding dims.
        """
        return 't' in self.get_embedding_dims()

    def has_space(self) -> bool:
        """
        Check whether any embedder outputs a spatial dimension.

        :return: True if "s" appears in the embedding dims.
        """
        return 's' in self.get_embedding_dims()
    
    def has_depth(self) -> bool:
        """
        Check whether any embedder outputs a depth dimension.

        :return: True if "d" appears in the embedding dims.
        """
        return 'd' in self.get_embedding_dims()
    
    def has_var(self) -> bool:
        """
        Check whether any embedder outputs a variable dimension.

        :return: True if "v" appears in the embedding dims.
        """
        return 'v' in self.get_embedding_dims()
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        sample_configs: Optional[Dict[str, Any]] = None,
        output_zoom: Optional[int] = None
    ) -> torch.Tensor:
        """
        Combine embeddings from each embedder according to the selected mode.

        :param inputs: Mapping from embedder name to input tensor, typically shaped like
            ``(b, v, t, n, d, f)`` or a subset based on the embedder's `keep_dims`.
        :param sample_configs: Optional sampling configuration dictionary.
        :param output_zoom: Optional output zoom level.
        :return: Combined embedding tensor with shape ``(b, v, t, s, c)``.
        """
        embeddings = []
        variable_embedder_size = self._get_variable_embedder_size(inputs)

        # Apply each embedder to its respective input
        for embedder_name, embedder in self.embedders.items():
            # Get the input tensor for the current embedder
            if embedder_name not in inputs:
                raise ValueError(f"Input for embedder '{embedder_name}' is missing.")

            input_tensor = inputs[embedder_name]
                    
            embed_output = embedder(input_tensor, sample_configs=sample_configs, output_zoom=output_zoom)     

            # Add time dimension
            if embed_output.ndim != len(embedder.keep_dims) + ((self.spatial_dim_count - 1) if "s" in embedder.keep_dims else 0):
                embed_output = embed_output.unsqueeze(1)


            # Reshape the output to the target output_shape
            embed_output = expand_tensor(embed_output, dims=4 + self.spatial_dim_count, keep_dims=embedder.keep_dims)
            if variable_embedder_size is not None and embed_output.shape[1] == 1 and variable_embedder_size > 1:
                embed_output = embed_output.expand(
                    embed_output.shape[0],
                    variable_embedder_size,
                    *embed_output.shape[2:],
                )
            embeddings.append(embed_output)

        # Combine embeddings according to the mode
        if self.mode == 'concat':
            # Concatenation does not apply PyTorch's implicit broadcasting. Expand
            # singleton variable/time/space/depth axes to the common shape first,
            # matching the broadcasting semantics used by sum/average modes.
            target_shape = [
                max(embedding.shape[dim] for embedding in embeddings)
                for dim in range(embeddings[0].ndim - 1)
            ]
            embeddings_broadcast = []
            for embedding in embeddings:
                incompatible_dims = [
                    dim
                    for dim, target_size in enumerate(target_shape)
                    if embedding.shape[dim] not in (1, target_size)
                ]
                if incompatible_dims:
                    raise ValueError(
                        "Cannot concatenate embeddings with incompatible shapes: "
                        f"{[tuple(value.shape) for value in embeddings]}."
                    )
                embeddings_broadcast.append(
                    embedding.expand(*target_shape, embedding.shape[-1])
                )

            embed_out = torch.cat(embeddings_broadcast, dim=-1)
        elif self.mode == 'sum':
            # Sum the embeddings
            emb_sum = embeddings[0]
            for emb in embeddings[1:]:
                emb_sum = emb_sum + emb
            embed_out = emb_sum
        elif self.mode == 'average':
            # Sum the embeddings
            emb_sum = embeddings[0]
            for emb in embeddings[1:]:
                emb_sum = emb_sum + emb
            embed_out = emb_sum / (len(embeddings) + 1)
        return self.activation(embed_out)

    @property
    def get_out_channels(self) -> Union[int, List[int]]:
        """
        Compute output channel size for the composed embedder.

        :return: Total channels for "concat" or the last embedder's channels.
        """
        if self.mode == "concat":
            return sum([emb.embed_dim for _, emb in self.embedders.items()])
        else:
            return [emb.embed_dim for _, emb in self.embedders.items()][-1]
        

def get_embedder_from_dict(dict_: Dict[str, Any]) -> Any:
    """
    Build an embedder or embedder list from a configuration dictionary.

    :param dict_: Configuration dictionary.
    :return: Embedder instance(s) or None.
    """
    if "embedder_names" in dict_.keys() and "embed_confs" in dict_.keys():
        embed_mode = dict_.get("mode","sum")
        return get_embedder(dict_["embed_names"],
                            dict_["embed_confs"],
                            embed_mode)
    else:
        return None


def get_embedder(
    embed_names: Sequence[Union[str, Sequence[str]]] = [],
    embed_confs: Dict[str, Any] = {},
    embed_mode: str = 'sum',
    **kwargs: Any
) -> Any:
    """
    Construct an embedder (or list of embedders) from names and configs.

    :param embed_names: Embedder name(s) or list of embedder name groups.
    :param embed_confs: Mapping from embedder name to constructor kwargs.
    :param embed_mode: Combination mode ("average", "sum", or "concat").
    :param kwargs: Extra keyword arguments forwarded to each embedder.
    :return: Embedder instance(s) or None.
    """
    
    if len(embed_names) >0:
        
        if not isinstance(embed_names[0], list) and not isinstance(embed_names[0], ListConfig):
            embed_names = [embed_names]
            return_list = False
        else:
            return_list = True

        embed_confs.update(**kwargs)

        embedders = []
        for embed_names_ in embed_names:
            emb_dict = nn.ModuleDict()
            for embed_name in embed_names_:
                emb = EmbedderManager().get_embedder(embed_name, **embed_confs[embed_name], **kwargs)
                emb_dict[emb.name] = emb
            
            embedders.append(EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = 1))

        if return_list:
            return embedders
        else:
            return embedders[0]

    else: 
        return None
