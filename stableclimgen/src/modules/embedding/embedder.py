import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from omegaconf import ListConfig

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
        emb = emb[self.zoom]

        if output_zoom is not None and output_zoom != self.zoom and 't' in self.keep_dims:
            t_dim =  2 - int('v' in self.keep_dims) - int('b' in self.keep_dims)

            ts_start = sample_configs[self.zoom]['n_past_ts'] - sample_configs[output_zoom]['n_past_ts']
            ts_end = sample_configs[self.zoom]['n_future_ts'] - sample_configs[output_zoom]['n_future_ts']

            nt = emb.shape[t_dim]

            emb = torch.index_select(emb, dim=t_dim, index=torch.arange(ts_start, nt - ts_end, device=emb.device))
        
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
        layer_confs: Dict[str, Any] = {},
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
        self.mlp: nn.Module = MLP_fac(self.embed_dim, self.embed_dim, mult=1, dropout=0, layer_confs=layer_confs, gamma=False)

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
        self.keep_dims: List[str] = ["b", "v", "t", "s", "c"]

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
        embs = embs.unsqueeze(dim=2)

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
        layer_confs: Dict[str, Any] = {},
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
        self.mlp: nn.Module = MLP_fac(self.embed_dim, self.embed_dim, mult=1, dropout=0, layer_confs=layer_confs, gamma=False)
    
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

    def __init__(self, name: str, in_channels: int, embed_dim: int, zoom: int, **kwargs) -> None:
        """
        Initialize the static variable embedder.

        :param name: Embedder name.
        :param in_channels: Number of input features.
        :param embed_dim: Dimensionality of the embedding output.
        :param zoom: Zoom level this embedder operates on.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(name, in_channels, embed_dim, zoom)

        self.keep_dims: List[str] = ["b", "t", "s", "c"]

        self.zoom: int = zoom

        self.embedding_fn: nn.Module = nn.Sequential(
            nn.Linear(self.in_channels, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
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
            if name not in self.shared_embedders.keys():
                self.shared_embedders[name] = embedder_class(name, in_channels, embed_dim, **kwargs)
            return self.shared_embedders[name]
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
            embeddings.append(embed_output)

        # Combine embeddings according to the mode
        if self.mode == 'concat':
            # Concatenate along the channel dimension
            embed_out = torch.cat(embeddings, dim=-1)
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
