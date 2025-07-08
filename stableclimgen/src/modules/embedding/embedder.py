import sys
from typing import Dict
from omegaconf import ListConfig

import torch
import torch.nn as nn
from torch import ModuleDict

from .embedding_layers import RandomFourierLayer, SinusoidalLayer, TimeScaleLayer
from ...modules.grids.grid_layer import GridLayer,get_idx_of_patch,get_nh_idx_of_patch
from ...utils.helpers import expand_tensor
from ..base import get_layer,IdentityLayer


class BaseEmbedder(nn.Module):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param in_channels: Number of input features.
    :param embed_dim: Dimensionality of the embedding output.
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.embedding_fn = None

        self.keep_dims = []

    def forward(self, emb: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform the forward pass to embed the tensor.

        :param emb: Input tensor containing values to be embedded.
        :return: Embedded tensor.
        """
        # Apply the embedder to the input tensor
        return self.embedding_fn(emb)


class TimeEmbedder(BaseEmbedder):
    def __init__(self, name: str, in_channels: int, embed_dim: int, time_scales, **kwargs):
        """
        Time2Vec module with fixed periodic components based on user-defined time scales.
        :param out_features: Number of output features (embedding dimension).
        :param time_scales: List of time scales (e.g., [24, 168, 720, 8760] for hourly, weekly, monthly, yearly).
        """
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "t", "c"]

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = torch.nn.Sequential(
            TimeScaleLayer(in_features=self.in_channels, n_neurons=self.embed_dim, time_scales=time_scales),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )


class CoordinateEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int, wave_length: float=1.0, wave_length_2: float=None, zoom=None, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "s", "c"]

        self.zoom = zoom

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = torch.nn.Sequential(
            RandomFourierLayer(in_features=self.in_channels, n_neurons=self.embed_dim, wave_length=wave_length, wave_length_2=wave_length_2),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, coordinates, **kwargs):
        sample_dict = kwargs.get('sample_dict', {'zoom': self.zoom})
        zoom_diff = int(sample_dict['zoom'][0] - self.zoom)

        coordinates= coordinates.view(coordinates.shape[0],-1, 4**zoom_diff, coordinates.shape[-1])[:,:,0]
        return self.embedding_fn(coordinates)
    

class MGEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, name: str, in_channels:int, embed_dim: int, mg_emb_confs, grid_layer=None, zoom=None, layer_confs={}, **kwargs) -> None:

        in_channels = mg_emb_confs['features']

        super().__init__(name, in_channels, embed_dim)

        n_vars_total = mg_emb_confs.get("n_vars_total")
        if n_vars_total ==1:
            self.get_emb_fcn = self.get_embeddings
        else:
            self.get_emb_fcn = self.get_embeddings_from_var_idx

        self.grid_layer = grid_layer
        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "v", "t", "s", "c"]

        emb_zoom = mg_emb_confs['zoom']
   
        if zoom == emb_zoom:
            layer = get_layer(in_channels, embed_dim, layer_confs=layer_confs) if in_channels != embed_dim else IdentityLayer()
            self.get_patch_fcn = self.get_patch

        elif zoom > emb_zoom:
            nh_dim = grid_layer.adjc.shape[-1]
            layer = get_layer([nh_dim, in_channels], [4**(zoom - emb_zoom), embed_dim], layer_confs=layer_confs)
            self.get_patch_fcn = self.get_patch_nh

        else:
            layer = get_layer([4**(emb_zoom - zoom), in_channels], [embed_dim], layer_confs=layer_confs)
            self.get_patch_fcn = self.get_patch

        self.layer = layer


    def get_embeddings(self, mg_emb, var_indices):
        return mg_emb[var_indices*0]
    
    def get_embeddings_from_var_idx(self, mg_emb, var_indices):
        return mg_emb[var_indices]
    

    def get_patch(self, embs, sample_dict={}):
    
        if 'patch_index' in sample_dict:
            idx = get_idx_of_patch(self.grid_layer.adjc, **sample_dict, return_local=False)
        else:
            idx = self.grid_layer.adjc[:,[0]].unsqueeze(dim=0)

        idx = idx.view(idx.shape[0],1,1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:3], idx.shape[-2], embs.shape[-1]))

        return embs
    
    def get_patch_nh(self, embs, sample_dict={}):

        if 'patch_index' in sample_dict:
            idx, mask = get_nh_idx_of_patch(self.grid_layer.adjc, **sample_dict, return_local=False)
        else:
            idx = self.grid_layer.adjc
            mask = torch.zeros_like(idx, device=idx.device, dtype=int)

        idx = idx.view(idx.shape[0],1,1,-1,1)
        mask = mask.view(idx.shape[0],1,1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:3], idx.shape[-2], embs.shape[-1]))
       
        
        return embs
    
    def forward(self, mg_emb_var, sample_dict={},**kwargs):

        embs, var_indices = mg_emb_var
        embs = self.get_emb_fcn(embs, var_indices)

        embs = self.get_patch_fcn(embs, sample_dict=sample_dict)

        embs = self.layer(embs, sample_dict=sample_dict, emb={'VariableEmbedder': var_indices})

        embs = embs.view(*embs.shape[:3],-1,embs.shape[-1])

        return embs

class DensityEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int, wave_length: float=1.0, wave_length_2: float=None, zoom=None, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "v" ,"t", "s", "c"]

        self.zoom = zoom

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = torch.nn.Sequential(
            RandomFourierLayer(in_features=self.in_channels, n_neurons=self.embed_dim, wave_length=wave_length, wave_length_2=wave_length_2),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )
    
    def forward(self, density, **kwargs):
        sample_dict = kwargs.get('sample_dict', {'zoom': self.zoom})
        zoom_diff = int(sample_dict['zoom'][0] - self.zoom)
                    
        density = density.view(*density.shape[:3],-1, 4**zoom_diff, density.shape[-1]).mean(dim=-2)
        return self.embedding_fn(density)


class UncertaintyEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int, in_channels_var:int, wave_length: float=1.0, wave_length_2: float=None, zoom=None, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "v" ,"t", "s", "c"]

        self.zoom = zoom

        self.betas = nn.Parameter(torch.ones(in_channels_var), requires_grad=True)

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = torch.nn.Sequential(
            RandomFourierLayer(in_features=self.in_channels, n_neurons=self.embed_dim, wave_length=wave_length, wave_length_2=wave_length_2),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim),
        )
    
    def forward(self, density_var, **kwargs):
        sample_dict = kwargs.get('sample_dict', {'zoom': self.zoom})
        zoom_diff = int(sample_dict['zoom'][0] - self.zoom)

        density, var_indices = density_var
        
        density = density.view(*density.shape[:3],-1, 4**zoom_diff, density.shape[-1]).mean(dim=-2)

        betas = self.betas[var_indices].view(*density.shape[:3],1,1)

        uncertainty =  1 - density** betas

        return self.embedding_fn(uncertainty)
    
class VariableEmbedder(BaseEmbedder):

    def __init__(self, name: str, in_channels: int, embed_dim: int, init_value:float = None, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims = ["b", "v" ,"t", "c"]

        self.embedding_fn = nn.Embedding(self.in_channels, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)

class MaskEmbedder(BaseEmbedder):

    def __init__(self, name: str,  in_channels: int, embed_dim: int, init_value:float = None, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims = ["b", "v", "t", "s", "c"]

        self.embedding_fn = nn.Embedding(2, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)

class GridEmbedder(BaseEmbedder):

    def __init__(self, name: str, in_channels: int, embed_dim: int, init_value:float = None, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims = ["v", "c"]

        self.embedding_fn = nn.Embedding(self.in_channels, self.embed_dim)

        if init_value is not None:
            self.embedding_fn.weight.data.fill_(init_value)


class DiffusionStepEmbedder(BaseEmbedder):
    """
    A neural network module that encodes diffusion steps.

    This class takes as input a sequence of diffusion steps, applies sinusoidal embeddings,
    and then processes these embeddings through a simple feedforward network.
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int, **kwargs):
        """
        Initializes the DiffusionStepEmbedder module.

        :param in_channels: Number of input channels for the embedding.
        :param embed_dim: Number of output channels for the final embedding.
        """
        super().__init__(name, in_channels, embed_dim)
        # keep batch and channel dimensions
        self.keep_dims = ["b", "t", "c"]

        # Define a feedforward network with SiLU activation
        self.embedding_fn = nn.Sequential(
            SinusoidalLayer(in_channels),
            nn.Linear(self.in_channels, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )


# Embedder manager to handle shared or non-shared instances
class EmbedderManager:
    _instance = None
    _initialized = False
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(EmbedderManager, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.shared_embedders = {}
            self._initialized = True

    def get_embedder(self, name, in_channels=None, embed_dim=None, shared=True, **kwargs):
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
    def __init__(self, embedders: ModuleDict, mode='sum', spatial_dim_count = 2):
        """
        Args:
            embedders (dict): A dictionary of embedders. Keys are names, values are instances of embedders.
            mode (str): Combination mode. Can be 'average', 'sum', or 'concat'.
        """
        super(EmbedderSequential, self).__init__()
        self.embedders = embedders
        assert mode in ['average', 'sum', 'concat'], "Mode must be 'average', 'sum', or 'concat'."
        self.mode = mode
        self.spatial_dim_count = spatial_dim_count
        self.activation = nn.Identity()

    def forward(self, inputs: Dict[str, torch.Tensor], sample_dict: Dict=None):
        """
        Args:
            inputs (dict): A dictionary of input tensors where each key corresponds to an embedder.

        Returns:
            torch.Tensor: The combined embedding tensor.
        """
        embeddings = []

        # Apply each embedder to its respective input
        for embedder_name, embedder in self.embedders.items():
            # Get the input tensor for the current embedder
            if embedder_name not in inputs:
                raise ValueError(f"Input for embedder '{embedder_name}' is missing.")

            input_tensor = inputs[embedder_name]
            embed_output = embedder(input_tensor, sample_dict=sample_dict)

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
    def get_out_channels(self):
        if self.mode == "concat":
            return sum([emb.embed_dim for _, emb in self.embedders.items()])
        else:
            return [emb.embed_dim for _, emb in self.embedders.items()][-1]
        

def get_embedder_from_dict(dict_: dict):
    if "embedder_names" in dict_.keys() and "embed_confs" in dict_.keys():
        embed_mode = dict_.get("mode","sum")
        return get_embedder(dict_["embed_names"],
                            dict_["embed_confs"],
                            embed_mode)
    else:
        return None


def get_embedder(embed_names: list=[], 
                 embed_confs: dict={}, 
                 embed_mode: str='sum',
                 **kwargs):
    
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
                emb: BaseEmbedder = EmbedderManager().get_embedder(embed_name, **embed_confs[embed_name], **kwargs)
                emb_dict[emb.name] = emb     
            
            embedders.append(EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = 1))

        if return_list:
            return embedders
        else:
            return embedders[0]

    else: 
        return None