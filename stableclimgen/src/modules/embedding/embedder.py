import sys
from typing import Dict
from omegaconf import ListConfig

import torch
import torch.nn as nn
from torch import ModuleDict

from .embedding_layers import SinusoidalLayer, TimeScaleLayer, RandomFourierLayer, get_mg_embeddings
from ...modules.grids.grid_layer import GridLayer,get_idx_of_patch,get_nh_idx_of_patch,RelativeCoordinateManager
from ...utils.helpers import expand_tensor

from ..base import get_layer,IdentityLayer,MLP_fac


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
    def __init__(self, name: str, in_channels: int, embed_dim: int, time_scales, time_min, time_max, **kwargs):
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
            TimeScaleLayer(in_features=self.in_channels, n_neurons=self.embed_dim, time_scales=time_scales, time_min=time_min, time_max=time_max),
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

    def __init__(self, name: str, in_channels: int, embed_dim: int, wave_length: float=1.0, wave_length_2: float=None, zoom=None, zoom_max=None, layer_confs={}, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "s", "c"]

        self.zoom = zoom
        self.zoom_max = zoom_max

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = RandomFourierLayer(in_features=self.in_channels, n_neurons=self.embed_dim, wave_length=wave_length, wave_length_2=wave_length_2)
        self.mlp = MLP_fac(self.embed_dim, self.embed_dim, mult=1, dropout=0, layer_confs=layer_confs,gamma=False) 

    def forward(self, coordinates_emb, **kwargs):
        coordinates, var_indices = coordinates_emb

        sample_configs = kwargs.get('sample_configs', {'zoom': self.zoom})
        zoom_diff = int(self.zoom_max - min(sample_configs['zoom_lvl'], self.zoom))

        coordinates= coordinates.view(coordinates.shape[0],-1, 4**zoom_diff, coordinates.shape[-1])[:,:,0]
        coord_emb = self.embedding_fn(coordinates)

        coord_emb = self.mlp(coord_emb, sample_configs=sample_configs, emb={'VariableEmbedder': var_indices})
        return coord_emb
    

class Embedding_interpolator(nn.Module):
    def __init__(self,  
                grid_layer_in:GridLayer, 
                grid_layer_out: GridLayer):
    
        super().__init__()

        self.with_nh = grid_layer_in.zoom<grid_layer_out.zoom
        self.grid_layer = grid_layer_in
        self.rcm = RelativeCoordinateManager(grid_layer_in,
                        grid_layer_out,
                        nh_in=self.with_nh,
                        ref='in' if self.with_nh else 'out',
                        precompute=True,
                        coord_system='polar')

        # could add a relative position bias here

    def forward(self, e, sample_configs={}, **kwargs):

        rel_dists = self.rcm(sample_configs=sample_configs)[0]

        if self.with_nh:
            mask = get_nh_idx_of_patch(self.grid_layer.adjc, **sample_configs, return_local=False)[1]
            rel_dists = rel_dists + (mask.unsqueeze(dim=-2) * 1e10)

       # else:
       #     rel_dists = rel_dists
        
        weights = 1/(1e-20+rel_dists)
        
        weights = weights/weights.sum(dim=-1,keepdim=True)
        
        e_inter = (e.view(*e.shape[:3],-1,1,weights.shape[-1],e.shape[-1])*weights.unsqueeze(dim=-1).unsqueeze(dim=1).unsqueeze(dim=2)).sum(dim=-2)

        return e_inter.view(*e_inter.shape[:3],-1,e_inter.shape[-1])

#class Mmbedder(BaseEmbedder):




class MGEmbedder(BaseEmbedder):


    def __init__(self, name: str, in_channels:int, embed_dim: int, grid_layers=None, zoom=None, n_variables=1, init_method='spherical_harmonics', **kwargs) -> None:

        in_channels = 2

        super().__init__(name, in_channels, embed_dim)

        mg_emb_confs = {}
        mg_emb_confs['zooms'] = [zoom]
        mg_emb_confs['features'] = [embed_dim]
        mg_emb_confs['n_variables'] = [n_variables]
        mg_emb_confs['init_methods'] = [init_method]

        self.mg_embedding = get_mg_embeddings(mg_emb_confs,
                                              grid_layers)[str(zoom)]
        self.zoom = zoom

        self.grid_layer = grid_layers[str(zoom)]
        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "v", "t", "s", "c"]

        if n_variables ==1:
            self.get_emb_fcn = self.get_embeddings
        else:
            self.get_emb_fcn = self.get_embeddings_from_var_idx
        

    def get_embeddings(self, var_indices):
        return self.mg_embedding[var_indices*0]
    
    def get_embeddings_from_var_idx(self, var_indices):
        return self.mg_embedding[var_indices]
    

    def get_patch(self, embs: torch.Tensor, sample_configs={}):
    
        idx = self.grid_layer.get_idx_of_patch(**sample_configs, return_local=False)

        idx = idx.view(idx.shape[0],1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:2], idx.shape[-2], embs.shape[-1]))

        return embs
    
    
    def forward(self, var_indices, sample_configs={},**kwargs):

        get_emb_fcn = self.get_emb_fcn

        embs = get_emb_fcn(var_indices)
        embs = self.get_patch(embs, sample_configs=sample_configs)
        embs = embs.unsqueeze(dim=2)

        return embs
    

class SharedMGEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, name: str, in_channels:int, embed_dim: int, mg_emb_confs, grid_layers=None, zoom=None, gamma=True, spatial_interpolation=True, layer_confs={}, **kwargs) -> None:

        in_channels = mg_emb_confs['features'][0]
        diff_mode = mg_emb_confs.get("diff_mode", True)

        super().__init__(name, in_channels, embed_dim)

        self.zoom = zoom

        self.grid_layers = grid_layers
        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "v", "t", "s", "c"]

        emb_zooms = list(mg_emb_confs['zooms'])

        if diff_mode:
            zooms_proc_idx = [k for k, emb_zoom in enumerate(emb_zooms) if emb_zoom <= zoom]
        else:
            zooms_proc_idx = [k for k, emb_zoom in enumerate(emb_zooms) if emb_zoom == zoom]
        
        if len(zooms_proc_idx) == 0:
            zooms_proc_idx += [int(torch.tensor(emb_zooms).argmin())]
        
        self.layers = nn.ModuleDict()
        self.get_patch_fcns = {}
        self.get_emb_fcns = {}

        self.grid_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()

        for zoom_proc_idx in zooms_proc_idx:
            zoom_proc = emb_zooms[zoom_proc_idx]
            self.grid_layers[str(zoom_proc)] = grid_layers[str(zoom_proc)]

            in_channels = mg_emb_confs['features'][zoom_proc_idx]
            n_variables = mg_emb_confs['n_variables'][zoom_proc_idx]

            if n_variables ==1:
                get_emb_fcn = self.get_embeddings
            else:
                get_emb_fcn = self.get_embeddings_from_var_idx


            if zoom == zoom_proc:
                layer = get_layer(in_channels, embed_dim, layer_confs=layer_confs)
                get_patch_fcn = self.get_patch

    #            if gamma and len(zooms_proc_idx)>0:
    #               self.gammas[str(zoom_proc)] = nn.Parameter(torch.ones(embed_dim)*1e-6, requires_grad=True)

            elif zoom > zoom_proc:
                get_patch_fcn = self.get_patch_nh

                if spatial_interpolation:
                    layer = Embedding_interpolator(grid_layers[str(zoom_proc)],
                                grid_layers[str(zoom)])
                else:
                    nh_dim = grid_layers[str(zoom_proc)].adjc.shape[-1]
                    layer = get_layer([nh_dim, *[1]*int((zoom -zoom_proc) -1) , in_channels], [*[4]*int(zoom-zoom_proc), embed_dim], layer_confs=layer_confs)

            else:
                get_patch_fcn = self.get_patch

                if spatial_interpolation:
                    layer = Embedding_interpolator(grid_layers[str(zoom_proc)],
                                grid_layers[str(zoom)])
                else:
                    layer = get_layer([*[4]*int(zoom_proc-zoom), in_channels], [*[1]*int((zoom_proc-zoom)), embed_dim], layer_confs=layer_confs)

    

            self.get_emb_fcns[zoom_proc] = get_emb_fcn
            self.get_patch_fcns[zoom_proc] = get_patch_fcn
            self.layers[str(zoom_proc)] = layer



    def get_embeddings(self, mg_emb, var_indices):
        return mg_emb[var_indices*0]
    
    def get_embeddings_from_var_idx(self, mg_emb, var_indices):
        return mg_emb[var_indices]
    

    def get_patch(self, embs: torch.Tensor, grid_layer: GridLayer, sample_configs={}):
    
        idx = grid_layer.get_idx_of_patch(**sample_configs, return_local=False)

        idx = idx.view(idx.shape[0],1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:2], idx.shape[-2], embs.shape[-1]))

        return embs
    
    def get_patch_nh(self, embs: torch.Tensor, grid_layer: GridLayer, sample_configs={}):

        idx = get_nh_idx_of_patch(grid_layer.adjc, **sample_configs, return_local=False)[0]

        idx = idx.view(idx.shape[0],1,-1,1)

        embs = torch.gather(embs, dim=-2, index=idx.expand(*embs.shape[:2], idx.shape[-2], embs.shape[-1]))
        
        return embs
    
    def forward(self, mg_emb_var, sample_configs={},**kwargs):

        embs_zooms, var_indices = mg_emb_var

        embs_out = 0
        for zoom_proc in self.get_patch_fcns.keys():
            get_emb_fcn = self.get_emb_fcns[zoom_proc]
            get_patch_fcn = self.get_patch_fcns[zoom_proc]

            embs = get_emb_fcn(embs_zooms[str(zoom_proc)], var_indices)
            embs = get_patch_fcn(embs, self.grid_layers[str(zoom_proc)], sample_configs=sample_configs)
            embs = embs.unsqueeze(dim=2)

            embs = self.layers[str(zoom_proc)](embs, sample_configs=sample_configs, emb={'VariableEmbedder': var_indices})

            embs = embs.reshape(*embs.shape[:3],-1,embs.shape[-1])
            
            embs_out = embs_out + embs

        return embs_out



class DensityEmbedder(BaseEmbedder):
    """
    A neural network module to embed longitude and latitude coordinates.

    :param embed_dim: Dimensionality of the embedding output.
    :param in_channels: Number of input coordinate features (default is 2).
    """

    def __init__(self, name: str, in_channels: int, embed_dim: int, wave_length: float=1.0, wave_length_2: float=None, zoom=None, layer_confs={}, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        # keep batch, spatial, variable and channel dimensions
        self.keep_dims = ["b", "v" ,"t", "s", "c"]

        self.zoom = zoom

        # Mesh embedder consisting of a RandomFourierLayer followed by linear and GELU activation layers
        self.embedding_fn = RandomFourierLayer(in_features=self.in_channels, n_neurons=self.embed_dim, wave_length=wave_length, wave_length_2=wave_length_2)
        self.mlp = MLP_fac(self.embed_dim, self.embed_dim, mult=1, dropout=0, layer_confs=layer_confs, gamma=False) 
    
    def forward(self, density_emb, **kwargs):
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

    def __init__(self, name: str, in_channels: int, embed_dim: int, init_value:float = None, **kwargs) -> None:
        super().__init__(name, in_channels, embed_dim)

        self.keep_dims = ["b", "v" ,"c"]

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

    def forward(self, inputs: Dict[str, torch.Tensor], sample_configs: Dict=None):
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
            embed_output = embedder(input_tensor, sample_configs=sample_configs)

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