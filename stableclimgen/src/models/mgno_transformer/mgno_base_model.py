import torch
import torch.nn as nn

from ...modules.icon_grids.grid_layer import GridLayer, MultiRelativeCoordinateManager
from ...modules.neural_operator.no_blocks import DenseLayer, get_lin_layer
from ...modules.neural_operator.no_helpers import add_coordinates_to_emb_dict, add_mask_to_emb_dict
from ...modules.neural_operator.no_helpers import get_embedder


class MGNO_base_model(nn.Module):
    def __init__(self, 
                 mgrids,
                 rotate_coord_system=True,
                 ) -> None:
        
                
        super().__init__()

        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[0]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        # Create grid layers for each unique global level
        global_levels = []
        self.grid_layers = nn.ModuleDict()
        for global_level, mgrid in enumerate(mgrids):
            self.grid_layers[str(int(global_level))] = GridLayer(global_level, mgrid['adjc_lvl'], mgrid['adjc_mask'], mgrid['coords'], coord_system='polar')
            global_levels.append(global_level)

        self.register_buffer('global_levels', torch.tensor(global_levels), persistent=False)

        self.grid_layer_0 = self.grid_layers["0"]
        # Construct blocks based on configurations

        self.rcm = MultiRelativeCoordinateManager(self.grid_layers,
                                                  rotate_coord_system=rotate_coord_system
                                                )
        

    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):
        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = global_indices_sampled.view(global_indices_sampled.shape[0], -1, 4**global_level)[:,:,0]   
        
        return global_indices_sampled // 4**global_level


class InputLayer(nn.Module):

    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 grid_layer_0,
                 embedder_names=None,
                 embed_confs=None,
                 embed_mode='sum',
                 n_vars_total=1,
                 rank_vars=4,
                 factorize_vars=False,
                 with_gamma=False
                 ) -> None:

        super().__init__()

        if embedder_names is not None:
            if 'CoordinateEmbedder' in embedder_names:
                self.grid_layer_0 = grid_layer_0

            self.embedder = get_embedder(embedder_names, embed_confs, embed_mode=embed_mode)

            emb_dim = self.embedder.get_out_channels if self.embedder is not None else None

            self.embedding_layer = nn.Linear(emb_dim, model_dim_out * 2)

            if with_gamma:
                self.gamma1 = nn.Parameter(torch.ones(model_dim_out) * 1e-6, requires_grad=True)
                self.gamma2 = nn.Parameter(torch.ones(model_dim_out) * 1e-6, requires_grad=True)

        self.linear = nn.Linear(model_dim_in, model_dim_out, bias=False)

        self.linear = get_lin_layer(model_dim_in, model_dim_out, n_vars_total=n_vars_total, rank_vars=rank_vars,
                                    factorize_vars=factorize_vars, bias=False)

    def forward(self, x, mask=None, emb=None, indices_sample=None):

        if hasattr(self, 'grid_layer_0') and hasattr(self, "embedding_layer"):
            emb = add_coordinates_to_emb_dict(self.grid_layer_0, indices_layers=indices_sample[
                "indices_layers"] if indices_sample else None, emb=emb)

        if mask is not None and hasattr(self, "embedding_layer"):
            emb = add_mask_to_emb_dict(emb, mask)

        if isinstance(self.linear, DenseLayer):
            x = self.linear(x, emb=emb)
        else:
            x = self.linear(x)

        x_shape = x.shape
        if hasattr(self, "embedding_layer"):
            emb_ = self.embedder(emb).squeeze(dim=1)
            scale, shift = self.embedding_layer(emb_).chunk(2, dim=-1)
            n = scale.shape[1]
            scale, shift = scale.view(scale.shape[0], scale.shape[1], -1, *x_shape[3:]), shift.view(scale.shape[0],
                                                                                                    scale.shape[1], -1,
                                                                                                    *x_shape[3:])

            if hasattr(self, "gamma1"):
                x = x * (self.gamma1 * scale + 1) + self.gamma2 * shift
            else:
                x = x * (scale + 1) + shift

        return x


def check_get(block_conf, arg_dict, defaults, key):
    if hasattr(block_conf,key):
        return getattr(block_conf, key)
    elif key in arg_dict:
        return arg_dict[key]
    elif key in defaults:
        return defaults[key]
    else:
        raise KeyError(f"Key '{key}' not found block_conf, model arguments and defaults")