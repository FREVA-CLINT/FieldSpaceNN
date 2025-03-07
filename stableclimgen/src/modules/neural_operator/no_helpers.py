from ...utils.helpers import check_get_missing_key
from .polar_normal import polNormal_NoLayer
from .von_mises import VonMises_NoLayer
from ...modules.icon_grids.grid_layer import GridLayer
import torch
import torch.nn as nn
from ...modules.embedding.embedder import EmbedderSequential, EmbedderManager, BaseEmbedder

def get_no_layer(type,
                 grid_layer_encode, 
                 grid_layer_no, 
                 grid_layer_decode, 
                 precompute_encode, 
                 precompute_decode, 
                 rotate_coordinate_system,
                 layer_settings,
                 normalize_to_mask=None
                 ):
    
    n_params = check_get_missing_key(layer_settings, "n_params", ref=type)
    global_params_learnable = check_get_missing_key(layer_settings, "global_params_learnable", ref=type)

    if type == 'polNormal':

        assert len(n_params)==3, "len(n_params) should be equal to 3 for polNormal_NoLayer"
        assert len(global_params_learnable)==2, "len(global_params_learnable) should be equal to 2 for polNormal_NoLayer"

        no_layer = polNormal_NoLayer(
                grid_layer_encode,
                grid_layer_no,
                grid_layer_decode,
                n_phi=n_params[0],
                n_dist=n_params[1],
                n_sigma=n_params[2],
                dist_learnable=global_params_learnable[0],
                sigma_learnable=global_params_learnable[1],
                nh_in_encode=layer_settings.get("nh_in_encode",True), 
                nh_in_decode=layer_settings.get("nh_in_decode",True),
                precompute_encode=precompute_encode,
                precompute_decode=precompute_decode,
                rotate_coord_system=rotate_coordinate_system,
                normalize_to_mask=normalize_to_mask
            )
    
    elif 'VonMises' in type:

        assert len(n_params)==1, "len(n_params) should be equal to 1 for von mises NO layer"
        assert len(global_params_learnable)==2, "len(global_params_learnable) should be equal to 2 for von mises NO layer"

        kappa = check_get_missing_key(layer_settings, "kappa", ref=type)
        sigma = check_get_missing_key(layer_settings, "sigma", ref=type)

        no_layer = VonMises_NoLayer(
                grid_layer_encode,
                grid_layer_no,
                grid_layer_decode,
                n_phi=n_params[0],
                kappa=kappa,
                sigma=sigma,
                kappa_learnable=global_params_learnable[0],
                sigma_learnable=global_params_learnable[1],
                nh_in_encode=layer_settings.get("nh_in_encode",True), 
                nh_in_decode=layer_settings.get("nh_in_decode",True),
                precompute_encode=precompute_encode,
                precompute_decode=precompute_decode,
                rotate_coord_system=rotate_coordinate_system,
                diff = 'diff' in type
            )
    return no_layer


def add_mask_to_emb_dict(emb_dict: dict, mask: torch.tensor):

    if mask.dim()==5:
        mask = mask.squeeze(dim=2).squeeze(dim=-1)

    if mask.dim()==4:
        mask = mask.squeeze(dim=2)

    emb_dict['MaskEmbedder'] = mask.int()

    return emb_dict


def add_coordinates_to_emb_dict(grid_layer: GridLayer, indices_layers, emb):

    coords = grid_layer.get_coordinates_from_grid_indices(
        indices_layers[int(grid_layer.global_level)] if indices_layers else None)
    
    if emb is None:
        emb = {}

    emb['CoordinateEmbedder'] = coords

    return emb



def update_mask(mask, level_diff, mask2=None):
    if mask is None:
        return None
    
    if level_diff > 0:
        mask = mask.view(mask.shape[0], -1, 4**level_diff, *mask.shape[2:])
        mask = mask.sum(dim=2) == 4**level_diff
    elif level_diff < 0:
        mask = mask.unsqueeze(dim=2).repeat_interleave(4**(-1*level_diff), dim=2)
        mask = mask.view(mask.shape[0],-1, *mask.shape[3:])

    if mask2 is not None:
        mask = torch.logical_and(mask.view(mask2.shape) ,mask2)
        pass
    return mask


def get_embedder_from_dict(dict_: dict):
    if "embed_names" in dict_.keys() and "embed_confs" in dict_.keys():
        embed_mode = dict_.get("mode","sum")
        return get_embedder(dict_["embed_names"],
                            dict_["embed_confs"],
                            embed_mode)
    else:
        return None


def get_embedder(embed_names:list, 
                 embed_confs:list, 
                 embed_mode: list):
    
    emb_dict = nn.ModuleDict()
    for embed_name in embed_names:
        emb: BaseEmbedder = EmbedderManager().get_embedder(embed_name, **embed_confs[embed_name])
        emb_dict[emb.name] = emb     
        
    embedder = EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = 1)

    return embedder