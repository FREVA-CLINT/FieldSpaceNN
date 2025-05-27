from ...utils.helpers import check_get_missing_key
from .polar_normal import polNormal_NoLayer
from .von_mises import VonMises_NoLayer
from typing import List
from ...modules.grids.grid_layer import GridLayer
import torch
import torch.nn as nn
from ...modules.embedding.embedder import EmbedderSequential, EmbedderManager, BaseEmbedder

def get_no_layer(rcm,
                 type,
                 in_zoom: int,
                 no_zoom: int,
                 out_zoom: int,
                 precompute_encode, 
                 precompute_decode,
                 layer_settings,
                 normalize_to_mask=True
                 ):
    
    n_params = check_get_missing_key(layer_settings, "n_params", ref=type)
    global_params_learnable = check_get_missing_key(layer_settings, "global_params_learnable", ref=type)

    if type == 'polNormal':

        assert len(n_params)==3, "len(n_params) should be equal to 3 for polNormal_NoLayer"
        assert len(global_params_learnable)==2, "len(global_params_learnable) should be equal to 2 for polNormal_NoLayer"

        no_layer = polNormal_NoLayer(
                rcm,
                in_zoom,
                no_zoom,
                out_zoom,
                n_phi=n_params[0],
                n_dist=n_params[1],
                n_sigma=n_params[2],
                dist_learnable=global_params_learnable[0],
                sigma_learnable=global_params_learnable[1],
                nh_in_encode=layer_settings.get("nh_in_encode",True), 
                nh_in_decode=layer_settings.get("nh_in_decode",True),
                precompute_encode=precompute_encode,
                precompute_decode=precompute_decode,
                normalize_to_mask=normalize_to_mask
            )
    
    elif 'VonMises' in type:

        assert len(n_params)==1, "len(n_params) should be equal to 1 for von mises NO layer"
        assert len(global_params_learnable)==2, "len(global_params_learnable) should be equal to 2 for von mises NO layer"

        kappa = check_get_missing_key(layer_settings, "kappa", ref=type)
        sigma = check_get_missing_key(layer_settings, "sigma", ref=type)

        no_layer = VonMises_NoLayer(
                rcm,
                in_zoom,
                no_zoom,
                out_zoom,
                n_phi=n_params[0],
                kappa=kappa,
                sigma=sigma,
                kappa_learnable=global_params_learnable[0],
                sigma_learnable=global_params_learnable[1],
                nh_in_encode=layer_settings.get("nh_in_encode",True), 
                nh_in_decode=layer_settings.get("nh_in_decode",True),
                precompute_encode=precompute_encode,
                precompute_decode=precompute_decode,
                diff = 'diff' in type,
                normalize_to_mask=normalize_to_mask
            )
    return no_layer