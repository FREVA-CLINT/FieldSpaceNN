import torch
import torch.nn as nn
import torch.nn.functional as F

import xarray as xr
import omegaconf

def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
    return value

class ICON_Transformer(nn.Module):
    def __init__(self, 
                 icon_grid: str,
                 global_levels_block_encoder: list,
                 model_dims_encoder: list,
                 mg_encoder_simul: list | bool,
                 mg_encoder_n_sigma: list | int,
                 mg_decoder_n_sigma: list | int,
                 mg_encoder_n_dist: list | int=1,
                 mg_encoder_n_phi: list | int=1,
                 mg_encoder_phi_attention: list | bool = False,
                 mg_encoder_dist_attention: list | bool = False,
                 mg_encoder_sigma_attention: list | bool = False,
                 mg_decoder_n_dist: list | int=1,
                 mg_decoder_n_phi: list | int=1,
                 mg_decoder_phi_attention: list | bool = False,
                 mg_decoder_dist_attention: list | bool = False,
                 mg_decoder_sigma_attention: list | bool = False,
                 mg_decoder_nh_projection : list | bool = True,
                 global_levels_block_decoder: list=[],
                 model_dims_decoder: list =[],
                 dist_learnable: list | bool = True,
                 sigma_learnable: list | bool = True,
                 use_von_mises: list | bool = True,
                 with_mean_res: list | bool = True,
                 with_channel_res: list | bool = False,
                 kappa_init: list | float = 1.,
                 mg_spa_method: list | str = None,
                 mg_spa_min_lvl: list | str = None,
                 mg_encoder_kernel_settings_for_spa: bool = True,
                 nh: int=1,
                 seq_lvl_att: int=2,
                 model_dim_in: int=1,
                 model_dim_out: int=1,
                 n_head_channels:int=16,
                 pos_emb_calc: str='cartesian_km',
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True
                 ) -> None: 
        
                
        super().__init__()