from ...utils.helpers import check_get_missing_key
from .polar_normal import polNormal_NoLayer

def get_no_layer(type,
                 grid_layer_encode, 
                 grid_layer_no, 
                 grid_layer_decode, 
                 precompute_encode, 
                 precompute_decode, 
                 rotate_coordinate_system,
                 layer_settings,
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
                rotate_coord_system=rotate_coordinate_system
            )
    return no_layer