from typing import List, Dict
import copy
import omegaconf

def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
    elif (isinstance(value, list) or isinstance(value, omegaconf.listconfig.ListConfig)) and len(value)<=1 and len(value)< n_repeat:
        value = [list(value) for _ in range(n_repeat)] if len(value)==0 else list(value)*n_repeat
    return value

defaults = {
    "predict_var":False,
    'n_vars_total': 1,
    'n_head_channels': 32, 
    'factorize_vars': True,
    'rank_vars': 8,
    'rotate_coord_system': True,
    'p_dropout': 0,
    'block_type': 'pre_layer_norm',
    'with_gamma': True,
    'omit_backtransform': True,
    'rank': 0.7,
    'rank_cross': 0.7,
    'no_rank_decay': 0,
    'no_level_step': 1,
    'concat_model_dim': 1,
    'seq_level': 2,
    'layer_type': 'Dense',
    'embed_mode': 'sum',
    'embed_names': [],
    'embed_confs': {},
    'interpolate_input': True,
    'learn_residual': True,
    'input_embed_names': [],
    'input_embed_confs': {},
    'input_embed_mode': "sum",
    'mg_reduction_embed_confs': {},
    'mg_reduction_embed_names': [],
    'mg_reduction_embed_names_mlp': [],
    'mg_reduction_embed_mode': "sum",
    'mg_att_dim': 64,
    'mg_n_head_channels': 16,
    'level_diff_zero_linear': True
}

class NOBlockConfig:

    def __init__(self, 
                 block_type: str,
                 model_dims_out: list,
                 layer_settings: list,
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
                setattr(self, input, value)
        
        for key, value in kwargs.items():
            if len(value) == len(self.layer_settings):
                for index in range(value):
                    if key not in self.layer_settings[index]:
                        self.layer_settings[index][key] = value


class MGEncoderDecoderConfig:
    def __init__(self, 
                 global_levels_output: list,
                 global_levels_no: list,
                 model_dims_out: list,
                 no_layer_settings: dict={},
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)


class MGStackedEncoderDecoderConfig:
    def __init__(self, 
                 global_levels_output: list,
                 global_levels_no: int,
                 model_dims_out: list,
                 no_layer_settings: dict={},
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)
      

class MGProcessingConfig:
    def __init__(self, 
                 layer_settings_levels: List[List],
                 model_dims_out: List[List]):

        if isinstance(layer_settings_levels[0], omegaconf.DictConfig):
            layer_settings_levels = [layer_settings_levels for _ in range(len(model_dims_out))]

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
                setattr(self, input, value)