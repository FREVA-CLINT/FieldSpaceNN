from typing import List, Dict
import copy
import omegaconf

def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
    elif (isinstance(value, list) or isinstance(value, omegaconf.listconfig.ListConfig)) and len(value)<=1 and len(value)< n_repeat:
        value = [list(value) for _ in range(n_repeat)] if len(value)==0 else list(value)*n_repeat
    return value


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
                 block_type: str = 'post_layer_norm',
                 mg_reduction:str = 'linear',
                 mg_reduction_embed_confs: dict=None,
                 mg_reduction_embed_names: list=None,
                 mg_reduction_embed_names_mlp: list=None,
                 mg_reduction_embed_mode: str = 'sum',
                 embed_confs: dict=None,
                 embed_names: list=None,
                 embed_mode: str = 'sum',
                 with_gamma: bool=True,
                 omit_backtransform: bool=False,
                 mg_att_dim: int = 128,
                 mg_n_head_channels: int=16,
                 rule: str = 'fc',
                 level_diff_zero_linear = False,
                 layer_type='Dense',
                 rank=4):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
                setattr(self, input, value)


class MGStackedEncoderDecoderConfig:
    def __init__(self, 
                 global_levels_output: list,
                 global_levels_no: int,
                 model_dims_out: list,
                 no_layer_settings: dict={},
                 block_type: str = 'post_layer_norm',
                 no_level_step: int = 1,
                 layer_type = 'Tucker',
                 concat_layer_type='Tucker',
                 reduction_layer_type = 'CrossTucker', 
                 concat_model_dim = 1,
                 p_dropout=0,
                 mask_as_embedding = False,
                 rank=4,
                 rank_cross=2,
                 with_gamma=False,
                 embed_confs: dict=None,
                 embed_names: list=None,
                 embed_mode: str = 'sum'):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
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