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


class MGEncoderConfig:
    def __init__(self, 
                 global_levels_output: list,
                 global_levels_no: list,
                 model_dims_out: list,
                 layer_settings: list,
                 stacked_encoding: bool = False):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
                setattr(self, input, value)


class MGProcessingConfig:
    def __init__(self, 
                 layer_settings: List[List]):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
                setattr(self, input, value)


class MGDecoderConfig:
    def __init__(self, 
                 layer_settings: List[List]):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
                setattr(self, input, value)