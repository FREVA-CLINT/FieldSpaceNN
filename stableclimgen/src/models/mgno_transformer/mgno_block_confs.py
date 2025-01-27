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
                 global_levels: int|list, 
                 layer_settings: List[Dict],
                 global_res: bool=False, 
                 skip_mode: str='',
                 is_encoder_only: bool=False,
                 is_decoder_only: bool=False,
                 **kwargs):

        n_no_layers = len(global_levels)

        inputs = copy.deepcopy(locals())
        self.block_type = block_type
        self.global_res = global_res
        self.skip_mode = skip_mode

        for input, value in inputs.items():
       #     if input != 'self' and input != 'block_type' and input != "global_res" and input != "skip_mode" and input not in kwargs.keys():
                setattr(self, input, value)
        
        for key, value in kwargs.items():
            for layer_settings in self.layer_settings:
                if key not in layer_settings.keys():
                    layer_settings[key]=value