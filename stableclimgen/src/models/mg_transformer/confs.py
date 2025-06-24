from typing import List, Dict
import copy


defaults = {
    "predict_var":False,
    'n_head_channels': 32,
    'layer_confs': {},
    'input_layer_confs': {},
    'embed_confs': {},
    'input_embed_confs': {},
    'output_embed_confs': {},
    'rotate_coord_system': True,
    'p_dropout': 0,
    'layer_type': 'Dense',
    'interpolate_input': True,
    'learn_residual': True,
    'with_residual': False,
    'aggregation': 'linear'
}

class MGEncoderConfig:
    def __init__(self, 
                 out_zooms: List,
                 type: str='diff',
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)

class MGDecoderConfig:
    def __init__(self,
                 type: str='sum', 
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)