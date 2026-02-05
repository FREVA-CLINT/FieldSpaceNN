from typing import List, Dict
import copy


defaults = {
    "predict_var":False,
    'n_head_channels': 32,
    'att_dim': 256,
    'layer_confs': {},
    'layer_confs_emb': {},
    'input_layer_confs': {},
    'embed_confs': {},
    'dropout': 0,
    'learn_residual': False,
    'with_residual': False,
    'masked_residual': False,
    'use_mask': False
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