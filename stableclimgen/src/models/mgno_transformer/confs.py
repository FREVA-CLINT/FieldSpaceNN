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
    'block_type': 'pre_layer_norm',
    'with_gamma': True,
    'omit_backtransform': True,
    'no_zoom_step': 1,
    'concat_model_dim': 1,
    'layer_type': 'Dense',
    'interpolate_input': True,
    'learn_residual': True,
    'concat_prev': False
}


class MGNOEncoderDecoderConfig:
    def __init__(self, 
                 out_zooms: List,
                 no_zooms: List,
                 out_features: List,
                 no_layer_settings: Dict={},
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)


class MGNOStackedEncoderDecoderConfig:
    def __init__(self, 
                 out_zooms: List,
                 no_zooms: int,
                 out_features: List,
                 no_layer_settings: Dict={},
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)