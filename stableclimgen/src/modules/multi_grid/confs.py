from typing import List, Dict
import copy
import omegaconf


class MGProcessingConfig:
    def __init__(self, 
                 layer_settings: List,
                 out_features: List):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
                setattr(self, input, value)


class MGEncoderDecoderConfig:
    def __init__(self, 
                 out_zooms: List,
                 no_zooms: List,
                 in_features: List,
                 no_layer_settings: Dict={},
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)