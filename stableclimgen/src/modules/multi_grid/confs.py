from typing import List, Dict
import copy
import omegaconf


class MGProcessingConfig:
    def __init__(self, 
                 layer_settings: List,
                 out_features: List,
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)

class MGCrossProcessingConfig:
    def __init__(self, 
                 layer_settings: List,
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)