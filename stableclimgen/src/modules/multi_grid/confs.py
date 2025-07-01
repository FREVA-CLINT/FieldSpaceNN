from typing import List, Dict
import copy
import omegaconf


class MGProcessingConfig:
    def __init__(self, 
                 out_features: List,
                 layer_settings: Dict={},
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)


class MGEncoderConfig:
    def __init__(self, 
                 out_zooms: List,
                 layer_settings: Dict={},
                 **kwargs):
        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            elif input != 'self':
                setattr(self, input, value)

class MGDiffEncoderConfig(MGEncoderConfig):
    pass


class MGDecoderConfig:
    def __init__(self, 
                 out_zoom: int,
                 layer_settings: Dict={},
                 **kwargs):
        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            elif input != 'self':
                setattr(self, input, value)


class MGDiffEncoderConfig(MGEncoderConfig):
    pass


class MGConservativeConfig:
    pass


class MGCoordinateEmbeddingConfig:
    def __init__(self, 
                 emb_zoom,
                 features,
                 **kwargs):
        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            elif input != 'self':
                setattr(self, input, value)

class MGSelfProcessingConfig:
    def __init__(self, 
                 layer_settings: List,
                 out_features: int,
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)