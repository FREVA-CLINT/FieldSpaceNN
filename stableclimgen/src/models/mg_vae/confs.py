from typing import List, Dict
import copy

defaults = {
    "embed_confs": {},
    "latent_ch": [4, 4],
    "block_type": "TransformerBlock",
    'layer_settings': {
        "blocks": ["s", "s", "mlp"],
        "nh": False
    },
    'n_head_channels': 16
}

class MGQuantConfig:
    def __init__(self, **kwargs):
        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)