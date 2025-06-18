import omegaconf


def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig) and not isinstance(value, tuple):
        value = [value]*n_repeat
    return value

"""
def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
    elif (isinstance(value, list) or isinstance(value, omegaconf.listconfig.ListConfig)) and len(value)<=1 and len(value)< n_repeat:
        value = [list(value) for _ in range(n_repeat)] if len(value)==0 else list(value)*n_repeat
    return value
"""

def check_get(confs, key):

    for conf in confs:
        if isinstance(conf, dict):
            if key in conf:
                return conf[key]
        elif hasattr(conf, key):
            return getattr(conf, key)
        
    raise KeyError(f"Key '{key}' not found block_conf, model arguments and defaults")

def check_get_missing_key(dict_: dict, key: str, ref=None):
    if key not in dict_.keys():
        if ref is None and 'type' in dict_.keys():
            raise Exception(f"key {key} is required for config {dict_['type']}")
        elif ref is not None:
            raise Exception(f"key {key} is required for config {ref}")
        else:
            raise Exception(f"key {key} is required")
    else:
        return dict_[key]
    
def get_parameter_group_from_state_dict(state_dict, key, return_reduced_keys=False):
    parameter_group = {}
    for state_key, state_value in state_dict.items():
        if key in state_key:
            k = state_key.split('.')[-1] if return_reduced_keys else state_key
            parameter_group[k] = state_value

    if len(parameter_group)==0:
        parameter_group=None
    
    return parameter_group

def expand_tensor(tensor, dims=5, keep_dims=None):
    if dims == 5:
        dim_dict = {
            "b": 0,
            "v": 1,
            "t": 2,
            "s": 3,
            "c": 4
        }
    else:
        dim_dict = {
            "b": 0,
            "v": 1,
            "t": 2,
            "s": [3, 4],
            "c": 5
        }

    if keep_dims is None:
        # keep all first dimensions
        keep_dims = dim_dict.keys()[:len(tensor.shape)]

    keep_dims = [item for dim in keep_dims for item in
                 (dim_dict[dim] if isinstance(dim_dict[dim], list) else [dim_dict[dim]])]

    assert len(tensor.shape) == len(keep_dims)

    for d in range(dims):
        if d not in keep_dims:
            tensor = tensor.unsqueeze(d)

    return tensor