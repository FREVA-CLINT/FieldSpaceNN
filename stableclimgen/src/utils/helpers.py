import omegaconf


def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig) and not isinstance(value, tuple):
        value = [value]*n_repeat
    return value

def check_get_missing_key(dict_: dict, key: str):
    if key not in dict_.keys():
        raise Exception(f"key {key} is required for layer {dict_['type']}")
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
            "t": 1,
            "s": 2,
            "v": 3,
            "c": 4
        }
    else:
        dim_dict = {
            "b": 0,
            "t": 1,
            "s": [2, 3],
            "v": 4,
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