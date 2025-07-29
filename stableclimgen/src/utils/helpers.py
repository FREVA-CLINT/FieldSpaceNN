import omegaconf
from collections import defaultdict
from typing import List, Tuple, Optional
import re
import torch

def load_from_state_dict(model, ckpt_path, print_keys=True):
    weights = torch.load(ckpt_path) 
    res = model.load_state_dict(weights['state_dict'], strict=False)

    if print_keys:
        zoom_counts_missing, block_counts_missing = analyze_keys(res.missing_keys)
        zoom_counts, block_counts = analyze_keys(res.unexpected_keys)

        print("Missing keys in checkpoint:")
        for zoom, count in sorted(zoom_counts_missing.items()):
            print(f"  Zoom level {zoom}: {count} keys")

        print("Unexpected keys in checkpoint:")
        for zoom, count in sorted(zoom_counts.items()):
            print(f"  Zoom level {zoom}: {count} keys")
    
    return model

def extract_block_and_zoom_from_key(key: str) -> Optional[Tuple[int, int]]:
    """
    Extracts (block, zoom) from a parameter key, supporting patterns like:
        - model.encoder_blocks.{block}.blocks.{zoom}.
        - model.decoder_blocks.{block}.blocks.{zoom}.
        - model.{block}.blocks.{zoom}.

    Returns:
        Tuple of (block, zoom) if matched, else None.
    """
    match = re.search(
        r'model(?:\.(?:encoder_blocks|decoder_blocks|Blocks))?\.(\d+)\.blocks\.(\d+)\.', key
    )
    if match:
        block = int(match.group(1))
        zoom = int(match.group(2))
        return block, zoom
    return None

def analyze_keys(missing_keys: List[str]):
    """
    Analyzes missing state_dict keys and counts how many belong
    to each zoom level and block.

    Returns:
        zoom_level_counts: {zoom_level: count}
        block_zoom_counts: {(block, zoom_level): count}
    """
    zoom_level_counts = defaultdict(int)
    block_zoom_counts = defaultdict(int)

    for key in missing_keys:
        result = extract_block_and_zoom_from_key(key)
        if result:
            block, zoom = result
            zoom_level_counts[zoom] += 1
            block_zoom_counts[(block, zoom)] += 1

    return dict(zoom_level_counts), dict(block_zoom_counts)

def get_zoom_keys(model, zooms: List[int]) -> List[str]:
    """
    Returns parameter names in the model that belong to the specified zoom levels.

    Parameters:
        model (nn.Module): The model to search.
        zooms (List[int]): List of zoom levels.

    Returns:
        List[str]: Matching parameter keys.
    """
    matched_keys = []

    for name, _ in model.named_parameters():
        result = extract_block_and_zoom_from_key(name)
        if result:
            _, zoom = result
            if zoom in zooms:
                matched_keys.append(name)

    return matched_keys

def freeze_zoom_levels(model, zooms: List[int]):
    """
    Freezes parameters in the model that belong to specified zoom levels.

    Parameters:
        model (nn.Module): Model whose parameters will be modified.
        zooms (List[int]): Zoom levels to freeze.
    """
    zoom_keys = set(get_zoom_keys(model, zooms))
    for name, param in model.named_parameters():
        if name in zoom_keys:
            param.requires_grad = False


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