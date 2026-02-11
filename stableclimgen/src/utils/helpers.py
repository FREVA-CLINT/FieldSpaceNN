import omegaconf
from collections import defaultdict
import re
import torch
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def load_from_state_dict(model, ckpt_path, device=None, print_keys: bool = True):
    """
    Load a model from a checkpoint state dict with optional key diagnostics.

    :param model: Model instance with a compatible state dict.
    :param ckpt_path: Path to the checkpoint file.
    :param device: Optional device for torch.load map_location.
    :param print_keys: Whether to print missing/unexpected key summaries.
    :return: Tuple of (model, matching_keys).
    """
    weights = torch.load(ckpt_path, map_location=device)
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
    
    matching_keys = [key for key in weights['state_dict'].keys() if (key in model.state_dict().keys())]
    return model, matching_keys

def extract_block_and_zoom_from_key(key: str) -> Optional[Tuple[int, int]]:
    """
    Extracts (block, zoom) from a parameter key, supporting patterns like:
        - model.encoder_blocks.{block}.blocks.{zoom}.
        - model.decoder_blocks.{block}.blocks.{zoom}.
        - model.{block}.blocks.{zoom}.

    :return: Tuple of (block, zoom) if matched, else None.
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

    :param missing_keys: List of state_dict key names to analyze.
    :return: Tuple of (zoom_level_counts, block_zoom_counts).
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

def get_zoom_keys(model, zooms: List[int]):
    """
    Returns parameter names in the model that belong to the specified zoom levels.

    :param model: The model to search.
    :param zooms: List of zoom levels.
    :return: Matching parameter keys.
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

    :param model: Model whose parameters will be modified.
    :param zooms: Zoom levels to freeze.
    """
    zoom_keys = set(get_zoom_keys(model, zooms))
    freeze_params(model, zoom_keys)

def freeze_params(model, keys: List[str]):
    """
    Freezes parameters in the model that belong to specified zoom levels.

    :param model: Model whose parameters will be modified.
    :param keys: Parameter names to freeze.
    """
    for name, param in model.named_parameters():
        if name in keys:
            param.requires_grad = False

def check_value(value: Any, n_repeat: int):
    """
    Expand a scalar or singleton into a repeated list.

    :param value: Input value or list-like.
    :param n_repeat: Number of repeats if value is not list-like.
    :return: List of values with length n_repeat.
    """
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

def check_get(confs: Sequence[Any], key: str):
    """
    Retrieve a key from a list of dicts or objects, first match wins.

    :param confs: Sequence of dicts or objects to search.
    :param key: Key or attribute name to retrieve.
    :return: Retrieved value.
    """

    for conf in confs:
        if isinstance(conf, dict):
            if key in conf:
                return conf[key]
        elif hasattr(conf, key):
            return getattr(conf, key)
        
    raise KeyError(f"Key '{key}' not found block_conf, model arguments and defaults")

def check_get_missing_key(dict_: dict, key: str, ref: Optional[str] = None):
    """
    Ensure a key exists in a dict, raising an exception if missing.

    :param dict_: Dictionary to inspect.
    :param key: Required key.
    :param ref: Optional reference string for error messages.
    :return: Value associated with key.
    """
    if key not in dict_.keys():
        if ref is None and 'type' in dict_.keys():
            raise Exception(f"key {key} is required for config {dict_['type']}")
        elif ref is not None:
            raise Exception(f"key {key} is required for config {ref}")
        else:
            raise Exception(f"key {key} is required")
    else:
        return dict_[key]
    
def get_parameter_group_from_state_dict(state_dict: Mapping[str, Any], key: str, return_reduced_keys: bool = False):
    """
    Extract a subset of a state dict that matches a key substring.

    :param state_dict: State dict mapping parameter names to tensors.
    :param key: Substring to filter parameters.
    :param return_reduced_keys: Whether to strip leading modules in keys.
    :return: Dict of matching parameters or None if no matches.
    """
    parameter_group = {}
    for state_key, state_value in state_dict.items():
        if key in state_key:
            k = state_key.split('.')[-1] if return_reduced_keys else state_key
            parameter_group[k] = state_value

    if len(parameter_group)==0:
        parameter_group=None
    
    return parameter_group

def expand_tensor(tensor: torch.Tensor, dims: int = 5, keep_dims: Optional[Sequence[str]] = None):
    """
    Expand a tensor by inserting singleton dimensions to match a target layout.

    :param tensor: Input tensor.
    :param dims: Total number of dimensions in the output.
    :param keep_dims: Dimension labels to keep from the input.
    :return: Tensor expanded with singleton dimensions.
    """
    if dims == 5:
        dim_dict = {
            "b": 0,
            "v": 1,
            "t": 2,
            "s": 3,
            "d": 4,
            "c": 5
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
        # Keep all first dimensions.
        keep_dims = list(dim_dict.keys())[:len(tensor.shape)]

    keep_dims = [item for dim in keep_dims for item in
                 (dim_dict[dim] if isinstance(dim_dict[dim], list) else [dim_dict[dim]])]

    assert len(tensor.shape) == len(keep_dims)

    for d in range(dims):
        if d not in keep_dims:
            tensor = tensor.unsqueeze(d)

    return tensor

def merge_sampling_dicts(
    sample_configs: Mapping[int, Dict[str, Any]],
    patch_index_zooms: Mapping[int, torch.Tensor],
) -> Dict[int, Dict[str, Any]]:
    """
    Merge patch indices into the per-zoom sampling configuration.

    :param sample_configs: Sampling configuration dictionary per zoom.
    :param patch_index_zooms: Patch indices per zoom (shape ``(b,)``).
    :return: Updated sampling configuration dictionary.
    """

    sample_configs = sample_configs.copy()

    for key, value in patch_index_zooms.items():
        if key in sample_configs.keys():
            sample_configs[key]['patch_index'] = value

    # Ensure every zoom has a sampling config by inheriting from the lowest defined zoom.
    for z in range(max(sample_configs.keys())):
        if z not in sample_configs.keys():
            sample_configs[z] = sample_configs[min(sample_configs.keys())]

    return sample_configs
