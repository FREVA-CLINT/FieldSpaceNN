from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Union

import yaml


NormDictInput = Optional[
    Union[Mapping[Any, Any], str, os.PathLike[str]]
]


def _to_plain_container(value: Any) -> Any:
    """Convert config containers, such as OmegaConf objects, to Python containers."""
    if isinstance(value, Mapping):
        return {key: _to_plain_container(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_container(item) for item in value]
    return value


def load_norm_dict(value: NormDictInput, name: str = "norm_dict") -> Optional[Dict[Any, Any]]:
    """Resolve an inline normalization mapping or a legacy YAML/JSON file path."""
    if value is None:
        return None

    if isinstance(value, (str, os.PathLike)):
        path = os.path.expanduser(os.fspath(value))
        with open(path, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, Mapping):
            raise ValueError(
                f"`{name}` file must contain a mapping at its root, got "
                f"{type(loaded).__name__}."
            )
        return _to_plain_container(loaded)

    if not isinstance(value, Mapping):
        raise TypeError(
            f"`{name}` must be a mapping or a path string, got {type(value).__name__}."
        )
    return _to_plain_container(value)
