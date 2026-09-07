"""Quantization helpers for FieldSpace hyper-prior models."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through estimator."""

    return (torch.round(x) - x).detach() + x


def quantize_training(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply training-time quantization approximation."""

    if mode == "noise":
        return x + torch.empty_like(x).uniform_(-0.5, 0.5)
    if mode == "ste":
        return round_ste(x)
    if mode == "none":
        return x
    raise ValueError(f"Unknown quantization mode '{mode}'.")


def map_nested(
    fn: Callable[[torch.Tensor], torch.Tensor],
    groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
) -> List[Optional[Dict[int, torch.Tensor]]]:
    """Apply ``fn`` to every tensor in a nested zoom-group structure."""

    output: List[Optional[Dict[int, torch.Tensor]]] = []
    for group in groups:
        if group is None:
            output.append(None)
            continue
        output.append({int(zoom): fn(tensor) for zoom, tensor in group.items()})
    return output
