"""Output conversion helpers for native regular multigrid models."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from ..modules.grids.grid_utils import decode_zooms
from ..modules.grids.regular_grid import morton_to_grid
from .visualization import regular_plot


def _decode_group(
    group: Mapping[int, torch.Tensor],
    sample_configs: Mapping[int, Dict[str, Any]],
) -> tuple[int, torch.Tensor]:
    max_zoom = max(int(zoom) for zoom in group)
    first = next(iter(group.values()))
    probabilistic = first.ndim == 7
    if probabilistic:
        batch, samples = first.shape[:2]
        flattened = {
            int(zoom): tensor.reshape(batch * samples, *tensor.shape[2:])
            for zoom, tensor in group.items()
        }
        decoded = decode_zooms(
            flattened,
            dict(sample_configs),
            max_zoom,
            root_cell_count=4,
        )[max_zoom]
        decoded = decoded.reshape(batch, samples, *decoded.shape[1:])
    else:
        decoded = decode_zooms(
            dict(group),
            dict(sample_configs),
            max_zoom,
            root_cell_count=4,
        )[max_zoom]
    return max_zoom, decoded


def regular_grid_outputs(
    zoom_output_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    sample_configs: Mapping[int, Dict[str, Any]],
) -> List[Optional[torch.Tensor]]:
    """Decode groups and replace their Morton spatial axis with ``H,W``."""
    converted: List[Optional[torch.Tensor]] = []
    for group in zoom_output_groups:
        if not group:
            converted.append(None)
            continue
        _, decoded = _decode_group(group, sample_configs)
        converted.append(morton_to_grid(decoded, spatial_dim=-3))
    return converted


def package_regular_prediction(
    zoom_output_groups: Sequence[Optional[Mapping[int, torch.Tensor]]],
    sample_configs: Mapping[int, Dict[str, Any]],
    *,
    encode_only: bool = False,
    mask_groups: Any = None,
) -> Dict[str, Any]:
    """Return both internal and row-major prediction layouts."""
    return {
        "zoom_outputs": zoom_output_groups,
        "grid_outputs": None if encode_only else regular_grid_outputs(zoom_output_groups, sample_configs),
        "mask": mask_groups,
    }


def _plot_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Convert ``B,V,T,H,W,D,F`` to ``B,(V*D*F),T,H,W``."""
    if tensor.ndim != 7:
        raise ValueError(f"Expected B,V,T,H,W,D,F for plotting, got {tuple(tensor.shape)}.")
    return tensor.permute(0, 1, 5, 6, 2, 3, 4).reshape(
        tensor.shape[0],
        tensor.shape[1] * tensor.shape[5] * tensor.shape[6],
        tensor.shape[2],
        tensor.shape[3],
        tensor.shape[4],
    )


def regular_multigrid_plot(
    input_zooms: Mapping[int, torch.Tensor],
    output_zooms: Mapping[int, torch.Tensor],
    target_zooms: Mapping[int, torch.Tensor],
    directory: str,
    *,
    sample_configs: Mapping[int, Dict[str, Any]],
    plot_name: str = "regular_multigrid",
    **kwargs: Any,
) -> List[str]:
    """Decode Morton zoom dictionaries and save ordinary image panels."""
    groups = [input_zooms, output_zooms, target_zooms]
    converted = []
    for group in groups:
        if not group:
            converted.append(None)
            continue
        _, decoded = _decode_group(group, sample_configs)
        converted.append(_plot_layout(morton_to_grid(decoded, spatial_dim=-3)))
    input_grid, output_grid, target_grid = converted
    if output_grid is None or target_grid is None:
        return []
    if input_grid is None:
        input_grid = torch.zeros_like(target_grid)
    return regular_plot(
        target_grid,
        input_grid,
        output_grid,
        plot_name,
        directory,
    )
