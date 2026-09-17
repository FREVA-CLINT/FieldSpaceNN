"""Utilities for square, non-periodic multiresolution regular grids.

Regular zoom ``z`` has side length ``2 ** (z + 1)`` and therefore
``4 ** (z + 1)`` cells.  Tensors are stored internally in Morton (Z-order), so
the four children of every quadtree cell are contiguous just as they are for a
nested HEALPix hierarchy.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch


REGULAR_ROOT_CELL_COUNT = 4


def side_from_zoom(zoom: int) -> int:
    """Return the square side length for a regular-grid zoom."""
    zoom = int(zoom)
    if zoom < 0:
        raise ValueError(f"Regular-grid zoom must be non-negative, got {zoom}.")
    return 2 ** (zoom + 1)


def zoom_from_side(side: int) -> int:
    """Return ``z`` for a side satisfying ``side == 2 ** (z + 1)``."""
    side = int(side)
    if side < 2 or side & (side - 1):
        raise ValueError(
            f"Regular-grid side must be 2 * 2**n for n >= 0, got {side}."
        )
    return int(math.log2(side)) - 1


def validate_regular_shape(height: int, width: int) -> int:
    """Validate a square regular grid and return its zoom."""
    if int(height) != int(width):
        raise ValueError(
            f"Regular multigrid inputs must be square, got {height}x{width}."
        )
    return zoom_from_side(int(height))


def _part1by1(values: torch.Tensor) -> torch.Tensor:
    """Spread the low 16 bits so another coordinate can be interleaved."""
    values = values.to(torch.int64) & 0x0000FFFF
    values = (values | (values << 8)) & 0x00FF00FF
    values = (values | (values << 4)) & 0x0F0F0F0F
    values = (values | (values << 2)) & 0x33333333
    values = (values | (values << 1)) & 0x55555555
    return values


def morton_order_indices(side: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return row-major indices in Morton traversal order."""
    zoom_from_side(side)
    rows = torch.arange(side, device=device, dtype=torch.int64).view(-1, 1)
    cols = torch.arange(side, device=device, dtype=torch.int64).view(1, -1)
    codes = _part1by1(cols) | (_part1by1(rows) << 1)
    row_major = torch.arange(side * side, device=device, dtype=torch.long)
    return row_major[torch.argsort(codes.reshape(-1))]


def inverse_morton_indices(side: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return a row-major-index to Morton-index lookup table."""
    order = morton_order_indices(side, device=device)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return inverse


def grid_to_morton(
    tensor: torch.Tensor,
    spatial_dims: Tuple[int, int] = (-2, -1),
) -> torch.Tensor:
    """Replace adjacent ``H,W`` axes by one Morton-ordered spatial axis."""
    ndim = tensor.ndim
    y_dim = spatial_dims[0] % ndim
    x_dim = spatial_dims[1] % ndim
    if x_dim != y_dim + 1:
        raise ValueError("spatial_dims must name adjacent H,W axes in that order.")
    height, width = tensor.shape[y_dim], tensor.shape[x_dim]
    validate_regular_shape(height, width)
    moved = tensor.movedim((y_dim, x_dim), (-2, -1))
    flat = moved.reshape(*moved.shape[:-2], height * width)
    flat = flat.index_select(-1, morton_order_indices(height, device=tensor.device))
    return flat.movedim(-1, y_dim)


def morton_to_grid(
    tensor: torch.Tensor,
    spatial_dim: int = -1,
    *,
    side: int | None = None,
) -> torch.Tensor:
    """Replace one Morton spatial axis by adjacent row-major ``H,W`` axes."""
    ndim = tensor.ndim
    spatial_dim = spatial_dim % ndim
    n_cells = int(tensor.shape[spatial_dim])
    if side is None:
        side = int(math.isqrt(n_cells))
    if side * side != n_cells:
        raise ValueError(f"Spatial length {n_cells} is not a square grid.")
    zoom_from_side(side)
    moved = tensor.movedim(spatial_dim, -1)
    row_major = torch.empty_like(moved)
    order = morton_order_indices(side, device=tensor.device)
    row_major[..., order] = moved
    grid = row_major.reshape(*row_major.shape[:-1], side, side)
    return grid.movedim((-2, -1), (spatial_dim, spatial_dim + 1))


def mean_pool_2x2(tensor: torch.Tensor, spatial_dims: Tuple[int, int] = (-2, -1)) -> torch.Tensor:
    """Area-average adjacent 2x2 cells of a row-major grid."""
    ndim = tensor.ndim
    y_dim, x_dim = spatial_dims[0] % ndim, spatial_dims[1] % ndim
    if x_dim != y_dim + 1:
        raise ValueError("spatial_dims must name adjacent H,W axes in that order.")
    height, width = tensor.shape[y_dim], tensor.shape[x_dim]
    if height % 2 or width % 2:
        raise ValueError(f"Cannot 2x2-pool odd shape {height}x{width}.")
    moved = tensor.movedim((y_dim, x_dim), (-2, -1))
    pooled = moved.reshape(*moved.shape[:-2], height // 2, 2, width // 2, 2).mean(dim=(-3, -1))
    return pooled.movedim((-2, -1), (y_dim, x_dim))


def build_mean_pyramid(
    highest: torch.Tensor,
    zooms: Sequence[int],
    spatial_dims: Tuple[int, int] = (-2, -1),
) -> Dict[int, torch.Tensor]:
    """Build requested row-major zoom levels from the highest-resolution tensor."""
    requested = sorted({int(zoom) for zoom in zooms})
    if not requested:
        raise ValueError("At least one regular-grid zoom must be requested.")
    y_dim, x_dim = spatial_dims[0] % highest.ndim, spatial_dims[1] % highest.ndim
    highest_zoom = validate_regular_shape(highest.shape[y_dim], highest.shape[x_dim])
    if requested[-1] != highest_zoom:
        raise ValueError(
            f"Highest input is zoom {highest_zoom}, but requested maximum is {requested[-1]}."
        )
    pyramid = {highest_zoom: highest}
    current = highest
    for zoom in range(highest_zoom - 1, requested[0] - 1, -1):
        current = mean_pool_2x2(current, spatial_dims=spatial_dims)
        pyramid[zoom] = current
    return {zoom: pyramid[zoom] for zoom in requested}


def regular_coordinates(zoom: int) -> torch.Tensor:
    """Return Morton-ordered normalized pixel centers as ``(x,y)``."""
    side = side_from_zoom(zoom)
    x = (torch.arange(side, dtype=torch.float32) + 0.5) * (2.0 / side) - 1.0
    # Positive y points north/up while row indices increase downwards.
    y = 1.0 - (torch.arange(side, dtype=torch.float32) + 0.5) * (2.0 / side)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    return coords[morton_order_indices(side)]


def regular_adjacency(zoom: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return center+8-neighbor indices and an invalid-neighbor mask.

    Slots follow the existing directional convention:
    center, southeast, east, northeast, north, northwest, west, southwest, south.
    Invalid boundary neighbors use the center index as a safe gather fallback.
    """
    side = side_from_zoom(zoom)
    order = morton_order_indices(side)
    inverse = inverse_morton_indices(side)
    row = torch.div(order, side, rounding_mode="floor")
    col = order % side
    offsets = ((0, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0))
    neighbors: List[torch.Tensor] = []
    invalid: List[torch.Tensor] = []
    center = torch.arange(side * side, dtype=torch.long)
    for dy, dx in offsets:
        nr, nc = row + dy, col + dx
        bad = (nr < 0) | (nr >= side) | (nc < 0) | (nc >= side)
        safe_row_major = nr.clamp(0, side - 1) * side + nc.clamp(0, side - 1)
        neighbor = inverse[safe_row_major]
        neighbor = torch.where(bad, center, neighbor)
        neighbors.append(neighbor)
        invalid.append(bad)
    return torch.stack(neighbors, dim=-1), torch.stack(invalid, dim=-1)


def regular_grid_to_mgrid(zoom_max: int = 7, nh: int = 1) -> List[Dict[str, Any]]:
    """Build regular-grid metadata compatible with multigrid models."""
    if int(nh) != 1:
        raise ValueError("Regular grids currently support one center+8 neighborhood ring.")
    grids: List[Dict[str, Any]] = []
    for zoom in range(int(zoom_max) + 1):
        adjacency, adjacency_mask = regular_adjacency(zoom)
        grids.append(
            {
                "coords": regular_coordinates(zoom),
                "adjc": adjacency,
                "adjc_mask": adjacency_mask,
                "zoom": zoom,
                "side": side_from_zoom(zoom),
                "grid_type": "regular",
                "root_cell_count": REGULAR_ROOT_CELL_COUNT,
                "coord_system": "cartesian",
            }
        )
    return grids


def regular_output_to_grid(output: torch.Tensor, zoom: int, spatial_dim: int = 3) -> torch.Tensor:
    """Convert a model tensor's Morton spatial axis to row-major ``H,W``."""
    return morton_to_grid(output, spatial_dim=spatial_dim, side=side_from_zoom(zoom))


def make_loss_region_masks(
    reference_zooms: Mapping[int, torch.Tensor],
    loss_patch_zoom: int,
    patch_index: int,
    spatial_dim: int = 3,
) -> Dict[int, torch.Tensor]:
    """Build boolean inclusion masks for one aligned quadtree supervision region."""
    if loss_patch_zoom < 0:
        return {}
    masks: Dict[int, torch.Tensor] = {}
    for zoom, reference in reference_zooms.items():
        zoom = int(zoom)
        if zoom < loss_patch_zoom:
            raise ValueError(
                f"Loss patch zoom {loss_patch_zoom} exceeds target zoom {zoom}."
            )
        n_cells = 4 ** (zoom - loss_patch_zoom)
        start = int(patch_index) * n_cells
        stop = start + n_cells
        if stop > reference.shape[spatial_dim]:
            raise IndexError(f"Patch {patch_index} does not fit zoom {zoom}.")
        shape = [1] * reference.ndim
        shape[spatial_dim] = reference.shape[spatial_dim]
        mask = torch.zeros(shape, dtype=torch.bool, device=reference.device)
        index = [slice(None)] * reference.ndim
        index[spatial_dim] = slice(start, stop)
        mask[tuple(index)] = True
        masks[zoom] = mask.expand_as(reference)
    return masks
