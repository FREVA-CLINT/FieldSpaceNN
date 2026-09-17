"""Grid-layer implementation for non-periodic square regular grids."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .regular_grid import inverse_morton_indices, morton_order_indices, side_from_zoom


class RegularGridLayer(nn.Module):
    """API-compatible regular-grid counterpart of :class:`GridLayer`.

    Invalid neighbors are represented by safe self-indices plus a boolean mask.
    This prevents wraparound while keeping every gathered neighborhood rectangular.
    """

    grid_type = "regular"
    root_cell_count = 4
    coord_system = "cartesian"

    def __init__(
        self,
        zoom: int,
        adjc: torch.Tensor,
        adjc_mask: torch.Tensor,
        coordinates: torch.Tensor,
        coord_system: str = "cartesian",
        periodic_fov: Optional[float] = None,
        nh_shift_indices: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.zoom = int(zoom)
        self.side = side_from_zoom(self.zoom)
        self.coord_system = coord_system
        self.periodic_fov = periodic_fov
        self.nh_shift_indices = nh_shift_indices or {
            "south": 8,
            "southwest": 7,
            "west": 6,
            "northwest": 5,
            "north": 4,
            "northeast": 3,
            "east": 2,
            "southeast": 1,
        }
        self.reverse_shift = {
            "south": "north",
            "north": "south",
            "northeast": "southwest",
            "southwest": "northeast",
            "west": "east",
            "east": "west",
            "southeast": "northwest",
            "northwest": "southeast",
        }
        if tuple(adjc.shape) != (self.side * self.side, 9):
            raise ValueError(
                f"Regular zoom {zoom} needs adjacency {(self.side * self.side, 9)}, "
                f"got {tuple(adjc.shape)}."
            )
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("adjc", adjc.to(torch.long), persistent=False)
        # Unlike legacy GridLayer, True consistently means invalid/masked here.
        self.register_buffer("adjc_mask", adjc_mask.to(torch.bool), persistent=False)
        self.register_buffer(
            "fov_mask",
            adjc_mask[:, 1:].all(dim=-1, keepdim=True),
            persistent=False,
        )

        valid_edges = ~adjc_mask[:, 1:]
        centers = coordinates[:, None, :].expand(-1, 8, -1)
        neighbor_coords = coordinates[adjc[:, 1:]]
        deltas = neighbor_coords - centers
        distances = torch.linalg.vector_norm(deltas, dim=-1)
        valid_distances = distances[valid_edges]
        self.nh_dist = valid_distances.mean()
        self.nh_dist_lon = deltas[..., 0].abs()[valid_edges].mean()
        self.nh_dist_lat = deltas[..., 1].abs()[valid_edges].mean()
        quantile_source = valid_distances.clamp_min(1e-12)
        self.dist_quantiles = quantile_source.quantile(torch.linspace(0.01, 0.99, 20))
        self.min_dist = quantile_source.min()
        self.max_dist = quantile_source.max()
        self.mean_dist = quantile_source.mean()
        self.median_dist = quantile_source.median()

    def n_cells(self, zoom: Optional[int] = None) -> int:
        zoom = self.zoom if zoom is None else int(zoom)
        return self.root_cell_count * 4**zoom

    def _token_indices(self, zoom_patch_out: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return fixed-width core-plus-one-ring token indices and invalid mask."""
        if zoom_patch_out == self.zoom:
            return self.adjc, self.adjc_mask
        if zoom_patch_out < 0:
            indices = torch.arange(self.n_cells(), device=self.adjc.device).view(1, -1)
            return indices, torch.zeros_like(indices, dtype=torch.bool)
        if zoom_patch_out > self.zoom:
            raise ValueError(
                f"Token zoom {zoom_patch_out} cannot exceed grid zoom {self.zoom}."
            )

        patch_side = 2 ** (self.zoom - zoom_patch_out)
        patches_per_side = side_from_zoom(zoom_patch_out)
        patch_order = morton_order_indices(patches_per_side, device=self.adjc.device)
        inverse = inverse_morton_indices(self.side, device=self.adjc.device)
        rows = torch.div(patch_order, patches_per_side, rounding_mode="floor")
        cols = patch_order % patches_per_side

        all_indices = []
        all_invalid = []
        for patch_row, patch_col in zip(rows.tolist(), cols.tolist()):
            row0, col0 = patch_row * patch_side, patch_col * patch_side
            patch_morton = int((patch_order == patch_row * patches_per_side + patch_col).nonzero()[0])
            core_start = patch_morton * patch_side * patch_side
            core = torch.arange(
                core_start,
                core_start + patch_side * patch_side,
                device=self.adjc.device,
                dtype=torch.long,
            )
            halo = []
            halo_invalid = []
            fallback = int(core[0])
            for row in range(row0 - 1, row0 + patch_side + 1):
                for col in range(col0 - 1, col0 + patch_side + 1):
                    if row0 <= row < row0 + patch_side and col0 <= col < col0 + patch_side:
                        continue
                    invalid = row < 0 or row >= self.side or col < 0 or col >= self.side
                    halo_invalid.append(invalid)
                    if invalid:
                        halo.append(fallback)
                    else:
                        halo.append(int(inverse[row * self.side + col]))
            halo_tensor = torch.tensor(halo, device=self.adjc.device, dtype=torch.long)
            all_indices.append(torch.cat((core, halo_tensor)))
            all_invalid.append(
                torch.cat(
                    (
                        torch.zeros(core.numel(), device=self.adjc.device, dtype=torch.bool),
                        torch.tensor(halo_invalid, device=self.adjc.device, dtype=torch.bool),
                    )
                )
            )
        return torch.stack(all_indices), torch.stack(all_invalid)

    @staticmethod
    def _gather(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return x[:, indices]

    def get_global_with_nh(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        zoom_patch_out: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        zoom_patch_out = self.zoom if zoom_patch_out is None else int(zoom_patch_out)
        indices, invalid = self._token_indices(zoom_patch_out)
        x_out = self._gather(x, indices)
        invalid_values = invalid.view(1, *invalid.shape, *([1] * (x_out.ndim - 3)))
        x_out = x_out.masked_fill(invalid_values, 0)
        if mask is not None:
            mask_out = self._gather(mask, indices)
            invalid_expanded = invalid.view(1, *invalid.shape, *([1] * (mask_out.ndim - 3)))
            if mask_out.dtype == torch.bool:
                mask_out = mask_out | invalid_expanded
            else:
                mask_out = mask_out.masked_fill(invalid_expanded, float("inf"))
        else:
            mask_out = invalid.unsqueeze(0).unsqueeze(-1)
        return x_out, mask_out

    def get_sample_patch_with_nh(
        self,
        x: torch.Tensor,
        patch_index: Any = 0,
        zoom_patch_sample: int = -1,
        mask: Optional[torch.Tensor] = None,
        zoom_patch_out: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Regular training uses full-image model inputs.  This method remains useful
        # for callers that explicitly request an independent hard-edged patch.
        if int(zoom_patch_sample) < 0:
            return self.get_global_with_nh(x, mask=mask, zoom_patch_out=zoom_patch_out)
        indices, invalid = self._token_indices(int(zoom_patch_sample))
        patch_index = torch.as_tensor(patch_index, device=indices.device).reshape(-1).to(torch.long)
        selected = indices[patch_index]
        selected_invalid = invalid[patch_index]
        x_out = self._gather(x, selected)
        invalid_values = selected_invalid.view(
            1, *selected_invalid.shape, *([1] * (x_out.ndim - 3))
        )
        x_out = x_out.masked_fill(invalid_values, 0)
        if mask is not None:
            mask_out = self._gather(mask, selected)
            invalid_expanded = selected_invalid.view(1, *selected_invalid.shape, *([1] * (mask_out.ndim - 3)))
            mask_out = mask_out | invalid_expanded if mask_out.dtype == torch.bool else mask_out.masked_fill(invalid_expanded, float("inf"))
        else:
            mask_out = selected_invalid.unsqueeze(0).unsqueeze(-1)
        return x_out, mask_out

    def get_nh(
        self,
        x: torch.Tensor,
        input_zoom: Optional[int] = None,
        patch_index: Any = 0,
        zoom_patch_sample: int = -1,
        mask: Optional[torch.Tensor] = None,
        zoom_patch_out: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if zoom_patch_out is None:
            zoom_patch_out = self.zoom
        b, v, t, spatial = x.shape[:4]
        feature_shape = x.shape[4:]
        # Some shared tokenizers carry the generated static boundary mask from
        # one zoom iteration into the next. It is not an input-data mask for the
        # new zoom and must not be reshaped against a different spatial length.
        if mask is not None and (mask.ndim < 4 or mask.shape[3] != spatial):
            mask = None
        if input_zoom is None:
            if int(zoom_patch_sample) >= 0:
                input_zoom = int(round(math.log(spatial, 4))) + int(zoom_patch_sample)
            else:
                input_zoom = int(round(math.log(spatial / self.root_cell_count, 4)))
        zoom_diff = int(input_zoom) - self.zoom
        if zoom_diff < 0:
            raise ValueError(
                f"Input zoom {input_zoom} is coarser than regular layer zoom {self.zoom}."
            )
        layer_spatial = spatial // 4**zoom_diff
        x_flat = x.reshape(b * v * t, layer_spatial, math.prod(feature_shape) * 4**zoom_diff)
        mask_flat = None
        mask_tail = None
        if mask is not None:
            mask_tail = mask.shape[4:]
            mask_flat = mask.reshape(b * v * t, layer_spatial, -1)

        if int(zoom_patch_sample) >= 0:
            gathered, gathered_mask = self.get_sample_patch_with_nh(
                x_flat,
                patch_index=patch_index,
                zoom_patch_sample=int(zoom_patch_sample),
                mask=mask_flat,
                zoom_patch_out=zoom_patch_out,
            )
        else:
            gathered, gathered_mask = self.get_global_with_nh(
                x_flat, mask=mask_flat, zoom_patch_out=zoom_patch_out
            )

        n_tokens = spatial // 4 ** (int(input_zoom) - int(zoom_patch_out))
        gathered = gathered.reshape(b, v, t, n_tokens, -1, *feature_shape)
        if gathered_mask is not None:
            if mask_tail is None:
                gathered_mask = gathered_mask.expand(b * v * t, -1, -1, -1)
                gathered_mask = gathered_mask.reshape(b, v, t, n_tokens, -1, 1)
            else:
                gathered_mask = gathered_mask.reshape(b, v, t, n_tokens, -1, *mask_tail)
        return gathered, gathered_mask

    def get_number_of_points_in_patch(self, zoom_patch_out: int) -> int:
        return int(self._token_indices(int(zoom_patch_out))[0].shape[-1])

    def get_idx_of_patch(
        self,
        patch_index: Optional[Any] = None,
        zoom_patch_sample: Optional[int] = None,
        return_local: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        if patch_index is None or zoom_patch_sample is None or int(zoom_patch_sample) < 0:
            return torch.arange(self.n_cells(), device=self.adjc.device).view(1, -1)
        n_cells = 4 ** (self.zoom - int(zoom_patch_sample))
        patch_index = torch.as_tensor(patch_index, device=self.adjc.device).reshape(-1).to(torch.long)
        offsets = torch.arange(n_cells, device=self.adjc.device)
        indices = patch_index[:, None] * n_cells + offsets
        return offsets.expand_as(indices) if return_local else indices

    def get_coordinates(
        self,
        patch_index: Optional[Any] = None,
        zoom_patch_sample: Optional[int] = None,
        with_nh: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if with_nh and patch_index is None:
            indices = self.adjc.unsqueeze(0)
        elif with_nh:
            indices = self._token_indices(int(zoom_patch_sample))[0][
                torch.as_tensor(patch_index, device=self.adjc.device).reshape(-1).to(torch.long)
            ]
        else:
            indices = self.get_idx_of_patch(
                patch_index=patch_index,
                zoom_patch_sample=zoom_patch_sample,
                return_local=False,
            )
        return self.coordinates[indices]

    def apply_shift(
        self,
        x: torch.Tensor,
        shift_direction: str,
        reverse: bool = False,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        gathered, gathered_mask = self.get_nh(x, mask=mask, **kwargs)
        direction = self.reverse_shift[shift_direction] if reverse else shift_direction
        index = self.nh_shift_indices[direction]
        shifted = gathered.select(dim=4, index=index)
        shifted = shifted.reshape(*shifted.shape[:3], -1, shifted.shape[-1])
        if gathered_mask is None:
            return shifted, None
        shifted_mask = gathered_mask.select(dim=4, index=index)
        shifted_mask = shifted_mask.reshape(
            *shifted_mask.shape[:3], -1, shifted_mask.shape[-1]
        )
        return shifted, shifted_mask
