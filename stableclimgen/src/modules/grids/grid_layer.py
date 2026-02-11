import math

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple

from .grid_utils import get_distance_angle


def get_nh_idx_of_patch(
    adjc: torch.Tensor,
    patch_index: int = 0,
    zoom_patch_sample: int = -1,
    return_local: bool = True,
    **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get neighborhood indices and mask for a patch.

    :param adjc: Adjacency tensor of shape ``(n, k)``.
    :param patch_index: Patch index to select.
    :param zoom_patch_sample: Patch sampling zoom.
    :param return_local: Whether to return local indices within the patch.
    :param kwargs: Additional keyword arguments (unused).
    :return: Tuple of (adjacency_patch, adjacency_mask) with shape ``(p, s, k)``,
        where ``p`` is the number of selected patches (1 for a scalar index) and
        ``s`` is the points per patch.
    """
    # Infer the global zoom from the adjacency size.
    zoom = int(math.log((adjc.shape[0]) / 4, 4))

    if zoom_patch_sample > -1:

        n_pts_patch = 4**(zoom-zoom_patch_sample)

        # Reshape adjacency into patches and select the requested patch.
        adjc_patch = adjc.reshape(-1, n_pts_patch, adjc.shape[-1])[patch_index].clone()

        index_range = (patch_index * n_pts_patch, (patch_index + 1) * n_pts_patch)

        # Mask indices that fall outside the selected patch.
        adjc_mask = (adjc_patch < index_range[0].view(-1, 1, 1)) | (adjc_patch >= index_range[1].view(-1, 1, 1))
    else:
        adjc_patch = adjc.clone().unsqueeze(dim=0)
        adjc_mask = (adjc_patch < 0) | (adjc_patch >= adjc_patch.shape[1])

    
    # Replace invalid entries with a valid index to keep downstream gathers safe.
    ind = torch.where(adjc_mask)
    adjc_patch[ind] = adjc_patch[ind[0],ind[1],0]

    if return_local:
        # Convert global indices to local patch indices.
        adjc_patch = adjc_patch - index_range[0].view(-1,1,1)
    
    return adjc_patch, adjc_mask

def get_idx_of_patch(
    adjc: torch.Tensor,
    patch_index: int,
    zoom_patch_sample: int,
    return_local: bool = True,
    **kwargs: Any
) -> torch.Tensor:
    """
    Get patch indices from adjacency.

    :param adjc: Adjacency tensor of shape ``(n, k)``.
    :param patch_index: Patch index to select.
    :param zoom_patch_sample: Patch sampling zoom.
    :param return_local: Whether to return local indices within the patch.
    :param kwargs: Additional keyword arguments (unused).
    :return: Patch index tensor of shape ``(s,)`` or ``(p, s)``.
    """
    
    # for icon and healpix
    zoom = int(math.log((adjc.shape[0]) / 4, 4))
    
    n_pts_patch = 4**(zoom-zoom_patch_sample) if zoom_patch_sample != -1 else adjc.shape[0]

    adjc_patch = adjc[:,0].reshape(-1, n_pts_patch)[patch_index].clone()
    
    index_range = (patch_index * n_pts_patch, (patch_index + 1) * n_pts_patch)

    if return_local:
        adjc_patch = adjc_patch - index_range[0].view(-1,1)
    return adjc_patch


def gather_nh_data(x: torch.Tensor, adjc_patch: torch.Tensor) -> torch.Tensor:
    """
    Gather neighborhood data for a batched tensor.

    :param x: Input tensor of shape ``(bvt, n, f)``, where ``bvt = b * v * t``.
    :param adjc_patch: Adjacency patch of shape ``(b, n, nh)``.
    :return: Neighborhood tensor of shape ``(bvt, n, nh, f)``, where ``f`` may
        represent flattened ``(d, f)`` features from the base shape.
    """
    bvt,s,f = x.shape
    b = adjc_patch.shape[0]

    x = x.view(b, -1, s, f)

    nh = adjc_patch.shape[-1]
    
    # Expand adjacency to gather across time and feature dimensions.
    adjc_patch = adjc_patch.view(adjc_patch.shape[0], 1, adjc_patch.shape[1]*adjc_patch.shape[2], 1)

    adjc_patch = adjc_patch.expand(-1, x.shape[1], -1, x.shape[-1])

    x = torch.gather(x, -2, index=adjc_patch)

    x = x.view(bvt, s, nh, f)

    return x

def propagate_assignments(
    adjc: torch.Tensor,
    adjc_mask: torch.Tensor,
    coords: torch.Tensor,
    nh_k: int = 2
):
    """
    Propagate missing neighborhood assignments.

    :param adjc: Adjacency tensor of shape ``(n, k)``.
    :param adjc_mask: Boolean mask for adjacency entries of shape ``(n, k)``.
    :param coords: Coordinate tensor of shape ``(n, 2)``.
    :param nh_k: Neighborhood index to propagate.
    :return: Tuple of (assignment_index, assignment_target) index tensors of shape
        ``(m,)``. Returns empty lists when no propagation is required.
    """

    shift_dir = 1 if nh_k <= 4 else -1 


    # Candidates are nodes with valid neighbors in direction nh_k.
    candidates = torch.where(~adjc_mask[:, nh_k])[0]
    # Missing targets are nodes that are never referenced by that neighbor slot.
    missing_targets = torch.where(
        torch.bincount(adjc[:, nh_k], minlength=adjc.shape[0]) == 0
    )[0]

    assignment_index, assignment_target = [], []

    if len(missing_targets) == 0 or len(candidates) == 0:
        return assignment_index, assignment_target

    all_missing_in_candidates = False
    while not all_missing_in_candidates and len(candidates) > 0:
        # Choose a lateral shift based on hemisphere to find a viable target.
        shift = torch.ones_like(candidates)
        shift[coords[candidates, 1] > 0] = -1
        shift = nh_k + shift_dir*shift

        # Map candidates to their shifted neighbor target.
        candidates_target = adjc[candidates, shift]
        assignment_index.append(candidates)
        assignment_target.append(candidates_target)

        # Keep propagating along the chain of targets.
        candidates = torch.stack(
            [torch.where(adjc[:, nh_k] == t)[0] for t in candidates_target]
        ).view(-1)

        all_missing_in_candidates = torch.stack([
            ((mt - torch.stack(assignment_target)) == 0).any()
            for mt in missing_targets
        ]).all()

    return torch.concat(assignment_index), torch.concat(assignment_target)


class GridLayer(nn.Module):
    """
    A grid layer with coordinate transforms, neighborhood gathering, and relative positioning.
    """
    
    def __init__(
        self,
        zoom: int,
        adjc: torch.Tensor,
        adjc_mask: torch.Tensor,
        coordinates: torch.Tensor,
        coord_system: str = "polar",
        periodic_fov: Optional[float] = None,
        nh_shift_indices: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Initializes the GridLayer with given parameters.

        :param zoom: The global zoom for hierarchy.
        :param adjc: The adjacency tensor of shape ``(n, k)``.
        :param adjc_mask: Mask for the adjacency tensor of shape ``(n, k)``.
        :param coordinates: The coordinates of grid points of shape ``(n, 2)``.
        :param coord_system: The coordinate system. Defaults to "polar".
        :param periodic_fov: The periodic field of view. Defaults to None.
        :param nh_shift_indices: Optional custom neighborhood shift indices.
        :return: None.
        """
        super().__init__()

        if nh_shift_indices is None:
            self.nh_shift_indices: Dict[str, int] = {
                'south': 8,
                'southwest': 7,
                'west': 6,
                'northwest': 5,
                'north': 4,
                'northeast': 3,
                'east': 2,
                'southeast': 1
            }
        else:
            self.nh_shift_indices = nh_shift_indices

        self.reverse_shift: Dict[str, str] = {
            'south': 'north',
            'north': 'south',
            'northeast': 'southwest',
            'southwest': 'northeast',
            'west': 'east',
            'east': 'west',
            'southeast': 'northwest',
            'northwest': 'southeast'
        }

        # Initialize attributes
        self.zoom: int = zoom
        self.coord_system: str = coord_system
        self.periodic_fov: Optional[float] = periodic_fov

        # Register buffers for coordinates and adjacency information
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.coordinates: torch.Tensor
        
        if zoom >= 1:
            # Propagate missing assignments for polar discontinuities.
            for k in [2,6]:
                i_shift, i_target = propagate_assignments(adjc, adjc_mask==False, coordinates, nh_k=k)
                if len(i_shift) > 0:
                    adjc[i_shift, k] = i_target 
        
        for k in [1,3,4,5,7,8]:
            # Fill missing adjacency entries with a consistent fallback direction.
            c = torch.where(adjc_mask[:,k])[0]
            adjc[c,k] = k -1 if k>1 else 8

        self.register_buffer("adjc", adjc, persistent=False)
        self.adjc: torch.Tensor

        # Create mask where adjacency is false
        self.register_buffer("adjc_mask", adjc_mask == False, persistent=False)
        self.adjc_mask: torch.Tensor
        # Mask for the field of view
        self.register_buffer("fov_mask", ((adjc_mask == False).sum(dim=-1) == adjc_mask.shape[1]).view(-1, 1), persistent=False)
        self.fov_mask: torch.Tensor

        # Sample distances for neighborhood statistics.
        n_samples = torch.min(torch.tensor([self.adjc.shape[0] - 1, 500]))
        nh_samples = self.adjc[:n_samples]

        coords_nh = coordinates[nh_samples]
        # Calculate relative distances in both polar and cartesian frames.

        coords_lon_nh, coords_lat_nh = coords_nh[...,0], coords_nh[...,1]
        
        dists, _ = get_distance_angle(coords_lon_nh[:,[0]], coords_lat_nh[:,[0]], coords_lon_nh, coords_lat_nh, base="polar", rotate_coords=True)
        dists_lon, dists_lat = get_distance_angle(coords_lon_nh[:,[0]], coords_lat_nh[:,[0]], coords_lon_nh, coords_lat_nh, base="cartesian", rotate_coords=True)

        
        self.nh_dist: torch.Tensor = dists[:,1:].mean()
        self.nh_dist_lon: torch.Tensor = dists_lon[:,1:].abs().mean()
        self.nh_dist_lat: torch.Tensor = dists_lat[:,1:].abs().mean()
        # Compute distance statistics
        self.dist_quantiles: torch.Tensor = dists[dists > 1e-10].quantile(torch.linspace(0.01,0.99,20))

        self.min_dist: torch.Tensor = dists[dists > 1e-6].min()
        self.max_dist: torch.Tensor = dists[dists > 1e-10].max()
        self.mean_dist: torch.Tensor = dists[dists > 1e-10].mean()
        self.median_dist: torch.Tensor = dists[dists > 1e-10].median()


    def get_sample_patch_with_nh(
        self,
        x: torch.Tensor,
        patch_index: int = 0,
        zoom_patch_sample: int = -1,
        mask: Optional[torch.Tensor] = None,
        zoom_patch_out: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gather neighborhood data for a sampled patch.

        :param x: Input tensor of shape ``(bvt, n, f)``, where ``bvt = b * v * t``.
        :param patch_index: Patch index to sample.
        :param zoom_patch_sample: Patch sampling zoom.
        :param mask: Optional mask tensor of shape ``(bvt, n, m)``.
        :param zoom_patch_out: Optional output zoom override.
        :return: Tuple of (x_with_nh, mask_with_nh) with shapes ``(bvt, n, nh, f)``
            and ``(b, ..., n, nh, m)`` respectively.
        """

        # Get neighborhood indices and adjacency mask.
        adjc_patch, adjc_mask = get_nh_idx_of_patch(self.adjc, patch_index, zoom_patch_sample)
        
        # Gather neighborhood data
        x = gather_nh_data(x, adjc_patch)

        if mask is not None:
            # Combine provided mask with adjacency mask.
            mask = gather_nh_data(mask, adjc_patch)
            mask = mask.view(adjc_mask.shape[0], -1, *mask.shape[1:])
            if mask.dtype == torch.bool:
                # Mark invalid neighbors as masked.
                mask = torch.logical_or(mask, adjc_mask.unsqueeze(dim=-1).unsqueeze(dim=1).expand_as(mask))
            else:
                # Fill invalid neighbors for float masks (e.g., additive attention masks).
                mask.masked_fill_(adjc_mask.unsqueeze(dim=-1).unsqueeze(dim=1).expand_as(mask), float("inf"))
        else:
            # Use adjacency mask if no mask is provided.
            mask = adjc_mask.unsqueeze(dim=-1).unsqueeze(dim=1)
        
        return x, mask
    
    def get_global_with_nh(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        zoom_patch_out: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gather global neighborhood data at the requested output zoom.

        :param x: Input tensor of shape ``(b, n, f)`` or compatible.
        :param mask: Optional mask tensor of shape ``(b, n, m)``.
        :param zoom_patch_out: Output zoom for patching.
        :return: Tuple of (x_with_nh, mask_with_nh) with shapes ``(b, p, s, f)``
            and ``(b, p, s, m)``, where ``p`` is the number of patches and ``s``
            is the patch size plus neighborhood points.
        """
        
        zoom_patch_out = self.zoom if zoom_patch_out is None else zoom_patch_out
        # Select full or reduced neighborhood indices depending on zoom.
        if (zoom_patch_out - self.zoom) == 0:
            indices = self.adjc
        else:
            # this removes rundundancy, but still has some
            indices = self.adjc[:,[2,4,6,8]]
      
        # Build patch start indices and group adjacency into patches.
        patch_indices = torch.arange(0, indices.shape[0] + 4**(self.zoom - zoom_patch_out), 4**(self.zoom - zoom_patch_out), device=x.device, dtype=indices.dtype).view(-1,1,1)

        indices = indices.view(-1, 4**(self.zoom - zoom_patch_out),  indices.shape[-1])

        # Identify neighbors that fall outside each patch.
        out_of_patch = (indices < patch_indices[:-1]) | (indices >= patch_indices[1:])

        assert len(out_of_patch.sum(dim=[-1,-2]).unique())==1, "neighbourhood not consistent"

        # Append out-of-patch neighbors to the patch indices.
        nh_indices = indices[out_of_patch].view(indices.shape[0], int(out_of_patch.sum(dim=[-1,-2])[0]))

        patch_indices = torch.concat((indices[:,:,0].view(indices.shape[0],-1), nh_indices), dim=-1)

        x = x[:,patch_indices]

        if mask is not None:
            mask = mask[:,patch_indices]
                
        return x, mask
    
    def get_number_of_points_in_patch(self, zoom_patch_out: int) -> int:
        """
        Compute the number of points in a patch at the given zoom.

        :param zoom_patch_out: Output zoom for patching.
        :return: Number of points in the patch.
        """
        if zoom_patch_out > -1:
            x_ = torch.zeros_like(self.adjc[:,0].unsqueeze(dim=0))
            n = self.get_global_with_nh(x_, zoom_patch_out=zoom_patch_out)[0].shape[-1]
        else:
            n = self.adjc[:,0].shape[-1]
        return n


    def apply_shift(
        self,
        x: torch.Tensor,
        shift_direction: str,
        patch_index: int = 0,
        zoom_patch_sample: int = -1,
        reverse: bool = False,
        mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply a directional neighborhood shift.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)`` or compatible.
        :param shift_direction: Direction key for the shift.
        :param patch_index: Patch index to sample.
        :param zoom_patch_sample: Patch sampling zoom.
        :param reverse: Whether to apply the reverse direction.
        :param mask: Optional mask tensor of shape ``(b, v, t, n, m)``.
        :param kwargs: Additional keyword arguments forwarded to `get_nh`.
        :return: Tuple of (shifted_x, shifted_mask) with shapes ``(b, v, t, n', f)``
            and ``(b, v, t, n', m)``, where ``n'`` may include flattened depth.
        """
        x, mask = self.get_nh(x, patch_index=patch_index, zoom_patch_sample=zoom_patch_sample, mask=mask, **kwargs)

        x = x.view(*x.shape[:4],self.adjc.shape[-1],-1,x.shape[-1])

        # Choose forward or reverse neighbor index.
        if reverse:
            index = self.nh_shift_indices[self.reverse_shift[shift_direction]]
        else:
            index = self.nh_shift_indices[shift_direction]

        x = x[:,:,:,:,index].reshape(*x.shape[:3],-1,x.shape[-1])

        if mask is not None:
            mask = mask.view(*mask.shape[:4],self.adjc.shape[-1],-1,mask.shape[-1])
            mask = mask[:,:,:,:,index].view(*mask.shape[:3],-1,mask.shape[-1])

        return x, mask
    
        
    def get_nh(
        self,
        x: torch.Tensor,
        input_zoom: Optional[int] = None,
        patch_index: int = 0,
        zoom_patch_sample: int = -1,
        mask: Optional[torch.Tensor] = None,
        zoom_patch_out: Optional[int] = None,
        **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gather neighborhood data for a tensor at the configured zooms.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)`` or compatible.
        :param input_zoom: Optional input zoom override.
        :param patch_index: Patch index to sample.
        :param zoom_patch_sample: Patch sampling zoom.
        :param mask: Optional mask tensor of shape ``(b, v, t, n, m)``.
        :param zoom_patch_out: Output zoom override.
        :param kwargs: Additional keyword arguments forwarded to sampling methods.
        :return: Tuple of (x_with_nh, mask_with_nh) with shapes
            ``(b, v, t, n_out, nh, d, f)`` (or ``(b, v, t, n_out, nh, f)``)
            and ``(b, v, t, n_out, nh, m)``.
        """
        
        if zoom_patch_out is None:
            zoom_patch_out = self.zoom

        b,v,t,s = x.shape[:4]
        bvt = (b,v,t)
        fs = x.shape[4:]
    
        if input_zoom is None:
            # Infer the input zoom from the spatial dimension.
            input_zoom = int(math.log(s) / math.log(4) + zoom_patch_sample) if zoom_patch_sample > -1 else int(math.log(s/5) / math.log(4))

        zoom_diff = input_zoom - self.zoom

        # Flatten extra feature dimensions so neighborhood gathering operates on 3D tensors.
        x = x.reshape(-1, s//4**zoom_diff, math.prod(fs)*4**zoom_diff)

        bvt_mask = mask.shape[:-3] if mask is not None else None

        mask = mask.reshape(math.prod(bvt), s//4**zoom_diff, -1) if mask is not None else None

        if zoom_patch_sample != -1:
            x, mask = self.get_sample_patch_with_nh(x, patch_index, zoom_patch_sample, mask, zoom_patch_out=zoom_patch_out)

        else:
            x, mask = self.get_global_with_nh(x, mask=mask, zoom_patch_out=zoom_patch_out)

        if bvt_mask is not None:
            # Restore original batch/variable/time grouping for masks.
            mask = mask.view(*bvt, s//4**zoom_diff, -1)

        elif mask is not None:
            # Broadcast masks when they do not include b/v/t axes.
            mask = mask.unsqueeze(dim=1).expand(-1, bvt[1], bvt[2], -1, -1, 4**zoom_diff, -1)

        x = x.view(*bvt, s//4**(input_zoom - zoom_patch_out), -1, *fs)
    
        return x, mask


    def get_idx_of_patch(
        self,
        patch_index: Optional[int] = None,
        zoom_patch_sample: Optional[int] = None,
        return_local: bool = True,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Get indices for a patch.

        :param patch_index: Optional patch index.
        :param zoom_patch_sample: Optional patch sampling zoom.
        :param return_local: Whether to return local indices within the patch.
        :param kwargs: Additional keyword arguments (unused).
        :return: Patch index tensor of shape ``(p, s)`` or ``(s,)``.
        """
        if patch_index is None:
            return self.adjc[:,0].unsqueeze(dim=0)
        else:
            return get_idx_of_patch(self.adjc, patch_index, zoom_patch_sample, return_local=return_local)


    def get_coordinates(
        self,
        patch_index: Optional[int] = None,
        zoom_patch_sample: Optional[int] = None,
        with_nh: bool = False,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Get coordinates for a patch or neighborhood.

        :param patch_index: Optional patch index.
        :param zoom_patch_sample: Optional patch sampling zoom.
        :param with_nh: Whether to include neighborhood coordinates.
        :param kwargs: Additional keyword arguments (unused).
        :return: Coordinate tensor of shape ``(b, s, 2)`` or ``(b, s, nh, 2)``
            when neighborhoods are included.
        """
        
        if patch_index is not None and not with_nh:
            indices = get_idx_of_patch(
                                    self.adjc,
                                    patch_index, 
                                    zoom_patch_sample,
                                    return_local=False)
        elif patch_index is not None and with_nh:
            indices = get_nh_idx_of_patch(
                                    self.adjc,
                                    patch_index, 
                                    zoom_patch_sample,
                                    return_local=False)[0]
        elif with_nh:
            indices = self.adjc.unsqueeze(dim=0)

        else:
            indices = self.adjc[:,[0]].unsqueeze(dim=0)

        return self.coordinates[indices]
