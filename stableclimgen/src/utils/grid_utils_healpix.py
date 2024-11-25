import math

import healpy as hp
import torch
import xarray as xr

from stableclimgen.src.utils.grid_utils_icon import get_coords_as_tensor, scale_coordinates


def healpix_get_adjacent_cell_indices(nside: int, nh: int = 5):
    """
    Function to get neighbored cell indices for Healpix grid.

    :param nside: Healpix resolution parameter
    :param nh: number of neighbor levels

    :returns: adjacent cell indices, duplicates mask
    """
    npix = hp.nside2npix(nside)
    global_indices = torch.arange(npix)

    # The first level of neighbors
    adjcs = [global_indices.view(-1, 1)]  # List of adjacency tensors, starting with the pixel indices
    duplicates = [torch.zeros_like(adjcs[0], dtype=torch.bool)]  # Mask for duplicates

    visited_neighbors = torch.full((npix, 1), fill_value=-1, dtype=torch.long)
    visited_neighbors[:, 0] = global_indices

    # Iterate over each neighbor level
    for level in range(1, nh + 1):
        current_neighbors = []

        # Find neighbors for the previous level
        for pixel in range(npix):
            # Get current pixel's neighbors
            prev_neighbors = adjcs[-1][pixel].tolist()  # Previous level neighbors
            new_neighbors = set()

            for prev_pixel in prev_neighbors:
                if prev_pixel >= 0:  # Ignore invalid entries
                    neighbors = set(hp.get_all_neighbours(nside, prev_pixel, nest=True))
                    neighbors.discard(-1)  # Remove invalid neighbors
                    new_neighbors.update(neighbors)

            # Remove already visited pixels
            new_neighbors.difference_update(visited_neighbors[pixel].tolist())
            current_neighbors.append(list(new_neighbors))

        # Ensure all rows have the same length by padding with -1
        max_len = max(len(neigh) for neigh in current_neighbors)
        padded_neighbors = [neigh + [-1] * (max_len - len(neigh)) for neigh in current_neighbors]
        adjc_tensor = torch.tensor(padded_neighbors, dtype=torch.long)

        # Update visited pixels
        if visited_neighbors.size(1) < max_len:
            padding_size = max_len - visited_neighbors.size(1)
            visited_neighbors = torch.cat([visited_neighbors, -torch.ones((npix, padding_size), dtype=torch.long)], dim=1)

        # Update the visited pixels
        visited_neighbors[:, :max_len] = adjc_tensor

        # Handle duplicates
        check_indices = torch.concat(adjcs, dim=-1).unsqueeze(dim=-2)
        is_prev = adjc_tensor.unsqueeze(dim=-1) - check_indices == 0
        is_prev = is_prev.sum(dim=-1) > 0

        is_removed = is_prev
        is_removed_count = is_removed.sum(dim=-1)

        # Resolve duplicates by majority as in the original function
        unique, counts = is_removed_count.unique(return_counts=True)
        majority = unique[counts.argmax()]

        for minority in unique[unique != majority]:
            where_minority = torch.where(is_removed_count == minority)[0]
            ind0, ind1 = torch.where(is_removed[where_minority])

            ind0 = ind0.reshape(len(where_minority), -1)[:, :minority - majority].reshape(-1)
            ind1 = ind1.reshape(len(where_minority), -1)[:, :minority - majority].reshape(-1)

            is_removed[where_minority[ind0], ind1] = False

        adjc_tensor = adjc_tensor[~is_removed]
        adjc_tensor = adjc_tensor.reshape(npix, -1)

        if level > 1:
            counts = []
            uniques = []
            for row in adjc_tensor:
                unique, count = row.unique(return_counts=True)
                uniques.append(unique)
                counts.append(len(unique))

            adjc_tensor = torch.nn.utils.rnn.pad_sequence(uniques, batch_first=True, padding_value=-1)
            duplicates_mask = adjc_tensor == -1
        else:
            duplicates_mask = torch.zeros_like(adjc_tensor)

        adjcs.append(adjc_tensor)
        duplicates.append(duplicates_mask)

    # Concatenate results
    adjc = torch.concat(adjcs, dim=-1)
    duplicates = torch.concat(duplicates, dim=-1)

    return adjc, duplicates


def healpix_pixel_lonlat_torch(nside):
    """
    Get the longitude and latitude coordinates of each pixel in a Healpix grid using PyTorch tensors.

    :param nside: Healpix resolution parameter
    :return: A tuple of PyTorch tensors (lon, lat) with the longitude and latitude of each pixel.
    """
    npix = hp.nside2npix(nside)  # Total number of pixels

    # Get pixel indices as a PyTorch tensor
    pixel_indices = torch.arange(npix, dtype=torch.long)

    # Get theta (colatitude) and phi (longitude) for each pixel using healpy
    theta, phi = hp.pix2ang(nside, pixel_indices.numpy(), nest=True)

    # Convert theta and phi to PyTorch tensors
    theta_tensor = torch.tensor(theta, dtype=torch.float32) - 0.5 * torch.pi
    phi_tensor = torch.tensor(phi, dtype=torch.float32) - torch.pi

    return torch.stack([phi_tensor, theta_tensor]).float()


def get_distance_angle(lon1: torch.tensor, lat1: torch.tensor, lon2: torch.tensor, lat2: torch.tensor,
                       base: str = 'polar', periodic_fov: list = None, rotate_coords=True) -> torch.tensor:
    """
    :param lon1: target longitude
    :param lat1: target latitude
    :param lon2: source longitude
    :param lat2: source latitude
    :param base: Optional: Returns relative coordinates in either polat coordinates (distance, angle) or cartesian coordiantes (distance longitude, distance latitude)
    :param periodic_fov: Optinal if data is defined on a local patch with periodic boundary conditions

    :returns: distances on the sphere between lon1,lat1 and lon2,lat2
    """
    # does not produce proper results for now
    # lat2 = torch.arcsin(torch.cos(theta)*torch.sin(lat2) - torch.cos(lon2)*torch.sin(theta)*torch.cos(lat2))
    # lon2 = torch.atan2(torch.sin(lon2), torch.tan(lat2)*torch.sin(theta) + torch.cos(lon2)*torch.cos(theta)) - phi

    if rotate_coords:
        theta = lat1
        phi = lon1

        x = (torch.cos(lon2) * torch.cos(lat2))
        y = (torch.sin(lon2) * torch.cos(lat2))
        z = (torch.sin(lat2))

        rotated_x = torch.cos(theta) * torch.cos(phi) * x + torch.cos(theta) * torch.sin(phi) * y + torch.sin(theta) * z
        rotated_y = -torch.sin(phi) * x + torch.cos(phi) * y
        rotated_z = -torch.sin(theta) * torch.cos(phi) * x - torch.sin(theta) * torch.sin(phi) * y + torch.cos(
            theta) * z

        lon2 = torch.atan2(rotated_y, rotated_x)
        lat2 = torch.arcsin(rotated_z)

        lat1 = lon1 = 0

        d_lons = (lon2).abs()
        d_lats = (lat2).abs()

        sgn_lat = torch.sign(lat2)
        sgn_lon = torch.sign(lon2)

    else:
        d_lons = 2 * torch.arcsin(torch.cos(lat1) * torch.sin(torch.abs(lon2 - lon1) / 2))
        d_lats = (lat2 - lat1).abs()
        sgn_lat = torch.sign(lat1 - lat2)
        sgn_lon = torch.sign(lon1 - lon2)

    sgn_lat[(d_lats).abs() / torch.pi > 1] = sgn_lat[(d_lats).abs() / torch.pi > 1] * -1
    d_lats = d_lats * sgn_lat

    sgn_lon[(d_lons).abs() / torch.pi > 1] = sgn_lon[(d_lons).abs() / torch.pi > 1] * -1
    d_lons = d_lons * sgn_lon

    if periodic_fov is not None:
        rng_lon = (periodic_fov[1] - periodic_fov[0])
        d_lons[d_lons > rng_lon] = d_lons[d_lons > rng_lon] - rng_lon
        d_lons[d_lons < -rng_lon] = d_lons[d_lons < -rng_lon] + rng_lon

    if base == "polar":
        distance = torch.sqrt(d_lats ** 2 + d_lons ** 2)
        phi = torch.atan2(d_lats, d_lons)

        return distance.float(), phi.float()

    else:
        return d_lons.float(), d_lats.float()


def get_nearest_to_healpix_recursive(c_t_global: torch.tensor, c_i: torch.tensor, level: int = 7,
                                     global_indices_i: torch.tensor = None, nh: int = 5, search_radius: int = 5,
                                     periodic_fov: list = None):
    """
    Function to get index tensor mapping any input grid to icon grid

    :param c_t_global: coordinates of input grid
    :param c_i: coordinates of icon grid
    :param level: coarsen level at which neighbours are calculated
    :param global_indices_i: Input indices specifying out of range data points
    :param nh: Number of neighbours to keep
    :param search_radius: Search radius in units of the grid distance
    :param periodic_fov: Optinal if data is defined on a local patch with periodic boundary conditions


    :returns: indices in range, in radius mask, tuple(distances, angles) between points
    """

    _, n_sec_i, _ = c_i.shape
    n_target = c_t_global.shape[-1]

    id_t = torch.arange(n_target)

    n_level = n_target // 4 ** level

    if level > 0:
        mid_points_corners = id_t.reshape(-1, 4, 4 ** (level - 1))[:, 1:, 0]
        mid_points = id_t.reshape(-1, 4, 4 ** (level - 1))[:, [0], 0]
    else:
        mid_points_corners = id_t.reshape(-1, 4)[:, 1:].repeat_interleave(4, dim=0)
        mid_points = id_t.unsqueeze(dim=-1)

    # get radius
    c_t_ = c_t_global[:, mid_points_corners]
    c_t_m = c_t_global[:, mid_points]

    dist_corners = get_distance_angle(c_t_[0].unsqueeze(dim=-1), c_t_[1].unsqueeze(dim=-1), c_t_m[0].unsqueeze(dim=-2),
                                      c_t_m[1].unsqueeze(dim=-2))[0]
    dist_corners_max = search_radius * dist_corners.max(dim=-1).values.max()

    c_i_ = c_i

    c_t_m = c_t_m.reshape(2, n_sec_i, -1)

    dist, phi = get_distance_angle(c_t_m[0].unsqueeze(dim=-1), c_t_m[1].unsqueeze(dim=-1), c_i_[0].unsqueeze(dim=-2),
                                   c_i_[1].unsqueeze(dim=-2), periodic_fov=periodic_fov)
    dist = dist.reshape(n_level, -1)
    phi = phi.reshape(n_level, -1)

    in_rad = dist <= dist_corners_max

    dist_values, indices_rel = dist.topk(in_rad.sum(dim=-1).max(), dim=-1, largest=False, sorted=True)

    if global_indices_i is None:
        global_indices = indices_rel
    else:
        global_indices = torch.gather(global_indices_i, index=indices_rel.reshape(global_indices_i.shape[0], -1),
                                      dim=-1)
        global_indices = global_indices.reshape(n_level, -1)

    if nh is not None:
        n_keep = nh
    else:
        n_keep = dist_values.shape[1]

    indices_keep = dist_values.topk(int(n_keep), dim=-1, largest=False, sorted=True)[1]

    dist_values = torch.gather(dist_values, index=indices_keep, dim=-1)
    in_rad = torch.gather(in_rad.reshape(n_level, -1), index=indices_keep, dim=-1)
    global_indices = torch.gather(global_indices, index=indices_keep, dim=-1)

    phi_values = torch.gather(phi, index=indices_keep, dim=-1)

    return global_indices, in_rad, (dist_values, phi_values)


def get_mapping_to_healpix_grid(coords_healpix: torch.tensor, coords_input: torch.tensor, search_radius: int = 3,
                                max_nh: int = 10, lowest_level: int = 0, periodic_fov=None) -> dict:
    """
    Iterator of the get_nearest_to_icon_recursive function

    :param coords_healpix: coordinates of icon grid
    :param coords_input: coordinates of input grid
    :param search_radius: Search radius in units of the grid distance
    :param max_nh: Maximum number of neighbours to have in output
    :param lowest_level: Lowest coarsen level (HR) of which to cacluate the neighbours
    :param periodic_fov: Optional if data is defined on a local patch with periodic boundary conditions


    :returns: grid mapping (dict)
    """

    level_start = int(math.log(coords_healpix.shape[-1]) / math.log(4))

    grid_mapping = {
        "level": [],
        "indices": [],
        "pos": [],
        "in_rng_mask": []
    }
    for k in range(level_start - lowest_level):
        level = level_start - k - 1

        if level == lowest_level:
            nh = max_nh
        else:
            nh = None

        if k == 0:
            indices, in_rng, pos = get_nearest_to_healpix_recursive(coords_healpix, coords_input.unsqueeze(dim=1),
                                                                    level=level, nh=nh, search_radius=search_radius,
                                                                    periodic_fov=periodic_fov)
        else:
            indices, in_rng, pos = get_nearest_to_healpix_recursive(coords_healpix, coords_input[:, indices], level=level,
                                                                    global_indices_i=indices, nh=nh,
                                                                    search_radius=search_radius, periodic_fov=periodic_fov)

        grid_mapping['level'] = level
        grid_mapping['indices'] = indices
        grid_mapping['pos'] = pos
        grid_mapping['in_rng_mask'] = in_rng

    return grid_mapping


def healpix_grid_to_mgrid(reference_grid_level:int, n_grid_levels:int, nh:int=0)->list:
    """
    Convert an ICON grid to a multi-level grid by calculating coordinates and adjacency
    information at each grid level.

    Parameters:
    ----------
    grid : xarray.Dataset
        The ICON grid dataset containing coordinates and adjacency information.
    n_grid_levels : int
        The number of grid levels to generate in the multi-level grid.
    clon_fov : tuple of float, optional
        Longitude range (min, max) for the field of view to subset the grid.
    clat_fov : tuple of float, optional
        Latitude range (min, max) for the field of view to subset the grid.
    nh : int, optional
        Number of halo (neighbor) cells to include in adjacency calculations (default is 0).
    extension : float, optional
        Proportion to extend the field of view boundaries, used when subset coordinates are specified.

    Returns:
    -------
    grids : list of dict
        A list of dictionaries, each representing a grid level with coordinates, adjacency
        information, and masking details for each cell at that level.
    """
    level_nside_npix = {}

    for i in range(10):
        nside = 2 ** (9 - i)
        npix = hp.nside2npix(nside)
        level_nside_npix[str(i)] = (nside, npix)

    global_levels = []
    grids = []

    for level in range(n_grid_levels):
        global_levels.append(level)

        nside, npix = level_nside_npix[str(level + reference_grid_level)]

        adjc, adjc_mask = healpix_get_adjacent_cell_indices(nside, nh)

        grid_lvl = {
            "coords": healpix_pixel_lonlat_torch(nside),
            "adjc": adjc,
            "adjc_lvl": adjc,
            "adjc_mask": adjc_mask,
            "global_level": level
        }
        grids.append(grid_lvl)

    return grids