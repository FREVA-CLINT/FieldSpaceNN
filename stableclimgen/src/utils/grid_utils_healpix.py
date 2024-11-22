import healpy as hp
import torch


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
    theta_tensor = torch.tensor(theta, dtype=torch.float32)
    phi_tensor = torch.tensor(phi, dtype=torch.float32)

    # Use torch.rad2deg for conversion from radians to degrees
    lon = torch.rad2deg(phi_tensor)  # Convert longitude to degrees
    lat = 90.0 - torch.rad2deg(theta_tensor)  # Convert colatitude to latitude in degrees
    return torch.stack([phi_tensor, theta_tensor]).float()


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