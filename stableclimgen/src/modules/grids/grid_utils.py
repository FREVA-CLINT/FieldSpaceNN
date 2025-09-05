import torch
import xarray as xr
import numpy as np
import math
import healpy as hp
from typing import Dict

radius_earth= 6371

def get_zoom_from_npix(npix):
    try:
        nside = hp.npix2nside(npix)
        zoom_level = int(np.log2(nside))
        return zoom_level
    except ValueError:
        return None

def get_coords_as_tensor(ds: xr.Dataset, lon:str='vlon', lat:str='vlat', grid_type:str=None, target='torch'):
    """
    :param ds: Input Xarray Dataset to read data from
    :param lon: Input longitude variable to read
    :param lat: Input latitude variable to read
    :param grid_type: Either cell,edge or vertex to read different coordinates

    :returns: Dictionary of the longitude and latitude coordinates (pytorch tensors)
    """
    if grid_type == 'cell':
        lon, lat = 'clon', 'clat'

    elif grid_type == 'vertex':
        lon, lat = 'vlon', 'vlat'

    elif grid_type == 'edge':
        lon, lat = 'elon', 'elat'
    
    elif grid_type == 'lonlat':
        lon, lat = 'lon', 'lat'

    elif grid_type == 'longitudelatitude':
        lon, lat = 'longitude', 'latitude'
    
    if (lon not in ds.keys()) or (lat not in ds.keys()) and grid_type=='cell':
        zoom = get_zoom_from_npix(len(ds.cell))
        if zoom is not None:
            return healpix_pixel_lonlat_torch(zoom, return_numpy=target=='numpy')

    elif (lon not in ds.keys()) or (lat not in ds.keys()):
        return None

    if target=='torch':
        lons = torch.tensor(ds[lon].values)
        lats = torch.tensor(ds[lat].values)

        if grid_type=='lonlat' or grid_type =='longitudelatitude':
            lons, lats = torch.meshgrid((lons.deg2rad(),lats.deg2rad()), indexing='xy')
            lons = lons.reshape(-1)
            lats = lats.reshape(-1)


        coords = torch.stack((lons, lats), dim=-1).float()
    else:
        lons = np.array(ds[lon].values)
        lats = np.array(ds[lat].values)

        if grid_type=='lonlat':
            lons, lats = np.meshgrid(np.deg2rad(lons),np.deg2rad(lats),indexing='xy')
            lons = lons.reshape(-1)
            lats = lats.reshape(-1) 
        coords = np.stack((lons, lats), axis=-1).astype(np.float32)

    return coords

def get_grid_type_from_var(ds:xr.Dataset, variable:str) -> dict:
    dims = ds[variable].dims
    spatial_dim = dims[-1]

    if 'longitude' in spatial_dim or 'latitude' in spatial_dim:
        return 'longitudelatitude'

    if 'lon' in spatial_dim or 'lat' in spatial_dim:
        return 'lonlat'
    
    elif 'cell' in spatial_dim:
        return 'cell'

    elif 'vertex' in spatial_dim:
        return 'vertex'
    
    elif 'edge' in spatial_dim:
        return 'edge'
    


def get_coord_dict_from_var(ds:xr.Dataset, variable:str) -> dict:
    """
    :param ds: Input Xarray Dataset to read data from
    :param variable: Input variable to read
    :returns: Dictionary of the longitude and latitude coordinates (pytorch tensors)
    """
    dims = ds[variable].dims
    spatial_dim = dims[-1]
    
    if not 'lon' in spatial_dim and not 'lat' in spatial_dim:
        coords = list(ds[spatial_dim].coords.keys())
    else:
        coords = dims

    if len(coords)==0:
        coords = list(ds[variable].coords.keys())
        
    lon_c = [var for var in coords if 'lon' in var]
    lat_c = [var for var in coords if 'lat' in var]
    
    if len(lon_c)==0:
        len_points = len(ds.coords[spatial_dim].values)
        dims_match = [dim for dim in ds.coords if len(ds[dim].values)==len_points]

        lon_c = [var for var in dims_match if 'lon' in var]
        lat_c = [var for var in dims_match if 'lat' in var]

    assert len(lon_c)==1, "no longitude variable was found"
    assert len(lat_c)==1, "no latitude variable was found"

    return {'lon':lon_c[0],'lat':lat_c[0]}


def scale_coordinates(coords: torch.tensor, scale_factor:float)->torch.tensor:
    """
    Scale a set of coordinates around their centroid by a given scaling factor.

    Parameters:
    ----------
    coords : torch.Tensor
        Tensor containing coordinate points, with shape (dim, num_points), where `dim` is 
        the number of spatial dimensions (e.g., 2 for 2D coordinates) and `num_points` is 
        the number of coordinate points.
    scale_factor : float
        Factor by which to scale the coordinates around their mean.

    Returns:
    -------
    torch.Tensor
        The scaled coordinates, adjusted around their centroid by the specified scaling factor.
    """
    m = coords.mean(dim=1, keepdim=True)
    return (coords - m) * scale_factor + m


def distance_on_sphere(lon1: torch.tensor, lat1: torch.tensor, lon2: torch.tensor, lat2: torch.tensor) -> torch.tensor:
    """
    :param lon1: target longitude 
    :param lat1: target latitude 
    :param lon2: source longitude 
    :param lat2: source latitude 

    :returns: distances on the sphere between lon1,lat1 and lon2,lat2
    """
    d_lat = torch.abs(lat1-lat2)
    d_lon = torch.abs(lon1-lon2)
    asin = torch.sin(d_lat/2)**2

    d_rad = 2*torch.arcsin(torch.sqrt(asin + (1-asin-torch.sin((lat1+lat2)/2)**2)*torch.sin(d_lon/2)**2))
    return d_rad


def rotate_coord_system(lons: torch.tensor, lats: torch.tensor, rotation_lon: torch.tensor, rotation_lat: torch.tensor):
    """
    :param lons: input longitudes
    :param lats: input latitudes 
    :param rotation_lon: longitude of rotation angles 
    :param rotation_lat: latitudes of rotation angles 

    :returns: rotated coordinates as tuple(longitudes, latitudes)
    """

    theta = rotation_lat
    phi = rotation_lon

    theta = theta.view(-1,1)
    phi = phi.view(-1,1)

    x = (torch.cos(lons) * torch.cos(lats)).view(1,-1)
    y = (torch.sin(lons) * torch.cos(lats)).view(1,-1)
    z = (torch.sin(lats)).view(1,-1)

    rotated_x =  torch.cos(theta)*torch.cos(phi) * x + torch.cos(theta)*torch.sin(phi)*y + torch.sin(theta)*z
    rotated_y = -torch.sin(phi)*x + torch.cos(phi)*y
    rotated_z = -torch.sin(theta)*torch.cos(phi)*x - torch.sin(theta)*torch.sin(phi)*y + torch.cos(theta)*z

    rot_lon = torch.atan2(rotated_y, rotated_x)
    rot_lat = torch.arcsin(rotated_z)

    #lat = arcsin(cos(ϑ) sin(lat') - cos(lon') sin(ϑ) cos(lat'))
    #lon = atan2(sin(lon'), tan(lat') sin(ϑ) + cos(lon') cos(ϑ)) - φ
    
    return rot_lon, rot_lat




def get_distance_angle(lon1: torch.tensor, lat1: torch.tensor, lon2: torch.tensor, lat2: torch.tensor, base:str='polar', periodic_fov:list=None, rotate_coords=True) -> torch.tensor:
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
    #lat2 = torch.arcsin(torch.cos(theta)*torch.sin(lat2) - torch.cos(lon2)*torch.sin(theta)*torch.cos(lat2))
    #lon2 = torch.atan2(torch.sin(lon2), torch.tan(lat2)*torch.sin(theta) + torch.cos(lon2)*torch.cos(theta)) - phi

    if rotate_coords:
        theta = lat1
        phi = lon1

        x = (torch.cos(lon2) * torch.cos(lat2))
        y = (torch.sin(lon2) * torch.cos(lat2))
        z = (torch.sin(lat2))

        rotated_x =  torch.cos(theta)*torch.cos(phi) * x + torch.cos(theta)*torch.sin(phi)*y + torch.sin(theta)*z
        rotated_y = -torch.sin(phi)*x + torch.cos(phi)*y
        rotated_z = -torch.sin(theta)*torch.cos(phi)*x - torch.sin(theta)*torch.sin(phi)*y + torch.cos(theta)*z

        lon2 = torch.atan2(rotated_y, rotated_x)
        lat2 = torch.arcsin(rotated_z)

        lat1=lon1=0

        d_lons = (lon2).abs()
        d_lats = (lat2).abs() 

        sgn_lat = torch.sign(lat2)
        sgn_lon = torch.sign(lon2)

    else:
        d_lons =  2*torch.arcsin(torch.cos(lat1)*torch.sin(torch.abs(lon2-lon1)/2))
        d_lats = (lat2-lat1).abs() 
        sgn_lat = torch.sign(lat1-lat2)
        sgn_lon = torch.sign(lon1-lon2)
     
    sgn_lat[(d_lats).abs()/torch.pi>1] = sgn_lat[(d_lats).abs()/torch.pi>1]*-1
    d_lats = d_lats*sgn_lat

    sgn_lon[(d_lons).abs()/torch.pi>1] = sgn_lon[(d_lons).abs()/torch.pi>1]*-1
    d_lons = d_lons*sgn_lon

    if periodic_fov is not None:
        rng_lon = (periodic_fov[1] - periodic_fov[0])
        d_lons[d_lons > rng_lon] = d_lons[d_lons > rng_lon] - rng_lon
        d_lons[d_lons < -rng_lon] = d_lons[d_lons < -rng_lon] + rng_lon

    if base == "polar":
        distance = torch.sqrt(d_lats**2 + d_lons**2)
        phi = torch.atan2(d_lats, d_lons)

        return distance.float(), phi.float()

    else:
        return d_lons.float(), d_lats.float()
    



def global_indices_to_paths_dict(global_indices, zoom=None, sizes=None):
    """
    global_indices: 1D torch.Tensor of int64, shape [n]
    zoom: int, optional. If provided and sizes is None, sets the number of zoom levels (len(sizes)=zoom+1).
    sizes: list or 1D tensor of int, optional. If provided, gives the number of children per zoom level.
           If both are None, raises an error.
    Returns: dict {z: tensor of shape [n]} with z in [0, ..., num_levels-1], each tensor is the index at that zoom.
    """
   # global_indices = torch.as_tensor(global_indices, dtype=torch.long)

    if sizes is not None:
    #    sizes = torch.as_tensor(sizes, dtype=torch.long, device=global_indices.device)
        num_levels = len(sizes)
    elif zoom is not None:
        num_levels = zoom + 1
        sizes = torch.full((num_levels,), 4, dtype=torch.long, device=global_indices.device)
    else:
        raise ValueError("At least one of 'sizes' or 'zoom' must be provided.")

    # Compute strides: product of later sizes
    rev_sizes = sizes.flip(0)
    rev_cumprod = torch.cumprod(rev_sizes, 0)
    strides = torch.ones_like(sizes)
    strides[:-1] = rev_cumprod.flip(0)[1:]

    # Compute indices for all levels at once
    indices_matrix = (global_indices.unsqueeze(1) // strides) % sizes

    # Return as dict {zoom_level: 1D tensor}
    out = {z: indices_matrix[:, z] for z in range(num_levels)}
    return out

def get_zoom_x(x, zoom_patch_sample=None, **kwargs):
    s = x.shape[2]
    zoom_x = int(math.log(s) / math.log(4) + zoom_patch_sample) if zoom_patch_sample else int(math.log(s/5) / math.log(4))
    return zoom_x


def healpix_get_adjacent_cell_indices(zoom: int, nh: int = 5):
    """
    Function to get neighbored cell indices for Healpix grid.

    :param nside: Healpix resolution parameter
    :param nh: number of neighbor levels

    :returns: adjacent cell indices, duplicates mask
    """
    nside = 2**zoom
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


def healpix_pixel_lonlat_torch(zoom, return_numpy=False):
    """
    Get the longitude and latitude coordinates of each pixel in a Healpix grid using PyTorch tensors.

    :param nside: Healpix resolution parameter
    :return: A tuple of PyTorch tensors (lon, lat) with the longitude and latitude of each pixel.
    """
    nside = 2**zoom

    npix = hp.nside2npix(nside)  # Total number of pixels

    # Get pixel indices as a PyTorch tensor
    pixel_indices = torch.arange(npix, dtype=torch.long)

    # Get theta (colatitude) and phi (longitude) for each pixel using healpy
    theta, phi = hp.pix2ang(nside, pixel_indices.numpy(), nest=True)

    # Convert theta and phi to PyTorch tensors
    theta_tensor = torch.tensor(theta, dtype=torch.float32) - 0.5 * torch.pi
    phi_tensor = torch.tensor(phi, dtype=torch.float32) - torch.pi

    coords = torch.stack([phi_tensor, theta_tensor], dim=-1).float()

    if return_numpy:
        return coords.numpy()
    else:
        return coords


def healpix_grid_to_mgrid(zoom_max:int=10, nh:int=1)->list:
    
    zooms = []
    grids = []

    for zoom in range(zoom_max + 1):
        zooms.append(zoom)

        adjc, adjc_mask = healpix_get_adjacent_cell_indices(zoom, nh)

        grid_lvl = {
            "coords": healpix_pixel_lonlat_torch(zoom),
            "adjc": adjc,
            "adjc_mask": adjc_mask,
            "zoom": zoom
        }
        grids.append(grid_lvl)

    return grids


def estimate_healpix_cell_radius_rad(n_cells):
    return math.sqrt(4*math.pi / n_cells)


def hierarchical_zoom_distance_map(input_coords, max_zoom):
    """
    Iteratively computes distances between HEALPix cell centers and input_coords across zoom levels.

    Args:
        input_coords: (1, N, 2) in radians (lon, lat)
        max_zoom: int

    Returns:
        A dict of zoom_level → {
            "cell_centers": (B, 1, 2),
            "closest_input_coords": (B, K, 2),
            "distances": (B, K)
        }
    """
    #if input_coords.dim()==2:
    #    input_coords=input_coords.unsqueeze(dim=0)

    current_input = input_coords  # (1, N, 2)
    results = {}
    global_indices = torch.arange(input_coords.shape[0]).view(1,-1)

    for zoom in range(1, max_zoom + 1):
        c = healpix_pixel_lonlat_torch(zoom)  # (num_cells, 2)
        
        n_cells = c.shape[0]
        r = estimate_healpix_cell_radius_rad(n_cells)

        if zoom == 1:
            # First level: (1, n_cells, 1, 2)
            c = c.view(1, -1, 1, 2)
        else:
            # Next levels: batch size = current_input.shape[0]
            c = c.view(current_input.shape[0], -1, 1, 2)

        # Compute distance and angle
        d, _ = get_distance_angle(
            c[..., 0], c[..., 1],                   # shape: (B, M, 1)
            current_input[..., 0].unsqueeze(dim=-2), current_input[..., 1].unsqueeze(dim=-2),  # shape: (B, 1, N)
            base='polar',
            rotate_coords=False
        )  # Output: (B, M, N)

        in_search_radius = d <= r * 2
        n_radius = in_search_radius.sum(dim=-1)  # (B, M)
        n_keep = n_radius.max().item()   # scalar

        # Get top-k distances and indices
        d_sorted, idx_sorted = torch.topk(d, k=n_keep, dim=-1, largest=False)  # (B, M, k)


       # gathered_coords = torch.gather(current_input, dim=1,index=idx_sorted.view(idx_sorted.shape[0],-1,1).expand(-1,-1,2))
     #   gathered_coords = gathered_coords.view(gathered_coords.shape[0], c.shape[1], n_keep, 2)

        global_indices = torch.gather(global_indices, dim=1,index=idx_sorted.view(idx_sorted.shape[0],-1))
        global_indices = global_indices.view(idx_sorted.shape[0]*idx_sorted.shape[1], n_keep)

        current_input = input_coords[global_indices]

        c = c.view(-1,2)
        dim_out = c.shape[0]

        # Save current zoom level
        results[zoom] = {
 #           "cell_centers": c.view(-1,2),                 
            "indices": global_indices.view(dim_out,-1),
  #          "closest_input_coords": gathered_coords.view(dim_out,-1,2),           
            "distances": d_sorted.view(dim_out,-1),
            "resolution": r/2
        }
        
    #    current_input = gathered_coords.view(-1, gathered_coords.shape[-2], 2)

    return results


def get_mapping_weights_zoom(mapping_zooms: Dict, drop_mask=None,  mode='binary'): 

    weights_zooms = {}

    for zoom, mapping in mapping_zooms.items():
        weights_zooms[zoom] = get_mapping_weights(mapping, drop_mask=drop_mask, mode=mode)

    return weights_zooms

def get_mapping_weights(mapping, drop_mask=None,  mode='binary'): 

    weights = mapping['distances'] 

    if drop_mask is not None:
        drop_mask_zoom = drop_mask[mapping['indices']]
        weights = weights + (drop_mask_zoom * 1e5)

    if mode == '1/r':
        weights = mapping["resolution"]/weights 

        weights[weights> 1.] = 1.

        #weights = weights.mean(dim=-1)
    
    elif mode == 'normal':
        pass

    elif mode == 'binary':
        weights = mapping["resolution"]/weights 

        weights[weights> 1.] = 1.
        weights[weights< 1.] = 0.

        #weights = weights.mean(dim=-1).bool()

        weights = weights.bool()

    return weights


def to_zoom(x: torch.Tensor, in_zoom: int, out_zoom: int, mask: torch.Tensor = None, binarize_mask=False):
    """
    Rescales tensor x from in_zoom to out_zoom by averaging (zoom in) or repeating (zoom out).
    If mask is provided, performs masked averaging.
    """
    if in_zoom == out_zoom:
        return x, mask

    scale_factor = 4 ** abs(in_zoom - out_zoom)
    bvt = x.shape[:-2]
    c = x.shape[-1]

    if in_zoom > out_zoom:
        # Downsample by averaging
        x = x.view(*bvt, -1, scale_factor, c)
        if mask is not None:
            mask = mask.reshape(*bvt, -1, scale_factor, mask.shape[-1])
            weight = mask.sum(dim=-2, keepdim=True)
            x_zoom = (x * mask).sum(dim=-2, keepdim=True) / weight.clamp(min=1e-6)
            x_zoom[weight == 0] = 0

            mask_zoom = (weight / (1.*scale_factor))
            if binarize_mask:
                mask_zoom[mask_zoom > 0] = 1
                mask_zoom = mask_zoom.bool()

            x_zoom = x_zoom.view(*bvt, -1, c)
            mask_zoom = mask_zoom.view(*bvt, -1, mask_zoom.shape[-1])
            return x_zoom, mask_zoom
        else:
            x_zoom = x.mean(dim=-2)
            return x_zoom, None

    else:
        # Upsample by repeating
        x_zoom = x.unsqueeze(-2).repeat_interleave(scale_factor, dim=-2)
        if mask is not None:
            mask_zoom = mask.unsqueeze(-2).repeat_interleave(scale_factor, dim=-2)
            mask_zoom = mask_zoom * 1. if not binarize_mask else mask
            return x_zoom, mask_zoom
        else:
            return x_zoom, None

def get_sample_configs(sample_configs_zoom, zoom):
    if zoom in sample_configs_zoom:
        return sample_configs_zoom[zoom]
    
    else:
        for zoom_s in range(zoom+1, max(sample_configs_zoom.keys())+1):
            if zoom_s in sample_configs_zoom.keys():
                break
        
        cfgs = {'zoom_patch_sample': sample_configs_zoom[zoom_s]['zoom_patch_sample'],
                'patch_index': sample_configs_zoom[zoom_s]['patch_index'] // (4**(zoom_s-zoom))}
        return cfgs


def insert_matching_time_patch(x_h, x_s, zoom_h, zoom_target, sample_configs, base=12, add=False):
    if zoom_h == zoom_target:
        return x_s

    batched = True
    if x_h.dim() < 5:
        x_h = x_h.unsqueeze(0)
        x_s = x_s.unsqueeze(0)
        batched = False

    # Validate sizes
    assert sample_configs[zoom_h]['n_past_ts'] >= sample_configs[zoom_target]['n_past_ts'], \
        f'zoom {zoom_h} has a smaller number of past timesteps than {zoom_target}'
    assert sample_configs[zoom_h]['n_future_ts'] >= sample_configs[zoom_target]['n_future_ts'], \
        f'zoom {zoom_h} has a smaller number of future timesteps than {zoom_target}'
    assert sample_configs[zoom_h]['zoom_patch_sample'] <= sample_configs[zoom_target]['zoom_patch_sample'], \
        f'zoom {zoom_h} has a higher zoom_patch_sample than {zoom_target}'

    ts_start = sample_configs[zoom_h]['n_past_ts'] - sample_configs[zoom_target]['n_past_ts']
    ts_end = sample_configs[zoom_h]['n_future_ts'] - sample_configs[zoom_target]['n_future_ts']


    b, nv, nt, n = x_h.shape[:4]
    c = x_h.shape[4:]
    t_range = slice(ts_start, nt - ts_end)

    if sample_configs[zoom_h]['zoom_patch_sample'] == -1 and sample_configs[zoom_target]['zoom_patch_sample'] == -1:
        x_h_ = x_h.clone()
        x_h_[:, :, t_range] = x_s
        return x_h_ if batched else x_h_[0]
    
    else:
        base_exp = 0

        if sample_configs[zoom_h]['zoom_patch_sample'] == -1:
            zoom_patch_diff = sample_configs[zoom_target]['zoom_patch_sample']
            base_exp = 1
        else:
            zoom_patch_diff = sample_configs[zoom_target]['zoom_patch_sample'] - sample_configs[zoom_h]['zoom_patch_sample']

        n_patches = base**base_exp * 4**zoom_patch_diff
        patch_index = sample_configs[zoom_target]['patch_index'] % n_patches

        x_h_ = x_h.view(b, nv, nt, n_patches, -1, *c).clone()

        

        if isinstance(patch_index, int) or (isinstance(patch_index, torch.Tensor) and patch_index.numel() == 1):
            # Fill in the patch directly at index
            x_h_[:, :, t_range, patch_index] = x_s.unsqueeze(dim=3)
        else:
            # Broadcast and scatter each batch
            x_s = x_s.unsqueeze(dim=3)
            view_shape = (patch_index.shape[0], *([1] * (x_s.dim()-1)))
            expand_shape = list(x_s.shape)
            expand_shape[3] = 1
            patch_index_exp = patch_index.view(view_shape).expand(expand_shape)

            if add:
                x_h_[:, :, t_range] = x_h_[:, :, t_range].scatter_add(3, patch_index_exp, x_s)
            else:
                x_h_[:, :, t_range] = x_h_[:, :, t_range].scatter(3, patch_index_exp, x_s)


        x_h_ = x_h_.view(b, nv, nt, n, *c)

        return x_h_ if batched else x_h_[0]


def get_matching_time_patch(x_h, zoom_h, zoom_target, sample_configs, base=12):

    if zoom_h==zoom_target:
        return x_h
    
    batched = True

    if x_h.dim()<5:
        x_h = x_h.unsqueeze(dim=0)
        batched = False
    

    if sample_configs[zoom_h]['n_past_ts'] < sample_configs[zoom_target]['n_past_ts']:
        assert f'zoom {zoom_h} has a smaller number of past timesteps than {zoom_target}'

    if sample_configs[zoom_h]['n_future_ts'] < sample_configs[zoom_target]['n_future_ts']:
        assert f'zoom {zoom_h} has a smaller number of future timesteps than {zoom_target}'

    if sample_configs[zoom_h]['zoom_patch_sample'] > sample_configs[zoom_target]['zoom_patch_sample']:
        assert f'zoom {zoom_h} has a higher zoom_patch_sample than {zoom_target}'

    ts_start = sample_configs[zoom_h]['n_past_ts'] - sample_configs[zoom_target]['n_past_ts']
    ts_end = sample_configs[zoom_h]['n_future_ts'] - sample_configs[zoom_target]['n_future_ts']
    

    b,nv,nt,n = x_h.shape[:4]
    c = x_h.shape[4:]
    
    if sample_configs[zoom_h]['zoom_patch_sample'] == -1 and sample_configs[zoom_target]['zoom_patch_sample'] == -1:

        x_h = x_h[:, :, ts_start:(nt-ts_end)]
        return x_h if batched else x_h[0]
    
    else:
        base_exp = 0 
        if sample_configs[zoom_h]['zoom_patch_sample'] == -1:
            zoom_patch_diff = sample_configs[zoom_target]['zoom_patch_sample']
            base_exp = 1

        else:
            zoom_patch_diff = sample_configs[zoom_target]['zoom_patch_sample'] - sample_configs[zoom_h]['zoom_patch_sample']

        n_patches = base**base_exp * 4**zoom_patch_diff
        patch_index = sample_configs[zoom_target]['patch_index'] % n_patches

        

        x_h = x_h.view(b,nv,nt, n_patches, -1 ,*c)

        if isinstance(patch_index, int) or patch_index.numel()==1:
            x_h = x_h[:, :, ts_start:(nt-ts_end), patch_index]
        else:
            x_h = x_h[:, :, ts_start:(nt-ts_end)]
            view_shape = (patch_index.shape[0],*(1,)*(x_h.dim()-1))
            expand_shape = list(x_h.shape)
            expand_shape[3] = 1
            x_h = torch.gather(x_h, dim=3, index=patch_index.view(view_shape).expand(expand_shape))

        x_h = x_h.view(*x_h.shape[:3],-1,*c)

        if batched:
            return x_h
        else:
            return x_h[0]





def apply_zoom_diff(x_zooms: Dict[int, torch.Tensor], sample_configs: Dict):

    zooms = sorted(x_zooms.keys(),reverse=True)

    for k in range(len(zooms)-1):
        zoom = zooms[k]
        zoom_h = zooms[k+1]
        
        x = x_zooms[zoom]
        x_h = x_zooms[zoom_h]

        bvt = x.shape[:-2]
        x_h_patch = get_matching_time_patch(x_h, zoom_h, zoom, sample_configs)     

        x = x.view(*bvt, -1, 4**(zoom-zoom_h), x.shape[-1]) - x_h_patch.unsqueeze(dim=-2)
        x_zooms[zoom] = x.view(*bvt, -1, x.shape[-1])

    return x_zooms
    



def encode_zooms(x: torch.Tensor, in_zoom: int, out_zooms:int, apply_diff=True, mask: torch.Tensor=None, binarize_mask=False):
    bvt = x.shape[:-2]
    c = x.shape[-1]
    
    if mask is not None:
        mask = mask==0
    
    x_zooms = {}
    mask_zooms = {}
    for out_zoom in sorted(out_zooms, reverse=False):
        x = x.view(*bvt, -1, 4**(in_zoom - out_zoom), c)

        if mask is not None:
            mask = mask.view(*bvt,-1,4**(in_zoom - out_zoom),1)
            weight = mask.sum(dim=-2,keepdim=True)
            x_zoom = (x * mask).sum(dim=-2,keepdim=True)/weight
            x_zoom[weight == 0] = 0
            mask_zooms[out_zoom] = 1-(weight)/4**(in_zoom - out_zoom)
            if binarize_mask:
                mask_zooms[out_zoom][mask_zooms[out_zoom] < 1]=0
                mask_zooms[out_zoom] = mask_zooms[out_zoom].bool()
            
            mask_zooms[out_zoom] = mask_zooms[out_zoom].view(*bvt,-1)
        else:
            x_zoom = x.mean(dim=-2, keepdim=True)

        x_zooms[out_zoom] = x_zoom.view(*bvt,-1,c)
        
        if apply_diff:
            x = (x - x_zoom).view(*bvt,-1,c)
        else:
            x = x.view(*bvt,-1,c)

    return x_zooms, mask_zooms


def decode_zooms(x_zooms: dict, sample_configs, out_zoom):
    """
    Reconstructs the signal at a desired zoom level by summing contributions from multiple levels.

    Args:
        x_zooms (dict): Dictionary mapping zoom levels (int) to tensors of shape [..., N, C]
                        representing residuals or features at each zoom level.
        in_zoom (int): The coarsest (lowest resolution) zoom level available in x_zooms.
        out_zoom (int): The target (finest) zoom level to decode to.

    Returns:
        torch.Tensor: The reconstructed tensor at the desired out_zoom level, shape [..., 4**(in_zoom - out_zoom)*N, C]
    """
    x = 0
    for zoom in sorted(x_zooms.keys(),reverse=True):
        if zoom > out_zoom:
            continue  # Skip higher-resolution than target

        x_zoom = get_matching_time_patch(x_zooms[zoom],zoom,out_zoom, sample_configs)  # shape [..., N, C]
        up_factor = 4 ** (out_zoom - zoom)
        x_zoom = x_zoom.unsqueeze(-2).repeat_interleave(up_factor, dim=-2)

        x = x + x_zoom.view(*x_zoom.shape[:-3],-1,x_zoom.shape[-1])

    return {out_zoom: x}