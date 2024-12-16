import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import math

from scipy.interpolate import griddata

radius_earth= 6371

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
    
    if target=='torch':
        lons = torch.tensor(ds[lon].values)
        lats = torch.tensor(ds[lat].values)
        coords = torch.stack((lons, lats), dim=-1).float()
    else:
        lons = np.array(ds[lon].values)
        lats = np.array(ds[lat].values)
        coords = np.stack((lons, lats), axis=-1).astype(np.float32)

    return coords

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
    


def icon_get_adjacent_cell_indices(acoe:torch.tensor, eoc:torch.tensor, nh:int=5, coarsen_level:int=0):
    """
    Function to get neighboured cell indices 

    :param acoe: adjacent cells of edges tensor
    :param eoc: edge of cells tensor
    :param nh: number of neighbours
    :param coarsen_level: coarsen level if icon grid
    

    :returns: adjacent cell indices, duplicates mask
    """

    b = eoc.shape[-1]
    global_indices = torch.arange(b)

    nh1 = acoe.T[eoc.T].reshape(-1,4**coarsen_level,6)
    self_indices = global_indices.view(-1,4**coarsen_level)[:,0]
    self_indices = self_indices // 4**coarsen_level

    adjc_indices = nh1.view(nh1.shape[0],-1) // 4**coarsen_level

    adjc_unique = (adjc_indices).long().unique(dim=-1)

    is_self = adjc_unique - self_indices.view(-1,1) == 0

    adjc = adjc_unique[~is_self]

    adjc = adjc.reshape(self_indices.shape[0], -1)

    adjcs = [self_indices.view(-1,1), adjc]

    duplicates = [torch.zeros_like(adjcs[0], dtype=torch.bool), torch.zeros_like(adjcs[1], dtype=torch.bool)]
    
    b = adjc.shape[0]
    for k in range(1, nh):
        adjc_prev = adjcs[-1]

        adjc = adjcs[1][adjc_prev,:].view(b,-1)

        check_indices = torch.concat(adjcs, dim=-1).unsqueeze(dim=-2)

        is_prev = adjc.unsqueeze(dim=-1) - check_indices == 0
        is_prev = is_prev.sum(dim=-1) > 0

        is_removed = is_prev

        is_removed_count = is_removed.sum(dim=-1)
        
        unique, counts = is_removed_count.unique(return_counts=True)
        majority = unique[counts.argmax()]
        
        for minority in unique[unique!=majority]:

            where_minority = torch.where(is_removed_count==minority)[0]

            ind0, ind1 = torch.where(is_removed[where_minority])

            ind0 = ind0.reshape(len(where_minority),-1)[:,:minority-majority].reshape(-1)
            ind1 = ind1.reshape(len(where_minority),-1)[:,:minority-majority].reshape(-1)

            is_removed[where_minority[ind0], ind1] = False

        adjc = adjc[~is_removed]

        adjc = adjc.reshape(b, -1)
        
        if k > 1:
            counts = [] 
            uniques=[]
            for row in adjc:
                unique, count = row.unique(return_counts=True)
                uniques.append(unique)
                counts.append(len(unique))
        
            adjc = torch.nn.utils.rnn.pad_sequence(uniques, batch_first=True, padding_value=-1)
            duplicates_mask = adjc==-1
        else:
            duplicates_mask = torch.zeros_like(adjc)
            
        adjcs.append(adjc)
        duplicates.append(duplicates_mask)

    adjc = torch.concat(adjcs, dim=-1)
    duplicates = torch.concat(duplicates, dim=-1)

    return adjc, duplicates



def get_nearest_to_icon_recursive(c_t_global: torch.tensor, c_i: torch.tensor, level:int=7, global_indices_i: torch.tensor=None, nh: int=5, search_radius:int=5, periodic_fov:list=None):
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


   # _, n_sec_i, _ = c_i.shape
    n_target = c_t_global.shape[0]

    id_t = torch.arange(n_target)

    n_level = n_target // 4**level

    if level > 0:
        mid_points_corners = id_t.reshape(-1, 4, 4**(level-1))[:,1:,0]
        mid_points = id_t.reshape(-1, 4, 4**(level-1))[:,[0],0]
    else:
        mid_points_corners = id_t.reshape(-1,4)[:,1:].repeat_interleave(4, dim=0)
        mid_points = id_t.unsqueeze(dim=-1)

    # get radius
    c_t_ = c_t_global[mid_points_corners]
    c_t_m = c_t_global[mid_points]

    dist_corners = get_distance_angle(c_t_[:,:,0].unsqueeze(dim=-1),c_t_[:,:,1].unsqueeze(dim=-1), c_t_m[:,:,0].unsqueeze(dim=-2),c_t_m[:,:,1].unsqueeze(dim=-2))[0]
    dist_corners_max = search_radius*dist_corners.max(dim=1).values

    c_i_ = c_i

    c_t_m = c_t_m.reshape(c_i.shape[0], -1, 2)

    dist, phi = get_distance_angle(c_t_m[:,:,0].unsqueeze(dim=-1),c_t_m[:,:,1].unsqueeze(dim=-1), c_i_[:,:,0].unsqueeze(dim=-2), c_i_[:,:,1].unsqueeze(dim=-2), periodic_fov=periodic_fov)
    dist = dist.reshape(n_level, -1)
    phi = phi.reshape(n_level, -1)

    in_rad = dist <= dist_corners_max

    dist_values, indices_rel = dist.topk(in_rad.sum(dim=-1).max(), dim=-1, largest=False, sorted=True)
    

    if global_indices_i is None:
        global_indices = indices_rel
    else:
        global_indices = torch.gather(global_indices_i, index=indices_rel.reshape(global_indices_i.shape[0],-1), dim=-1)
        global_indices = global_indices.reshape(n_level,-1)
    
    if nh is not None:
        n_keep = torch.tensor([nh,dist_values.shape[1]]).min()
    else:
        n_keep = dist_values.shape[1]

    indices_keep = dist_values.topk(int(n_keep), dim=-1, largest=False, sorted=True)[1]

    dist_values = torch.gather(dist_values, index=indices_keep, dim=-1)
    in_rad = torch.gather(in_rad.reshape(n_level, -1), index=indices_keep, dim=-1)
    global_indices = torch.gather(global_indices, index=indices_keep, dim=-1)

    phi_values = torch.gather(phi, index=indices_keep, dim=-1)

    return global_indices, in_rad, (dist_values, phi_values)


def get_mapping_to_icon_grid(coords_icon: torch.tensor, coords_input: torch.tensor, search_radius:int=3, max_nh:int=10, lowest_level:int=0, periodic_fov=None)->dict:
    """
    Iterator of the get_nearest_to_icon_recursive function

    :param coords_icon: coordinates of icon grid
    :param coords_input: coordinates of input grid
    :param search_radius: Search radius in units of the grid distance
    :param max_nh: Maximum number of neighbours to have in output
    :param lowest_level: Lowest coarsen level (HR) of which to cacluate the neighbours
    :param periodic_fov: Optional if data is defined on a local patch with periodic boundary conditions
    

    :returns: grid mapping (dict)
    """
    if not isinstance(coords_icon, torch.Tensor):
        coords_icon = torch.tensor(coords_icon)

    if not isinstance(coords_input, torch.Tensor):
        coords_input = torch.tensor(coords_input)

    level_start = int(math.log(coords_icon.shape[0])/math.log(4))
    
    r = coords_icon.shape[-1]/4**level_start

    while math.floor(r)!=math.ceil(r):
        level_start -= 1
        r = coords_icon.shape[0]/4**level_start

    grid_mapping = []
    for k in range(level_start + 1 - lowest_level):
        level = level_start - (k)

        if level == lowest_level:
            nh = max_nh
        else:
            nh = None

        if k == 0:
            indices, in_rng, pos = get_nearest_to_icon_recursive(coords_icon, coords_input.unsqueeze(dim=0), level=level, nh=nh, search_radius=search_radius, periodic_fov=periodic_fov)
        else:
            indices, in_rng, pos = get_nearest_to_icon_recursive(coords_icon, coords_input[indices], level=level, global_indices_i=indices, nh=nh, search_radius=search_radius, periodic_fov=periodic_fov)

        grid_mapping.append({'level': level, 'indices': indices, 'pos': pos, 'in_rng_mask': in_rng}) 

    return grid_mapping


def get_nh_variable_mapping_icon(grid_icon:str|xr.Dataset,
                                grid_types_icon:str,
                                grid:str|xr.Dataset, 
                                grid_types:list, 
                                search_radius:int=3, 
                                max_nh:int=10, 
                                lowest_level:int=0, 
                                coords_icon:torch.tensor=None, 
                                scale_input:float=1., 
                                periodic_fov:list=None):
    """
    Generate a mapping between ICON grid coordinates and another grid's coordinates 
    by finding nearest neighbors within a specified radius.

    Parameters:
    ----------
    grid_file_icon : str
        Path to the ICON grid file containing the coordinate data for the ICON grid.
    grid_types_icon : list of str
        Types of grids within the ICON grid file for which mappings are needed.
    grid_file : str
        Path to the secondary grid file containing the coordinate data for mapping.
    grid_types : list of str
        Types of grids within the secondary grid file for which mappings are needed.
    search_radius : int, optional
        Maximum radius to search for neighbors between the grids (default is 3).
    max_nh : int, optional
        Maximum number of neighbors to find for each grid point (default is 10).
    lowest_level : int, optional
        Lowest level to consider in the mapping process (default is 0).
    coords_icon : torch.Tensor, optional
        Coordinates of the ICON grid as a tensor. If None, coordinates are fetched based on grid type.
    scale_input : float, optional
        Scaling factor to apply to secondary grid coordinates (default is 1.0).
    periodic_fov : tuple of float, optional
        Boundary conditions for periodic fields of view.

    Returns:
    -------
    mapping_icon : dict
        Dictionary mapping ICON grid types to the nearest neighbor indices in the secondary grid.
    in_range : dict
        Dictionary indicating which points are within range of the ICON grid for each grid type.
    """
    if isinstance(grid_icon, str):
        grid_icon = xr.open_dataset(grid_icon)
    
    if isinstance(grid, str):
        grid = xr.open_dataset(grid)
    
    mapping_icon = {}
    in_range = {}
    for grid_type_icon in grid_types_icon:
        
        if coords_icon is None:
            coords_icon = get_coords_as_tensor(grid_icon, grid_type=grid_type_icon)

        mapping_grid_type = {}
        in_range_grid_type = {}

        for grid_type in grid_types:
            coords_input = get_coords_as_tensor(grid, grid_type=grid_type)
            if scale_input > 1:
                coords_input = scale_coordinates(coords_input, scale_input)

            if grid_icon.attrs['uuidOfHGrid'] == grid.attrs['uuidOfHGrid']:
                indices = torch.arange(coords_icon.shape[1]).view(-1,1)
                in_rng_mask = torch.ones_like(indices, dtype=torch.bool)

            else:
                mapping = get_mapping_to_icon_grid(
                    coords_icon,
                    coords_input,
                    search_radius=search_radius,
                    max_nh=max_nh,
                    lowest_level=lowest_level,
                    periodic_fov=periodic_fov)
                                
                indices = mapping[-1]['indices']
                in_rng_mask = mapping[-1]['in_rng_mask']

            mapping_grid_type[grid_type] = indices
            in_range_grid_type[grid_type] = in_rng_mask

        mapping_icon[grid_type_icon] = mapping_grid_type
        in_range[grid_type_icon] = in_range_grid_type

    return mapping_icon, in_range


def icon_grid_to_mgrid(grid_file:str, n_grid_levels:int, clon_fov:list=None, clat_fov:list=None, nh:int=0, extension:float=0.1)->list:
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
    grid = xr.open_dataset(grid_file)
    
    clon =  torch.tensor(grid.clon.values)
    clat =  torch.tensor(grid.clat.values)

    eoc = torch.tensor(grid.edge_of_cell.values - 1)
    acoe = torch.tensor(grid.adjacent_cell_of_edge.values - 1)

    global_indices = torch.arange(len(grid.clon))
    cell_coords_global = get_coords_as_tensor(grid, lon='clon', lat='clat')
    #coordinates per level


    if clon_fov is not None or clat_fov is not None:

        indices_max_lvl = torch.arange(len(grid.clon)).reshape(-1,4**int(n_grid_levels))
       
        clon_max_lvl = clon[indices_max_lvl[:,0]]
        clat_max_lvl = clat[indices_max_lvl[:,0]]

        keep_indices_clon = torch.ones_like(clon_max_lvl, dtype=bool)
        keep_indices_clat = torch.ones_like(clat_max_lvl, dtype=bool)

        if clon_fov is not None:
            fov_ext = (clon_fov[1] - clon_fov[0])*extension/2
            keep_indices_clon = torch.logical_and(clon_max_lvl >= clon_fov[0]-fov_ext, clon_max_lvl <= clon_fov[1]+fov_ext)

        if clat_fov is not None:   
            fov_ext = (clat_fov[1] - clat_fov[0])*extension/2
            keep_indices_clat = torch.logical_and(clat_max_lvl >= clat_fov[0]-fov_ext, clat_max_lvl <= clat_fov[1]+fov_ext)

        keep_grid_indices_max_lvl = torch.logical_and(keep_indices_clon, keep_indices_clat)

        keep_grid_indices = keep_grid_indices_max_lvl.view(-1,1).repeat_interleave(indices_max_lvl.shape[-1], dim=1)

    else:
        keep_grid_indices = torch.ones_like(global_indices, dtype=bool)


    global_levels = []
    grids = []
    for grid_level_idx in range(n_grid_levels):

        global_level = grid_level_idx
        global_levels.append(global_level)

        adjc, adjc_mask_duplicates = icon_get_adjacent_cell_indices(acoe, eoc, nh=nh, coarsen_level=global_level)
        adjc_mask_duplicates = adjc_mask_duplicates==False

        keep_grid_indices_lvl = keep_grid_indices.reshape(-1,4**int(global_level))[:,0]

        indices_lvl = global_indices.reshape(-1,4**int(global_level))[:,0]

        indices_lvl = indices_lvl[keep_grid_indices_lvl]

        adjc_lvl = adjc[keep_grid_indices_lvl]
        adjc_mask_duplicates = adjc_mask_duplicates[keep_grid_indices_lvl]
        
        indices_in_fov = torch.where(keep_grid_indices_lvl)[0]
        index_shift = torch.concat((torch.tensor(0).view(1),indices_in_fov)).diff()
        wh = torch.where(index_shift>1)[0]

        for l, k in enumerate(wh):
            idx_shift = index_shift[k] if k==0 else index_shift[k]-1
            adjc_lvl[adjc_lvl >= adjc_lvl[k,0]] = adjc_lvl[adjc_lvl >= adjc_lvl[k,0]] - idx_shift

        cell_coords_lvl = cell_coords_global.reshape(-1,4**global_level,2)[:,0,:]

        if clon_fov is not None:
            fov_ext = (clon_fov[1] - clon_fov[0])*extension/2
            adjc_mask_lon = torch.logical_and(cell_coords_lvl[adjc[keep_grid_indices_lvl],0] >= clon_fov[0]-fov_ext, cell_coords_lvl[adjc[keep_grid_indices_lvl],0] <= clon_fov[1]+fov_ext)

            fov_ext = (clat_fov[1] - clat_fov[0])*extension/2
            adjc_mask_lat = torch.logical_and(cell_coords_lvl[adjc[keep_grid_indices_lvl],1] >= clat_fov[0]-fov_ext, cell_coords_lvl[adjc[keep_grid_indices_lvl],1] <= clat_fov[1]+fov_ext)

            adjc_mask = torch.logical_and(adjc_mask_lon, adjc_mask_lat)
            adjc_mask = torch.logical_and(adjc_mask, adjc_mask_duplicates)
        
        else:
            adjc_mask = adjc_mask_duplicates
        
        adjc_mask[adjc_lvl>(adjc_lvl.shape[0]-1)]=False

        r,c = torch.where(adjc_mask==False)
        adjc_lvl[r,c] = adjc_lvl[r,0] 

        grid_lvl = {
            'coords': cell_coords_global[indices_lvl,:],
            'adjc': adjc,
            'adjc_lvl': adjc_lvl,
            'adjc_mask': adjc_mask,
            "global_level": global_level
        }

        grids.append(grid_lvl)

    return grids


def sequenize(tensor, max_seq_level, seq_dim=1):
    
    seq_level = min([get_max_seq_level(tensor, seq_dim), max_seq_level])
    
    if seq_dim==1:
        if tensor.dim()==2:
            tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level))
        elif tensor.dim()==3:
            tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-1])
        elif tensor.dim()==4:
            tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-2], tensor.shape[-1])
        elif tensor.dim()==5:
            tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

    elif seq_dim==2:
        if tensor.dim()==3:
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], -1, 4**(seq_level))
        elif tensor.dim()==4:
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], -1, 4**(seq_level), tensor.shape[-1])
        elif tensor.dim()==5:
            tensor = tensor.view(tensor.shape[0], tensor.shape[1], -1, 4**(seq_level), tensor.shape[-2], tensor.shape[-1])    

    return tensor


def get_max_seq_level(tensor, seq_dim=1):
    seq_len = tensor.shape[seq_dim]
    max_seq_level_seq = int(math.log(seq_len)/math.log(4))
    return max_seq_level_seq