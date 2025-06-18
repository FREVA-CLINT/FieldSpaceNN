import torch
import xarray as xr
import math

from .grid_utils import get_coords_as_tensor,get_distance_angle,scale_coordinates

def icon_get_adjacent_cell_indices(acoe:torch.tensor, eoc:torch.tensor, nh:int=5, zoom_rel:int=2):
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

    nh1 = acoe.T[eoc.T].reshape(-1,4**zoom_rel,eoc.shape[0]*acoe.shape[0])
    self_indices = global_indices.view(-1,4**zoom_rel)[:,0]
    self_indices = self_indices // 4**zoom_rel

    adjc_indices = nh1.view(nh1.shape[0],-1) // 4**zoom_rel

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
    global_indices = torch.gather(global_indices, index=indices_keep, dim=-1)

    in_range_unique = dist_values <= dist_corners.max(dim=1).values

    phi_values = torch.gather(phi, index=indices_keep, dim=-1)

    return global_indices, in_range_unique, (dist_values, phi_values)


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
    positions = {}
    for grid_type_icon in grid_types_icon:
        
        if coords_icon is None:
            coords_icon = get_coords_as_tensor(grid_icon, grid_type=grid_type_icon)

        mapping_grid_type = {}
        in_range_grid_type = {}
        positions_grid_type = {}

        for grid_type in grid_types:
            coords_input = get_coords_as_tensor(grid, grid_type=grid_type)
            if scale_input > 1:
                coords_input = scale_coordinates(coords_input, scale_input)

            if 'uuidOfHGrid' in grid.attrs and grid_icon.attrs['uuidOfHGrid'] == grid.attrs['uuidOfHGrid']:
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
                pos = mapping[-1]['pos']

            mapping_grid_type[grid_type] = indices
            in_range_grid_type[grid_type] = in_rng_mask
            positions_grid_type[grid_type] = pos

        mapping_icon[grid_type_icon] = mapping_grid_type
        in_range[grid_type_icon] = in_range_grid_type
        positions[grid_type_icon] = positions_grid_type

    return mapping_icon, in_range, positions


def icon_grid_to_mgrid(grid_file:str, clon_fov:list=None, clat_fov:list=None, nh:int=0, extension:float=0.1)->list:
    """
    Convert an ICON grid to a multi-level grid by calculating coordinates and adjacency
    information at each grid level.

    Parameters:
    ----------
    grid : xarray.Dataset
        The ICON grid dataset containing coordinates and adjacency information.
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

    #eoc = torch.tensor(grid.edge_of_cell.values - 1)
    #acoe = torch.tensor(grid.adjacent_cell_of_edge.values - 1)
    voc = torch.tensor(grid.vertex_of_cell.values - 1)
    cov = torch.tensor(grid.cells_of_vertex.values - 1)

    indices = torch.arange(len(grid.clon))
    cell_coords_global = get_coords_as_tensor(grid, lon='clon', lat='clat')
    #coordinates per level

    zoom_max = int(math.log(len(clon) // 5, 4))


    zooms = []
    grids = []
    for zoom in range(zoom_max + 1):

        zooms.append(zoom)

        if zoom==0:
            adjc, adjc_mask = get_nh_zoom_0(clon,clat,nh=nh)
        else:
            adjc, adjc_mask = icon_get_adjacent_cell_indices(cov, voc, nh=nh, zoom_rel = zoom_max - zoom)
            adjc_mask = adjc_mask == 1

        adjc_mask[adjc>(adjc.shape[0]-1)]=True

        r,c = torch.where(adjc_mask)
        adjc[r,c] = adjc[r,0]

        grid_zoom = {
            'coords': cell_coords_global.view(-1,4**(zoom_max-zoom),2)[:,0],
            'adjc': adjc,
            'adjc_mask': adjc_mask,
            "zoom": zoom
        }

        grids.append(grid_zoom)

    return grids


def get_nh_zoom_0(clon, clat, nh=1):

    n_nh = {1: 4,
            2: 5,
            3: 5}
    
    clon = clon.view(5,-1)[:,[0]]
    clat = clat.view(5,-1)[:,[0]]

    dist, _ = get_distance_angle(clon.transpose(0,1), clat.transpose(0,1), clon, clat, base='polar')

    _, adjc = torch.topk(dist, k=n_nh[nh], largest=False)

    return adjc, torch.zeros_like(adjc, dtype=bool)
