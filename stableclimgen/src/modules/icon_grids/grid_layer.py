import torch
import torch.nn as nn
from typing import List

from ..transformer import transformer_modules as helpers

from ...utils.grid_utils_icon import get_distance_angle,sequenize


def get_nh_indices(adjc_global: torch.Tensor, global_level: int, global_cell_indices: torch.Tensor = None, local_cell_indices: torch.Tensor = None) -> tuple:
    """
    Calculate the neighborhood indices and corresponding mask for given cell indices.

    :param adjc_global: The global adjacency tensor.
    :param global_level: The global hierarchical level.
    :param global_cell_indices: Indices of cells at the global level. Defaults to None.
    :param local_cell_indices: Indices of cells at the local level. Defaults to None.
    :return: A tuple containing neighborhood indices and a mask.
    """
    if global_cell_indices is not None:
        # Derive local cell indices from global cell indices and level
        local_cell_indices = global_cell_indices // 4**global_level

    # Get neighborhood indices and mask using helper function
    local_cell_indices_nh, mask = helpers.get_nh_of_batch_indices(local_cell_indices, adjc_global)

    return local_cell_indices_nh, mask

    
def gather_nh_data(x: torch.Tensor, local_cell_indices_nh: torch.Tensor, batch_sample_indices: torch.Tensor, sampled_level: int, global_level: int) -> torch.Tensor:
    """
    Gather neighborhood data from the input tensor based on neighborhood indices.

    :param x: Input data tensor with dimensions [batch, cells, vertices, features].
    :param local_cell_indices_nh: Neighborhood indices for local cells.
    :param batch_sample_indices: Indices for batch sampling.
    :param sampled_level: The level at which data is sampled.
    :param global_level: The global hierarchical level.
    :return: A tensor with gathered neighborhood data.
    """
    # Ensure input tensor x has at least 4 dimensions
    if x.dim()<4:
        x = x.unsqueeze(dim=-1)

    # Extract shape parameters
    b, n, nv, e = x.shape
    nh = local_cell_indices_nh.shape[-1]

    # Compute batch indices adjusted for hierarchical levels
    local_cell_indices_nh_batch = local_cell_indices_nh - (batch_sample_indices * 4**(sampled_level - global_level)).view(-1, 1, 1)

    # Gather data based on neighborhood indices and reshape

    x = torch.gather(
        x.view(b, -1, nv, e), 1, 
        index=local_cell_indices_nh_batch.view(b, -1, 1, 1).repeat(1, 1, nv, e)
    ).view(b, n, nh, nv, e)

    return x


def get_relative_positions(coords1, coords2, polar=False, periodic_fov=None):
    
    if coords2.dim() > coords1.dim():
        coords1 = coords1.unsqueeze(dim=-1)

    if coords1.dim() > coords2.dim():
        coords2 = coords2.unsqueeze(dim=-2)

    if coords1.dim() == coords2.dim():
        coords1 = coords1.unsqueeze(dim=-1)
        coords2 = coords2.unsqueeze(dim=-2)

    distances, phis = get_distance_angle(coords1[0], coords1[1], coords2[0], coords2[1], base="polar" if polar else "cartesian", periodic_fov=periodic_fov)

    return distances.float(), phis.float()


class GridLayer(nn.Module):
    """
    A neural network module representing a grid layer with specific functionalities
    like coordinate transformations, neighborhood gathering, and relative positioning.

    Attributes:
        global_level (int): The global hierarchical level.
        coord_system (str): The coordinate system, e.g., "polar".
        periodic_fov (Any): The periodic field of view.
        coordinates (torch.Tensor): The coordinates of the grid.
        adjc (torch.Tensor): Adjacency tensor for the grid.
        adjc_mask (torch.Tensor): Mask indicating adjacency relations.
        fov_mask (torch.Tensor): Mask for field of view.
        min_dist (torch.Tensor): Minimum distance in the relative coordinates.
        max_dist (torch.Tensor): Maximum distance in the relative coordinates.
        mean_dist (torch.Tensor): Mean distance in the relative coordinates.
        median_dist (torch.Tensor): Median distance in the relative coordinates.
    """
    
    def __init__(self, global_level: int, adjc: torch.Tensor, adjc_mask: torch.Tensor, coordinates: torch.Tensor, coord_system: str = "polar", periodic_fov=None) -> None:
        """
        Initializes the GridLayer with given parameters.

        :param global_level: The global level for hierarchy.
        :param adjc: The adjacency tensor.
        :param adjc_mask: Mask for the adjacency tensor.
        :param coordinates: The coordinates of grid points.
        :param coord_system: The coordinate system. Defaults to "polar".
        :param periodic_fov: The periodic field of view. Defaults to None.
        """
        super().__init__()

        # Initialize attributes
        self.global_level = global_level
        self.coord_system = coord_system
        self.periodic_fov = periodic_fov

        # Register buffers for coordinates and adjacency information
        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("adjc", adjc, persistent=False)
        # Create mask where adjacency is false
        self.register_buffer("adjc_mask", adjc_mask == False, persistent=False)
        # Mask for the field of view
        self.register_buffer("fov_mask", ((adjc_mask == False).sum(dim=-1) == adjc_mask.shape[1]).view(-1, 1), persistent=False)

        # Sample distances for statistical analysis
        n_samples = torch.min(torch.tensor([self.adjc.shape[0] - 1, 500]))
        nh_samples = self.adjc[:n_samples]
        coords_nh = self.get_coordinates_from_grid_indices(nh_samples)
        # Calculate relative distances
        coords_nh = coords_nh[:,:,0]
        coords_lon_1, coords_lat_1 = coords_nh[:,:,0].unsqueeze(dim=-2), coords_nh[:,:,1].unsqueeze(dim=-2)
        coords_lon_2, coords_lat_2 = coords_nh[:,:,0].unsqueeze(dim=-1), coords_nh[:,:,1].unsqueeze(dim=-1)

        dists, _ = get_distance_angle(coords_lon_1, coords_lat_1, coords_lon_2, coords_lat_2, base="polar", rotate_coords=True)
        dists_lon, dists_lat = get_distance_angle(coords_lon_1, coords_lat_1, coords_lon_2, coords_lat_2, base="cartesian", rotate_coords=True)

        self.nh_dist = dists[:,0,1].mean()
        self.nh_dist_lon = dists_lon[:,0,1].abs().mean()
        self.nh_dist_lat = dists_lat[:,0,1].abs().mean()
        # Compute distance statistics
        self.dist_quantiles = dists[dists > 1e-10].quantile(torch.linspace(0.01,0.99,20))

        self.min_dist = dists[dists > 1e-6].min()
        self.max_dist = dists[dists > 1e-10].max()
        self.mean_dist = dists[dists > 1e-10].mean()
        self.median_dist = dists[dists > 1e-10].median()

    def get_nh(self, x: torch.Tensor, local_indices: torch.Tensor=None, sample_dict: dict=None, mask: torch.Tensor = None) -> tuple:
        """
        Get the neighborhood data, mask, and coordinates.

        :param x: Input data tensor.
        :param local_indices: Indices for local neighborhoods.
        :param sample_dict: Dictionary with sample information.
        :param relative_coordinates: Whether to use relative coordinates. Defaults to True.
        :param coord_system: Coordinate system to use. Defaults to None.
        :param mask: Optional mask tensor. Defaults to None.
        :return: A tuple containing neighborhood data, mask, and coordinates.
        """
        if local_indices is None or sample_dict is None:
            x = x[:,self.adjc]
            if mask is not None:
                mask = mask[:,self.adjc]
            return x, mask

        # Get neighborhood indices and adjacency mask
        indices_nh, adjc_mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))
        
        x_shape = x.shape
        x = x.reshape(*x_shape[:3],-1)
        # Gather neighborhood data
        x = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))

        if mask is not None:
            # Combine provided mask with adjacency mask
            mask = gather_nh_data(mask, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))
            mask = torch.logical_or(mask.view(x.shape[:-1]), adjc_mask.unsqueeze(dim=-1))
        else:
            # Use adjacency mask if no mask is provided
            mask = adjc_mask.unsqueeze(dim=-1).repeat_interleave(x.shape[-2], dim=-1)
        
        x = x.view(*x_shape[:2],x.shape[2],*x_shape[2:])

        return x, mask

    def get_coordinates_from_grid_indices(self, local_indices: torch.Tensor, nh:bool=False) -> torch.Tensor:
        """
        Retrieve coordinates based on grid indices.

        :param local_indices: Indices for grid points.
        :return: Coordinates corresponding to the indices.
        """
        if nh:
            local_indices = self.adjc[local_indices] if local_indices is not None else self.adjc.unsqueeze(dim=0)
            return self.coordinates[local_indices]
        
        elif local_indices is None:
            return self.coordinates.unsqueeze(dim=1).unsqueeze(dim=0)
        
        else:
            local_indices = local_indices.unsqueeze(dim=-1) if local_indices.dim()<3 else local_indices
            return self.coordinates[local_indices]

    def get_relative_coordinates_from_grid_indices(self, local_indices: torch.Tensor, coords: torch.Tensor = None, coord_system: str = None) -> torch.Tensor:
        """
        Get relative coordinates from grid indices.

        :param local_indices: Indices for local points.
        :param coords: Precomputed coordinates. Defaults to None.
        :param coord_system: Coordinate system to use. Defaults to None.
        :return: Relative coordinates.
        """
        if coord_system is None:
            coord_system = self.coord_system

        if coords is None:
            coords = self.get_coordinates_from_grid_indices(local_indices)

        # Compute relative distances and angles
        coords_rel = get_distance_angle(
            coords[0, :, :, [0]], coords[1, :, :, [0]], coords[0], coords[1],
            base=coord_system, periodic_fov=self.periodic_fov
        )

        return coords_rel

    def get_relative_coordinates_cross(self, local_indices: torch.Tensor, coords: torch.Tensor, coord_system: str = None) -> torch.Tensor:
        """
        Get cross relative coordinates between reference and provided points.

        :param local_indices: Reference indices.
        :param coords: Provided coordinates.
        :param coord_system: Coordinate system to use. Defaults to None.
        :return: Cross relative coordinates.
        """
        if coord_system is None:
            coord_system = self.coord_system

        # Get reference coordinates
        coords_ref = self.get_coordinates_from_grid_indices(local_indices)

        if coords_ref.dim() < 4:
            coords_ref = coords_ref.unsqueeze(dim=-1)

        # Compute cross relative distances and angles
        coords_rel = get_distance_angle(
            coords_ref[0, :, :, [0]], coords_ref[1, :, :, [0]], coords[0], coords[1],
            base=coord_system, periodic_fov=self.periodic_fov
        )

        return coords_rel

    def get_sections(self, x: torch.Tensor, local_indices: torch.Tensor, section_level: int = 1, relative_coordinates: bool = True, return_indices: bool = True, coord_system: str = None) -> tuple:
        """
        Divide the input into sections based on hierarchical levels.

        :param x: Input tensor.
        :param local_indices: Local indices for sections.
        :param section_level: Level of the section. Defaults to 1.
        :param relative_coordinates: Use relative coordinates. Defaults to True.
        :param return_indices: Return indices or not. Defaults to True.
        :param coord_system: Coordinate system to use. Defaults to None.
        :return: A tuple containing sectioned data, mask, coordinates, and indices (optional).
        """
        # Sequenize indices and data
        indices = sequenize(local_indices, max_seq_level=section_level)
        x = sequenize(x, max_seq_level=section_level)

        # Get coordinates
        if relative_coordinates:
            coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
        else:
            coords = self.get_coordinates_from_grid_indices(indices)

        # Field of view mask
        mask = self.fov_mask[indices]

        if return_indices:
            return x, mask, coords, indices
        else:
            return x, mask, coords


class MultiRelativeCoordinateManager(nn.Module):

    def __init__(self,  
                grid_layers: List[GridLayer], 
                rotate_coord_system=True) -> None:
                
        super().__init__()

        self.rotate_coord_system = rotate_coord_system
        self.rcms = nn.ModuleDict()

        global_levels = [int(global_level) for global_level in grid_layers.keys()]
        nh_dists = [grid_layer.nh_dist for grid_layer in grid_layers.values()]

        self.nh_dists = dict(zip(global_levels, nh_dists))
        self.grid_layers = grid_layers
        
    
    def register_rcm(self,
                     global_level_in, 
                     global_level_out, 
                     nh_in,
                     precompute,
                     coord_system,
                     ref='out'):
        
        global_level_in_str = str(global_level_in)
        global_level_out_str = str(global_level_out)

        if global_level_in_str not in self.rcms.keys():
            self.rcms[global_level_in_str] = nn.ModuleDict()
            
      
        if global_level_out_str not in self.rcms[global_level_in_str].keys():
            self.rcms[global_level_in_str][global_level_out_str] = RelativeCoordinateManager(
                    grid_layer_in=self.grid_layers[global_level_in_str],
                    grid_layer_out=self.grid_layers[global_level_out_str],
                    nh_in=nh_in,
                    precompute=precompute,
                    coord_system=coord_system,
                    rotate_coord_system=self.rotate_coord_system,
                    ref=ref
                )
            

    def forward(self, global_level_in, global_level_out, indices_sample=None, x=None, mask=None):
        indices_in  = indices_sample["indices_layers"][global_level_in] if indices_sample is not None else None
        indices_out = indices_sample["indices_layers"][global_level_out] if indices_sample is not None else None
        
        rcm = self.rcms[str(global_level_in)][str(global_level_out)]
        coordinates_rel = rcm(indices_in=indices_in, indices_out=indices_out, sample_dict=indices_sample)

        if x is None:
            return coordinates_rel
        else:
            if rcm.nh_in:
                x, mask = self.grid_layers[str(global_level_in)].get_nh(x, indices_in, indices_sample, mask=mask)
            else:
                x = x.unsqueeze(dim=2)
                if mask is not None:
                    mask = mask.unsqueeze(dim=2)
            
            return coordinates_rel, x, mask
    

class MultiStepRelativeCoordinateManager(nn.Module):
    def __init__(self,  
                grid_layers: List[GridLayer], 
                nh_up:bool= False,
                nh_down:bool= False,
                precompute:bool=True,
                coord_system:str='polar',
                rotate_coord_system=True,
                ref='out') -> None:
                
        super().__init__()

        self.managers_up = nn.ModuleList()
        self.managers_down = nn.ModuleList()
        self.nh_up = nh_up
        self.nh_down = nh_down

        self.nh_dists = [grid_layer.nh_dist for grid_layer in grid_layers]

        global_levels = list(grid_layers.keys())
        
        self.register_buffer('global_levels', torch.tensor(global_levels))
        
        for idx in range(1, len(global_levels)):
            global_level_in = global_levels[idx - 1]
            global_level_out = global_levels[idx]
            
            self.managers_up.append(RelativeCoordinateManager(
                grid_layer_in=grid_layers[global_level_in],
                grid_layer_out=grid_layers[global_level_out],
                nh_in=nh_up,
                precompute=precompute,
                coord_system=coord_system,
                rotate_coord_system=rotate_coord_system,
                ref=ref
            ))
            self.managers_down.append(manager = RelativeCoordinateManager(
                grid_layer_in=grid_layers[global_level_out],
                grid_layer_out=grid_layers[global_level_in],
                nh_in=nh_down,
                precompute=precompute,
                coord_system=coord_system,
                rotate_coord_system=rotate_coord_system,
                ref=ref
            ))

    def get_manager_from_levels(self, global_level_in, global_level_out):
        index = torch.where(self.global_levels == global_level_in)

        if global_level_in-global_level_out > 0:
            return self.managers_up(index)
        else:
            return self.managers_down(index)

    
    def forward(self, global_level_in, global_level_out, indices_sample=None):
        indices_in  = indices_sample["indices_layers"][global_level_in] if indices_sample is not None else None
        indices_out = indices_sample["indices_layers"][global_level_out] if indices_sample is not None else None

        coordinates_rel =self.get_manager_from_levels(global_level_in, global_level_out)(indices_in=indices_in, indices_out=indices_out, sample_dict=indices_sample)

        return coordinates_rel

class RelativeCoordinateManager(nn.Module):
    def __init__(self,  
                grid_layer_in:GridLayer, 
                grid_layer_out: GridLayer,
                nh_in:bool= False,
                seq_lvl:int = -1,
                precompute:bool=False,
                coord_system:str='polar',
                rotate_coord_system=True,
                ref='out') -> None:
                
        super().__init__()

        self.grid_layer_in = grid_layer_in
        self.grid_layer_out = grid_layer_out
        self.nh_in = nh_in
        self.seq_lvl = seq_lvl
        self.rotate_coord_system = rotate_coord_system

        self.coord_system = coord_system
        self.precomputed = precompute
        self.ref = ref

        if precompute:
            coordinates_rel = self.compute_rel_coordinates(
                indices_in=torch.arange(grid_layer_in.coordinates.shape[0]).view(1,-1),
                indices_out=torch.arange(grid_layer_out.coordinates.shape[0]).view(1,-1), nh_in=nh_in)
            
            self.register_buffer("coordinates_rel", torch.stack(coordinates_rel, dim=-1).squeeze(dim=1), persistent=False)
            


    def compute_rel_coordinates(self, indices_in=None, indices_out=None, coordinates_in=None, coordinates_out=None, sample_dict=None, nh_in=False):

        if coordinates_in is None:
            coordinates_in = self.grid_layer_in.get_coordinates_from_grid_indices(indices_in, nh=nh_in)

        elif coordinates_in is not None and nh_in:
            coordinates_in,_ = self.grid_layer_in.get_nh(coordinates_in, indices_in, sample_dict=sample_dict)

        if coordinates_out is None:
            coordinates_out = self.grid_layer_out.get_coordinates_from_grid_indices(indices_out)

        seq_dim_out_total = coordinates_out.shape[1]
        if coordinates_out.shape[1] > coordinates_in.shape[1]:
            coordinates_out = coordinates_out.view(coordinates_out.shape[0],coordinates_in.shape[1],-1,2)

        if self.seq_lvl != -1:
            coordinates_out = sequenize(coordinates_out, max_seq_level=self.seq_lvl)[:,:,[0]]
            coordinates_in = sequenize(coordinates_in, max_seq_level=self.seq_lvl)

        b, seq_dim_in, n_nh_in = coordinates_in.shape[:3]
        _, seq_dim_out, _ = coordinates_out.shape[:3]

      
        coordinates_in = coordinates_in.view(b, seq_dim_out, -1, 2)
        coordinates_out = coordinates_out.view(b, seq_dim_out, -1, 2)

        if self.ref =='out':
            coordinates_in = coordinates_in.unsqueeze(dim=-3)
            coordinates_out = coordinates_out.unsqueeze(dim=-2)

            coordinates_rel = get_distance_angle(
                                    coordinates_out[...,0], coordinates_out[...,1], 
                                    coordinates_in[...,0], coordinates_in[...,1], 
                                    base=self.coord_system, periodic_fov=None,
                                    rotate_coords=self.rotate_coord_system
                                )
        else:
            coordinates_out = coordinates_out.unsqueeze(dim=-2)
            coordinates_in = coordinates_in.unsqueeze(dim=-3)

            coordinates_rel = get_distance_angle(
                                    coordinates_in[...,0], coordinates_in[...,1], 
                                    coordinates_out[...,0], coordinates_out[...,1], 
                                    base=self.coord_system, periodic_fov=None,
                                    rotate_coords=self.rotate_coord_system
                                )

        if self.seq_lvl == -1:
            coordinates_rel = (coordinates_rel[0].view(b, seq_dim_out_total,  -1, n_nh_in),
                            coordinates_rel[1].view(b, seq_dim_out_total,  -1, n_nh_in))
        else:
            coordinates_rel = (coordinates_rel[0].view(b, -1, 1),
                            coordinates_rel[1].view(b, -1, 1))
        
        return coordinates_rel

    def forward(self, indices_in=None, indices_out=None, coordinates_in=None, coordinates_out=None, sample_dict=None):
        
        if not self.precomputed:
            coordinates_rel = self.compute_rel_coordinates(indices_in=indices_in,
                                         indices_out=indices_out,
                                         coordinates_in=coordinates_in,
                                         coordinates_out=coordinates_out,
                                         sample_dict=sample_dict,
                                         nh_in = self.nh_in)
        else:
          
            coordinates_rel = self.coordinates_rel

            c_shape = coordinates_rel.shape

            if indices_out is not None:
                coordinates_rel = coordinates_rel[:,indices_out].view(indices_out.shape[0],-1,*c_shape[2:])
            else:
                coordinates_rel = coordinates_rel.view(1,-1,*c_shape[2:])

            coordinates_rel = (coordinates_rel[...,0],
                           coordinates_rel[...,1])

        return coordinates_rel



import torch.nn as nn

def get_density_map(grid_dist_output, dists, mask_value=1e6, power=2):

    area_ref = grid_dist_output**power
    
    dists_weights = 1/((dists < mask_value).sum(dim=-2, keepdim=True)+1e-10)

    dists[dists >= mask_value] = 0

    area_m = ((dists*dists_weights).sum(dim=-2,keepdim=True))**power

    area_m[area_m==0] = mask_value

    density = (area_ref/area_m)

    return density


def get_interpolation(x: torch.tensor, 
                      mask: torch.tensor, 
                      grid_layer_search,
                      cutoff_dist: float,
                      dist: torch.tensor,
                      n_nh: int=3, 
                      power:int=2, 
                      mask_value:int=1e6, 
                      indices_sample: dict=None):
    x = x.clone()
    b,n,nh,nv,f = x.shape

    n_l = dist.shape[1]
    l = n // n_l

    x = x.view(b,n_l,-1)
    mask = mask.view(b,n_l,-1) if mask is not None else None

    local_indices = indices_sample['indices_layers'][int(grid_layer_search.global_level)] if indices_sample is not None else None
    x_nh, mask_nh = grid_layer_search.get_nh(x, local_indices=local_indices, sample_dict=indices_sample, mask = mask) 

    x_nh = x_nh.view(b,n_l,-1,nv,f)
    mask_nh = mask_nh.view(b,n_l,-1,nv)

    n = dist.shape[1]
    
    dist_ = dist.unsqueeze(dim=-1) + (mask_nh.unsqueeze(dim=2) * mask_value)

    dist_vals, indices = torch.topk(dist_, n_nh, dim=-2, largest=False)

    indices_offset = torch.arange(indices.shape[1], device=indices.device)
    offset = dist_.shape[-2]

    indices = indices + (indices_offset*offset).view(1,indices.shape[1],1,1,1)

    indices = indices.view(b, -1, n_nh, nv)

    x_nh = x_nh.view(b,n_l,-1,nv,f)

    x_2 = x_nh.reshape(b,-1,nv)
    indices = indices.reshape(b,-1,nv)

    x_gath = torch.gather(x_2, 1, indices)

    x_gath = x_gath.view(b,-1,n_nh,nv*f)

    dist_vals[dist_vals<=cutoff_dist] = cutoff_dist

    weights = 1/(dist_vals.view(x_gath.shape))**power

    weights = weights/weights.sum(dim=-2, keepdim=True)

    x_inter = (x_gath*weights).sum(dim=-2)

    x_inter = x_inter.view(b,n,-1,nv,f)

    x_inter = x_inter.view(b,-1,nv,f)

    dist_vals = dist_vals.view(b,-1,n_nh,nv)

    return x_inter, dist_vals


def get_dists_interpolation(grid_layers,
                            search_level: int=2, 
                            input_level: int=0, 
                            target_level: int=0, 
                            indices_sample: dict=None):


    coords = grid_layers[str(input_level)].get_coordinates_from_grid_indices(
        indices_sample['indices_layers'][input_level] if indices_sample is not None else None
        )
    
    b,n = coords.shape[:2]
    l = 4**(search_level-input_level)
    n_l = n // l

    coords = coords.view(b,n_l,l,2)

    local_indices = indices_sample['indices_layers'][search_level] if indices_sample is not None else None
    coords_nh, _ = grid_layers[str(search_level)].get_nh(coords, local_indices=local_indices, sample_dict=indices_sample, mask = None) 

    coords_nh = coords_nh.view(b,n_l,-1,2)

    target_coords = grid_layers[str(target_level)].get_coordinates_from_grid_indices(
        indices_sample['indices_layers'][target_level] if indices_sample is not None else None
        )

    target_coords = target_coords.view(b, coords_nh.shape[1],-1, 2)

    b,n,nt,_ = target_coords.shape

    dist,_ = get_distance_angle(target_coords[...,0].unsqueeze(dim=-1),target_coords[...,1].unsqueeze(dim=-1), coords_nh[...,0].unsqueeze(dim=-2), coords_nh[...,1].unsqueeze(dim=-2))

    return dist


class Interpolator(nn.Module):
    def __init__(self,
                 grid_layers,
                 search_level: int = 2,
                 input_level: int = 0,
                 target_level: int = 0,
                 precompute=True,
                 nh_inter=2,
                 power=2,
                 cutoff_dist_level=None,
                 cutoff_dist=None,
                 search_level_compute=None,
                 input_coords=None  # for arbitrary grids
                 ) -> None:

        super().__init__()

        self.precompute = precompute

        if precompute:
            dists = get_dists_interpolation(grid_layers,
                                            search_level=search_level,
                                            input_level=input_level,
                                            target_level=target_level,
                                            input_coords=input_coords)

            self.register_buffer('dists', dists, persistent=True)

        self.grid_layers = grid_layers

        self.search_level = search_level
        self.input_level = input_level
        self.target_level = target_level
        self.nh_inter = nh_inter
        self.power = power
        self.search_level_compute = search_level if search_level_compute is None else search_level_compute

        self.cutoff_dist_level = input_level if cutoff_dist_level is None else cutoff_dist_level
        self.cutoff_dist = cutoff_dist

    def forward(self,
                x,
                mask=None,
                calc_density=False,
                indices_sample=None,
                input_level=None,
                target_level=None,
                search_level=None,
                input_coords=None):

        compute_dists = (input_level is not None) | (target_level is not None) | (search_level is not None) | (
                    input_coords is not None)

        search_level = self.search_level if search_level is None else search_level
        input_level = self.input_level if input_level is None else input_level
        target_level = self.target_level if target_level is None else target_level

        compute_dists = compute_dists | (self.precompute == False)

        if not compute_dists and indices_sample is not None and isinstance(indices_sample, dict):
            dist = self.dists[0, indices_sample['indices_layers'][self.search_level]]

        elif not compute_dists and indices_sample is None:
            dist = self.dists

        else:
            dist = get_dists_interpolation(self.grid_layers,
                                           search_level=self.search_level_compute,
                                           input_level=input_level,
                                           target_level=target_level,
                                           input_coords=input_coords,
                                           indices_sample=indices_sample)

        if self.cutoff_dist is None:
            cutoff_dist = max(
                [self.grid_layers[str(self.cutoff_dist_level)].nh_dist, self.grid_layers[str(target_level)].nh_dist])
        else:
            cutoff_dist = self.cutoff_dist

        x, dist_ = get_interpolation(x,
                                     mask,
                                     self.grid_layers[str(self.search_level_compute)],
                                     cutoff_dist,
                                     dist,
                                     self.nh_inter,
                                     power=self.power,
                                     indices_sample=indices_sample)

        if calc_density:
            grid_dist_output = self.grid_layers[str(target_level)].nh_dist
            density = get_density_map(grid_dist_output, dist_)

        else:
            density = None

        return x, density