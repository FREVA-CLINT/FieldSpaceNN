import math

import torch
import torch.nn as nn
from typing import List,Dict

from .grid_utils import get_distance_angle


def get_nh_idx_of_patch(adjc, patch_index, zoom_patch_sample, return_local=True):
    zoom_patch_sample = zoom_patch_sample[0] if zoom_patch_sample.numel()>1 else zoom_patch_sample

    # for icon and healpix
    zoom = int(math.log((adjc.shape[0])/4, 4))
    
    n_pts_patch = 4**(zoom-zoom_patch_sample)

    adjc_patch = adjc.reshape(-1, n_pts_patch, adjc.shape[-1])[patch_index].clone()
    
    index_range = (patch_index * n_pts_patch, (patch_index + 1) * n_pts_patch)

    adjc_mask = (adjc_patch < index_range[0].view(-1,1,1)) | (adjc_patch >= index_range[1].view(-1,1,1))

    ind = torch.where(adjc_mask)
    adjc_patch[ind] = adjc_patch[ind[0],ind[1],0]

    if return_local:
        adjc_patch = adjc_patch - index_range[0].view(-1,1,1)
    
    return adjc_patch, adjc_mask


def get_idx_of_patch(adjc, patch_index, zoom_patch_sample, return_local=True):
    
    zoom_patch_sample = zoom_patch_sample[0] if zoom_patch_sample.numel()>1 else zoom_patch_sample

    # for icon and healpix
    zoom = int(math.log((adjc.shape[0])/4, 4))
    
    n_pts_patch = 4**(zoom-zoom_patch_sample)

    adjc_patch = adjc[:,0].reshape(-1, n_pts_patch)[patch_index].clone()
    
    index_range = (patch_index * n_pts_patch, (patch_index + 1) * n_pts_patch)

    if return_local:
        adjc_patch = adjc_patch - index_range[0].view(-1,1)
    
    return adjc_patch


def gather_nh_data(x: torch.Tensor, adjc_patch: torch.Tensor) -> torch.Tensor:
    
    x_dims = x.shape
    
    x = x.reshape(*x_dims[:3],-1)

    nh = adjc_patch.shape[-1]
    
    #unsqueeze time and feature dim
    adjc_patch = adjc_patch.view(adjc_patch.shape[0], 1, adjc_patch.shape[1]*adjc_patch.shape[2], 1)

    adjc_patch = adjc_patch.expand(-1, x_dims[1], -1, x.shape[-1])

    x = torch.gather(x, 2, index=adjc_patch)
    
    x = x.view(*x_dims[:3], nh, *x_dims[3:])

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
        zoom (int): The global hierarchical zoom.
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
    
    def __init__(self, zoom: int, adjc: torch.Tensor, adjc_mask: torch.Tensor, coordinates: torch.Tensor, coord_system: str = "polar", periodic_fov=None) -> None:
        """
        Initializes the GridLayer with given parameters.

        :param zoom: The global zoom for hierarchy.
        :param adjc: The adjacency tensor.
        :param adjc_mask: Mask for the adjacency tensor.
        :param coordinates: The coordinates of grid points.
        :param coord_system: The coordinate system. Defaults to "polar".
        :param periodic_fov: The periodic field of view. Defaults to None.
        """
        super().__init__()

        # Initialize attributes
        self.zoom = zoom
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

        coords_nh = coordinates[nh_samples]
        # Calculate relative distances

        coords_lon_nh, coords_lat_nh = coords_nh[...,0], coords_nh[...,1]
        
        dists, _ = get_distance_angle(coords_lon_nh[:,[0]], coords_lat_nh[:,[0]], coords_lon_nh, coords_lat_nh, base="polar", rotate_coords=True)
        dists_lon, dists_lat = get_distance_angle(coords_lon_nh[:,[0]], coords_lat_nh[:,[0]], coords_lon_nh, coords_lat_nh, base="cartesian", rotate_coords=True)

        
        self.nh_dist = dists[:,1:].mean()
        self.nh_dist_lon = dists_lon[:,1:].abs().mean()
        self.nh_dist_lat = dists_lat[:,1:].abs().mean()
        # Compute distance statistics
        self.dist_quantiles = dists[dists > 1e-10].quantile(torch.linspace(0.01,0.99,20))

        self.min_dist = dists[dists > 1e-6].min()
        self.max_dist = dists[dists > 1e-10].max()
        self.mean_dist = dists[dists > 1e-10].mean()
        self.median_dist = dists[dists > 1e-10].median()
        
    def get_nh_patch(self, x: torch.Tensor, patch_index: torch.Tensor=None, zoom_patch_sample: torch.Tensor=None, mask: torch.Tensor = None) -> tuple:

        # Get neighborhood indices and adjacency mask
        adjc_patch, adjc_mask = get_nh_idx_of_patch(self.adjc, patch_index, zoom_patch_sample)
        
        adjc_mask = adjc_mask.unsqueeze(dim=1)
       # x_shape = x.shape
       # x = x.reshape(*x_shape[:3],-1)
        # Gather neighborhood data
        x = gather_nh_data(x, adjc_patch)

        if mask is not None:
            # Combine provided mask with adjacency mask
            mask = gather_nh_data(mask, adjc_patch)
            mask = torch.logical_or(mask.view(*x.shape[:4], mask.shape[-1]), adjc_mask.unsqueeze(dim=-1))
        else:
            # Use adjacency mask if no mask is provided
            mask = adjc_mask.unsqueeze(dim=-1).repeat_interleave(x.shape[-2], dim=-1)
        
        return x, mask

    def get_nh(self, x, patch_index=None, zoom_patch_sample=None, with_nh: bool=True, mask=None, **kwargs):

        if zoom_patch_sample is None or patch_index is None:
            
            adjc = self.adjc if with_nh else self.adjc[:,0]
            
            x = x[:,:,adjc]
            if mask is not None:
                mask = mask[:,:,adjc]

            return x, mask

        elif with_nh:
            return self.get_nh_patch(x, patch_index=patch_index, zoom_patch_sample=zoom_patch_sample, mask=mask)
        
        else:
            indices = get_idx_of_patch(self.adjc, patch_index, zoom_patch_sample, return_local=True)

            if mask is not None:
                mask = mask[:,indices]

            return x[:,:,indices], mask


    def get_idx_of_patch(self, patch_index=None, zoom_patch_sample=None, return_local=True,**kwargs):
        if patch_index is None:
            return self.adjc[:,0].unsqueeze(dim=0)
        else:
            return get_idx_of_patch(self.adjc, patch_index, zoom_patch_sample, return_local=return_local)


    def get_coordinates(self, patch_index=None, zoom_patch_sample=None, with_nh:bool=False, **kwargs):
        
        if patch_index is not None:
            indices = self.get_idx_of_patch(patch_index, 
                                            zoom_patch_sample, 
                                            with_nh=with_nh,
                                            return_local=False)

        elif with_nh:
            indices = self.adjc.unsqueeze(dim=0)

        else:
            indices = self.adjc[:,[0]].unsqueeze(dim=0)

        return self.coordinates[indices]



class MultiRelativeCoordinateManager(nn.Module):

    def __init__(self,  
                grid_layers: Dict[str, GridLayer], 
                rotate_coord_system=True) -> None:
                
        super().__init__()

        self.rotate_coord_system = rotate_coord_system
        self.rcms = nn.ModuleDict()

        zooms = [int(zoom) for zoom in grid_layers.keys()]
        nh_dists = [grid_layer.nh_dist for grid_layer in grid_layers.values()]

        self.nh_dists = dict(zip(zooms, nh_dists))
        self.grid_layers = grid_layers
        
    
    def register_rcm(self,
                     in_zoom, 
                     out_zoom, 
                     nh_in,
                     precompute,
                     coord_system,
                     ref='out'):
        
        in_zoom_str = str(in_zoom)
        out_zoom_str = str(out_zoom)

        if in_zoom_str not in self.rcms.keys():
            self.rcms[in_zoom_str] = nn.ModuleDict()
            
      
        if out_zoom_str not in self.rcms[in_zoom_str].keys():
            self.rcms[in_zoom_str][out_zoom_str] = RelativeCoordinateManager(
                    grid_layer_in=self.grid_layers[in_zoom_str],
                    grid_layer_out=self.grid_layers[out_zoom_str],
                    nh_in=nh_in,
                    precompute=precompute,
                    coord_system=coord_system,
                    rotate_coord_system=self.rotate_coord_system,
                    ref=ref
                )
            

    def forward(self, in_zoom, out_zoom, sample_dict=None, x=None, mask=None):
        #indices_in  = sample_dict["indices_layers"][in_zoom] if sample_dict is not None else None
        #indices_out = sample_dict["indices_layers"][out_zoom] if sample_dict is not None else None
        
        rcm = self.rcms[str(in_zoom)][str(out_zoom)]
        coordinates_rel = rcm(sample_dict=sample_dict)

        if x is None:
            return coordinates_rel
        else:
            if rcm.nh_in:
                x, mask = self.grid_layers[str(in_zoom)].get_nh(x, **sample_dict, mask=mask)
            else:
                x = x.unsqueeze(dim=3)
                if mask is not None:
                    mask = mask.unsqueeze(dim=2)
            
            return coordinates_rel, x, mask
    

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
            coordinates_rel = self.compute_rel_coordinates(nh_in=nh_in)
            
            self.register_buffer("coordinates_rel", torch.stack(coordinates_rel, dim=-1).squeeze(dim=1), persistent=False)
            


    def compute_rel_coordinates(self, patch_index=None, zoom_patch_sample=None, coordinates_in=None, coordinates_out=None, nh_in=False, **kwargs):

        if coordinates_in is None:
            coordinates_in = self.grid_layer_in.get_coordinates(patch_index=patch_index, zoom_patch_sample=zoom_patch_sample, with_nh=nh_in)

        elif coordinates_in is not None and nh_in:
            coordinates_in,_ = self.grid_layer_in.get_nh(coordinates_in, patch_index=patch_index, zoom_patch_sample=zoom_patch_sample)

        if coordinates_out is None:
            coordinates_out = self.grid_layer_out.get_coordinates(patch_index=patch_index, zoom_patch_sample=zoom_patch_sample)

        seq_dim_out_total = coordinates_out.shape[1]
        if coordinates_out.shape[1] > coordinates_in.shape[1]:
            coordinates_out = coordinates_out.view(coordinates_out.shape[0],coordinates_in.shape[1],-1,2)

        #if self.seq_lvl != -1:
        #    coordinates_out = sequenize(coordinates_out, max_seq_zoom=self.seq_lvl)[:,:,[0]]
        #    coordinates_in = sequenize(coordinates_in, max_seq_zoom=self.seq_lvl)

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
            #pass
        else:
            coordinates_rel = (coordinates_rel[0].view(b, -1, 1),
                            coordinates_rel[1].view(b, -1, 1))
        
        return coordinates_rel
    

    def forward(self, coordinates_in=None, coordinates_out=None, sample_dict=None):
        
        if not self.precomputed:
            coordinates_rel = self.compute_rel_coordinates(**sample_dict,
                                         coordinates_in=coordinates_in,
                                         coordinates_out=coordinates_out,
                                         sample_dict=sample_dict,
                                         nh_in = self.nh_in)
        else:
          
            coordinates_rel = self.coordinates_rel

            c_shape = coordinates_rel.shape
            
            if sample_dict is not None:
                indices = self.grid_layer_out.get_idx_of_patch(**sample_dict)
                coordinates_rel = coordinates_rel[0,indices].view(*indices.shape[:2],*c_shape[2:])
            else:
                coordinates_rel = coordinates_rel.view(1,-1,*c_shape[2:])

            coordinates_rel = (coordinates_rel[...,0],
                           coordinates_rel[...,1])

        return coordinates_rel




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
                      cutoff_dist: float,
                      dist: torch.tensor,
                      n_nh: int=3, 
                      power:int=2, 
                      mask_value:int=1e6,
                      grid_layer_search: GridLayer = None, 
                      sample_dict: dict=None):
    
    return_time_dim = False

    b, nt, n, nh, nv, f = x.shape
   
    x = x.clone()
   
    n_l = dist.shape[1]
    l = n // n_l
    
    x = x.view(b,nt,n_l,-1)
    mask = mask.view(b,nt,n_l,-1) if mask is not None else None

    if grid_layer_search is not None:

        x_nh, mask_nh = grid_layer_search.get_nh(
            x, 
            **sample_dict,
            mask=mask
            )
        
        x_nh = x_nh.view(b, nt, n_l, -1, nv, f)
        mask_nh = mask_nh.view(b, nt, n_l, -1, nv) if mask_nh is not None else None
    
    else:
        x_nh = x.view(b, nt, n_l, -1, nv, f)
        mask_nh = mask.view(b, nt, n_l, -1, nv) if mask is not None else None

    n = dist.shape[1]

    if mask_nh is not None:
        dist_ = dist.unsqueeze(dim=-1).unsqueeze(dim=1) + (mask_nh.unsqueeze(dim=3) * mask_value)
    else:
        dist_ = dist.unsqueeze(dim=-1).repeat(*(1,)*dist.dim(),nv)
        if dist_.shape[0] < b:
            dist_ = dist_.repeat(b,*(1,)*(dist_.dim()-1))

    dist_vals, indices = torch.topk(dist_, n_nh, dim=-2, largest=False)

    indices_offset = torch.arange(indices.shape[2], device=indices.device)
    offset = dist_.shape[-2]

    indices = indices + (indices_offset * offset).view(1, indices.shape[2], 1, 1, 1)

    indices = indices.view(b, nt, -1, n_nh, nv)

    x_nh = x_nh.view(b, nt, n_l, -1, nv, f)

    x_2 = x_nh.reshape(b, nt, -1, nv, f)
    indices = indices.reshape(b, nt, -1, nv, 1).expand(-1, -1, -1, -1, f)

    x_gath = torch.gather(x_2, 2, indices)

    x_gath = x_gath.view(b, nt, -1, n_nh, nv, f)

    dist_vals[dist_vals <= cutoff_dist] = cutoff_dist

    weights = 1 / (dist_vals.view(*x_gath.shape[:-1], 1)) ** power

    weights = weights / weights.sum(dim=-3, keepdim=True)

    x_inter = (x_gath * weights).sum(dim=-3)

    x_inter = x_inter.view(b, nt, n, -1, nv, f)

    x_inter = x_inter.view(b, nt, -1, nv, f)

    dist_vals = dist_vals.view(b, nt, -1, n_nh, nv)

    return x_inter, dist_vals


def get_dists_interpolation(grid_layers: Dict,
                            search_zoom_rel: int=2, 
                            input_zoom: int=7, 
                            target_zoom: int=0, 
                            sample_dict: dict=None,
                            input_coords: torch.tensor = None):

    if input_coords is None:
        coords = grid_layers[str(input_zoom)].get_coordinates(
            patch_index = sample_dict['patch_index'],
            zoom_patch_sample = sample_dict['zoom_patch_sample']
            )

        coords = coords.unsqueeze(dim=-2)
    else:
        coords = input_coords

    nh = coords.shape[2]
    l = 4**(search_zoom_rel)
    coords = coords.view(coords.shape[0], -1, l*nh, 2)

    search_zoom = max([input_zoom - search_zoom_rel, 0])

    coords_nh, _ = grid_layers[str(search_zoom)].get_nh(
        coords.unsqueeze(dim=1), 
        patch_index = sample_dict['patch_index'],
        zoom_patch_sample = sample_dict['zoom_patch_sample']
        )
    coords_nh = coords_nh.squeeze(dim=1)

    target_coords = grid_layers[str(target_zoom)].get_coordinates(
            patch_index = sample_dict['patch_index'],
            zoom_patch_sample = sample_dict['zoom_patch_sample']
            )

    n_l = min([coords_nh.shape[1], target_coords.shape[1]])

    b = coords_nh.shape[0]
    coords_nh = coords_nh.view(b,n_l,-1,2)

    target_coords = target_coords.view(b, n_l,-1, 2)

    dist,_ = get_distance_angle(target_coords[...,0].unsqueeze(dim=-1),target_coords[...,1].unsqueeze(dim=-1), coords_nh[...,0].unsqueeze(dim=-2), coords_nh[...,1].unsqueeze(dim=-2))

    return dist


class Interpolator(nn.Module):

    def __init__(self,  
                grid_layers, 
                search_zoom_rel: int=2, 
                input_zoom: int=0, 
                target_zoom: int=0, 
                precompute = True,
                nh_inter=2,
                power=2,
                cutoff_dist_zoom=None,
                cutoff_dist=None,
                input_coords=None,
                input_dists=None
                ) -> None:
                

        super().__init__()

        self.precompute = precompute

        if precompute:
            if input_dists is None:
                dists = get_dists_interpolation(grid_layers, 
                                                search_zoom_rel=search_zoom_rel, 
                                                input_zoom=input_zoom,
                                                target_zoom=target_zoom,
                                                input_coords=input_coords,
                                                sample_dict={'patch_index': None, 'zoom_patch_sample': None})
            
            else:
                dists = input_dists


            self.register_buffer('dists', dists, persistent=False)

        self.grid_layers = grid_layers
        self.search_zoom_rel = search_zoom_rel
        self.input_zoom = input_zoom
        self.target_zoom = target_zoom
        self.nh_inter = nh_inter
        self.power = power

        self.cutoff_dist_zoom = input_zoom if cutoff_dist_zoom is None else cutoff_dist_zoom
        self.cutoff_dist = cutoff_dist

    def forward(self,
                x,
                mask=None,
                calc_density=False,
                sample_dict=None,
                input_zoom=None,
                target_zoom=None,
                search_zoom_rel=None,
                input_coords=None,
                input_dists=None):
        
        compute_dists = (input_zoom is not None) | (target_zoom is not None) | (search_zoom_rel is not None) | (input_coords is not None)

        compute_dists = compute_dists & (input_dists is None)


        search_zoom_rel = self.search_zoom_rel if search_zoom_rel is None else search_zoom_rel
        input_zoom = self.input_zoom if input_zoom is None else input_zoom
        target_zoom = self.target_zoom if target_zoom is None else target_zoom

        compute_dists = compute_dists | (self.precompute == False)

        zoom_search = max([0,input_zoom - search_zoom_rel])
        grid_layer_search = self.grid_layers[str(zoom_search)]

        nh_inter = self.nh_inter
        if not compute_dists and sample_dict is not None and input_dists is None:
            
            indices = self.grid_layers[str(zoom_search)].get_idx_of_patch(**sample_dict, return_local=False)

            dist = self.dists[0,indices]


        elif not compute_dists and sample_dict is None and input_dists is None:
            dist = self.dists
        
        elif compute_dists:
            dist = get_dists_interpolation(self.grid_layers, 
                                    search_zoom_rel = search_zoom_rel,
                                    input_zoom = input_zoom,
                                    target_zoom = target_zoom,
                                    input_coords=input_coords,
                                    sample_dict=sample_dict)
        else:
            grid_layer_search = None
            nh_inter = input_dists.shape[-1]
            dist = input_dists.view(*input_dists.shape[:2],1,input_dists.shape[-1])

        
        if self.cutoff_dist is None:
            cutoff_dist = max([self.grid_layers[str(self.cutoff_dist_zoom)].nh_dist, self.grid_layers[str(target_zoom)].nh_dist])
        else:
            cutoff_dist = self.cutoff_dist
  

        x, dist_ = get_interpolation(x,
                                    mask, 
                                    cutoff_dist, 
                                    dist, 
                                    nh_inter, 
                                    power=self.power,
                                    sample_dict=sample_dict,
                                    grid_layer_search=grid_layer_search)
        

        if calc_density:
            grid_dist_output = self.grid_layers[str(target_zoom)].nh_dist
            density = get_density_map(grid_dist_output, dist_, power=self.power)

        else:
            density = None

        return x, density