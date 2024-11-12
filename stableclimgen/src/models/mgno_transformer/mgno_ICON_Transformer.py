import json,os
import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import omegaconf

from ..modules import transformer_modules as helpers
from utils.grid_utils_icon import get_distance_angle, get_coords_as_tensor, get_nh_variable_mapping_icon, icon_get_adjacent_cell_indices, icon_grid_to_mgrid, scale_coordinates


class grid_layer(nn.Module):
    def __init__(self, global_level, adjc, adjc_mask, coordinates, coord_system="polar", periodic_fov=None) -> None: 
        super().__init__()

        # introduce is_regid
        # if not add learnable parameters? like federkonstante, that are added onto the coords
        # all nodes have offsets, just the ones from the specified are learned
        self.global_level = global_level
        self.coord_system = coord_system
        self.periodic_fov = periodic_fov

        self.register_buffer("coordinates", coordinates, persistent=False)
        self.register_buffer("adjc", adjc, persistent=False)
        self.register_buffer("adjc_mask", adjc_mask==False, persistent=False)
        self.register_buffer("fov_mask", ((adjc_mask==False).sum(dim=-1)==adjc_mask.shape[1]).view(-1,1),persistent=False)

        n_samples = torch.min(torch.tensor([self.adjc.shape[0]-1, 100]))
        nh_samples = self.adjc[:n_samples]
        coords_nh = self.get_coordinates_from_grid_indices(nh_samples)
        dists = get_relative_positions(coords_nh, coords_nh, polar=True)[0]

        self.min_dist = dists[dists>1e-10].min()
        self.max_dist = dists[dists>1e-10].max()
        self.mean_dist = dists[dists>1e-10].mean()
        self.median_dist = dists[dists>1e-10].median()

    def get_nh(self, x, local_indices, sample_dict, relative_coordinates=True, coord_system=None, mask=None):

        indices_nh, adjc_mask = get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))
        
        x = gather_nh_data(x, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))

        if mask is not None:
            mask = gather_nh_data(mask, indices_nh, sample_dict['sample'], sample_dict['sample_level'], int(self.global_level))
            mask = torch.logical_or(mask.squeeze(dim=-1), adjc_mask.unsqueeze(dim=-1))
        else:
            mask = adjc_mask.unsqueeze(dim=-1).repeat_interleave(x.shape[-2],dim=-1)

        if relative_coordinates:
            coords = self.get_relative_coordinates_from_grid_indices(indices_nh, coord_system=coord_system)
        else:
            coords = self.get_coordinates_from_grid_indices(indices_nh)

        return x, mask, coords
    
    def get_nh_indices(self, local_indices):
        return get_nh_indices(self.adjc, local_cell_indices=local_indices, global_level=int(self.global_level))


    def get_coordinates_from_grid_indices(self, local_indices):
        coords = self.coordinates[:, local_indices]
        return coords
    
    def get_relative_coordinates_from_grid_indices(self, local_indices, coords=None, coord_system=None):
        
        if coord_system is None:
            coord_system = self.coord_system
        
        if coords is None:
            coords = self.get_coordinates_from_grid_indices(local_indices)

        coords_rel = get_distance_angle(coords[0,:,:,[0]], coords[1,:,:,[0]], coords[0], coords[1], base=coord_system, periodic_fov=self.periodic_fov)

        return coords_rel
    
    def get_relative_coordinates_cross(self, local_indices, coords, coord_system=None):

        if coord_system is None:
            coord_system = self.coord_system
        
        coords_ref = self.get_coordinates_from_grid_indices(local_indices)

       # if coords.dim()>3:
       #     n_c, b, n, nh = coords.shape
       #     coords = coords.view(n_c, b ,-1)
        if coords_ref.dim()<4:
            coords_ref = coords_ref.unsqueeze(dim=-1)

        coords_rel = get_distance_angle(coords_ref[0,:,:,[0]], coords_ref[1,:,:,[0]], coords[0], coords[1], base=coord_system, periodic_fov=self.periodic_fov) 
        
    #    if coords.dim()>3:
     #       coords_rel = coords_rel.view(n_c, b, n, nh)
        
        return coords_rel

    def get_sections(self, x, local_indices, section_level=1, relative_coordinates=True, return_indices=True, coord_system=None):
        indices = sequenize(local_indices, max_seq_level=section_level)
        x = sequenize(x, max_seq_level=section_level)
        if relative_coordinates:
            coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
        else:
            coords = self.get_coordinates_from_grid_indices(indices)

        mask = self.fov_mask[indices]

        if return_indices:
            return x, mask, coords, indices
        else:
            return x, mask, coords
   
   
    
    def get_position_embedding(self, indices, nh: bool, pos_embedder, coord_system, batch_dict=None, section_level=None):
        
        if nh:
            if isinstance(pos_embedder, position_embedder):
                indices = self.get_nh_indices(indices)[0]
                rel_coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
                pos_embeddings = pos_embedder(rel_coords[0], rel_coords[1])
            else:
                pos_embeddings = pos_embedder(self, indices, batch_dict)
        else:
            if isinstance(pos_embedder, position_embedder):
                indices = sequenize(indices, max_seq_level=section_level)
                coords = self.get_relative_coordinates_from_grid_indices(indices, coord_system=coord_system)
                pos_embeddings = pos_embedder(coords[0], coords[1])
            else:
                pos_embeddings = pos_embedder(self, indices)

        return pos_embeddings
    

class multi_grid_channel_attention(nn.Module):
    def __init__(self, model_dims_in, model_dim_out, n_chunks=4, n_heads=2, output_layer=False, sep_layer_norm=True) -> None: 
        super().__init__()

        model_dims_in_total = int(model_dims_in.sum())
 
        self.n_chunks = n_chunks

        model_dim_att = model_dims_in_total // n_chunks
        
        if output_layer:
            model_dim_att_out = model_dim_att 
        else:
            model_dim_att_out = model_dim_out 

        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim_att, model_dim_att_out//n_chunks, n_heads, input_dim=model_dim_att, qkv_proj=True
            )   

        self.sep_layer_norm = sep_layer_norm
        self.layer_norms = nn.ModuleList()
        if sep_layer_norm:
            for model_dim_in in model_dims_in:
                self.layer_norms.append(nn.LayerNorm(int(model_dim_in), elementwise_affine=True))
        else:
            self.layer_norms = nn.LayerNorm(model_dim_att, elementwise_affine=True)

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim_att_out, elementwise_affine=True),
            nn.Linear(model_dim_att_out, model_dim_att_out, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim_att_out, model_dim_att_out, bias=False)
        )

        self.res_layer = nn.Linear(model_dims_in_total, model_dim_att_out, bias=False)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim_att_out)*1e-6, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(model_dim_att_out)*1e-6, requires_grad=True)

        if output_layer:
            self.mlp_layer_out = nn.Sequential(nn.Linear(model_dim_att_out, model_dim_att_out//2, bias=False),
                                nn.SiLU(),
                                nn.Linear(model_dim_att_out//2, model_dim_out, bias=False))
        else:
            self.mlp_layer_out = nn.Identity()


    def forward(self, x_levels, mask=None):
        
        if self.sep_layer_norm:
            x = []
            for k, layer_norm in enumerate(self.layer_norms):
                x.append(layer_norm(x_levels[k]))
            x = torch.concat(x, dim=-1)
        else:
            x = torch.concat(x_levels, dim=-1)
            
        b,n,nv,f=x.shape
        x = x.view(b,n,nv,f)
        x = x.view(b*n,nv,f)
        x = x.view(b*n,nv*self.n_chunks,-1)

        if not self.sep_layer_norm:
            x = self.layer_norms(x)
        
        mask_chunk=mask
        if mask_chunk is not None:
            mask_chunk = mask_chunk.view(b*n,nv).repeat_interleave(self.n_chunks,dim=1)
            
        x = self.MHA(q=x, k=x, v=x, mask=mask_chunk)
        x = x.view(b*n,nv,-1)
        x = x.view(b,n,nv,-1)

        x = self.res_layer(torch.concat(x_levels, dim=-1)) + self.gamma * x

        x = x.view(b,n,nv,-1)
        x = x + self.gamma_mlp * self.mlp_layer(x)

        if mask is not None:
            mask = mask.view(b, x.shape[1], -1)
            mask[mask.sum(dim=-1)!=mask.shape[-1]] = False
            mask = mask.view(b,n,nv)

        return self.mlp_layer_out(x)
        
        


class icon_spatial_attention_ds(nn.Module):
    def __init__(self,
                 grid_layers, 
                 global_level,
                 model_dim,
                 n_heads,
                 seq_level_attention,
                 nh=1,
                 pos_emb_calc='cartesian_km',
                 emb_table_bins=16,
                 nh_attention=False, 
                 continous_pos_embedding=True) -> None: 
        super().__init__()
        
        # with interpolation to lowest grid

        self.grid_layers = grid_layers
        
        self.nh_attention = nh_attention  

        self.max_seq_level = seq_level_attention

        self.continous_pos_embedding=continous_pos_embedding

        self.grid_layer = grid_layers[global_level]
        self.global_level = global_level

        if continous_pos_embedding:
            if 'cartesian' in pos_emb_calc:
                self.coord_system = 'cartesian'
            else:
                self.coord_system = 'polar'
        
            self.position_embedder = position_embedder(0,0, emb_table_bins, model_dim, pos_emb_calc=pos_emb_calc)
        else:
            if nh_attention:
                self.position_embedder = nh_pos_embedding(grid_layers[global_level], nh, model_dim)
            else:
                self.position_embedder = seq_grid_embedding2(grid_layers[global_level], 4, seq_level_attention, model_dim, constant_init=False)



        self.embedding_layer = nn.Linear(model_dim, model_dim*2)

        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=True)
        
        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, model_dim, n_heads, input_dim=model_dim, qkv_proj=True
            )   

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )
    
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)


    def forward(self, x, mask, indices_grid_layers, batch_dict):
        
        b,n,nv,f = x.shape

        pos_embeddings = self.grid_layer.get_position_embedding(indices_grid_layers[int(self.global_level)], 
                                                                self.nh_attention, 
                                                                self.position_embedder, 
                                                                self.coord_system, 
                                                                batch_dict=batch_dict, 
                                                                section_level=self.max_seq_level)


        x = x.view(b,-1,nv,f)
                
        x_res = x 

        if self.nh_attention:
            x, mask, _ = self.grid_layer.get_nh(x, indices_grid_layers[int(self.global_level)], batch_dict, mask=mask)

            if mask is not None:
                mask_update = mask.clone()
                mask_update = mask_update.sum(dim=-1)==mask_update.shape[-1]
                mask_update = mask_update.view(b,n,nv)
        else:
            x = sequenize(x, max_seq_level=self.max_seq_level)

            if mask is not None:
                mask = sequenize(mask, max_seq_level=self.max_seq_level)
                mask_update = mask.clone().transpose(-1,-2)
                mask_update[(mask_update.sum(dim=-1)!=mask_update.shape[-1])]=False
                mask_update = mask_update.transpose(-1,-2)
                mask_update = mask_update.view(b,n,nv)
            else:
                mask_update=mask=None


        shift, scale = self.embedding_layer(pos_embeddings).unsqueeze(dim=-2).chunk(2, dim=-1)

        x = self.layer_norm(x) * (scale + 1) + shift

        b,n_seq,nh,nv,f = x.shape

        x = x.view(b*n_seq,nh*nv,f)

        if mask is not None:
            mask = mask.view(b*n_seq,-1)

        if self.nh_attention:
            q = x[:,[0]]
            kv = x
        else:
            q = kv = x
            
        x = self.MHA(q=q, k=kv, v=kv, mask=mask) 
        
        x = x.view(b,n,nv,f)

        x = x_res + self.gamma * x
        x = x + self.gamma_mlp * self.mlp_layer(x)

        return x, mask_update
    




class position_embedder(nn.Module):
    def __init__(self, min_dist, max_dist, emb_table_bins, emb_dim, pos_emb_calc="polar", phi_table=None) -> None: 
        super().__init__()
        self.pos_emb_calc = pos_emb_calc

        self.operation = None
        self.transform = None
        self.proj_layer = None
        self.cartesian = False

        if "descrete" in pos_emb_calc and "polar" in pos_emb_calc:
            self.pos1_emb = helpers.PositionEmbedder_phys_log(min_dist, max_dist, emb_table_bins, n_heads=emb_dim)
            if phi_table is not None:
                self.pos2_emb = phi_table
            else:
                self.pos2_emb = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, emb_table_bins, n_heads=emb_dim, special_token=True)

        if "semi" in pos_emb_calc and "polar" in pos_emb_calc:
            self.pos1_emb = nn.Sequential(nn.Linear(1, emb_dim), nn.SiLU())
            if phi_table is not None:
                self.pos2_emb = phi_table
            else:
                self.pos2_emb = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, emb_table_bins, n_heads=emb_dim, special_token=True)

        if "cartesian" in pos_emb_calc:
            self.proj_layer = nn.Sequential(nn.Linear(2, emb_dim, bias=True),
                                        nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim, bias=False),
                                        nn.Sigmoid())
            
            self.cartesian = True


        if "learned" in pos_emb_calc and "polar" in pos_emb_calc:
            self.proj_layer = nn.Sequential(nn.Linear(2*emb_dim, emb_dim, bias=True),
                                        nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim, bias=False),
                                        nn.Sigmoid())
        self.km_transform = False
        if 'km' in pos_emb_calc:
            self.km_transform = True
        
        
        if 'inverse' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_inv

        elif 'sig_log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_sig_log

        elif 'sig_inv_log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_inv_sig_log  

        elif 'log' in pos_emb_calc:
            self.transform = helpers.conv_coordinates_log
       
        if 'sum' in pos_emb_calc:
            self.operation = 'sum'

        elif 'product' in pos_emb_calc:
            self.operation = 'product'


    def forward(self, pos1, pos2):
        if self.cartesian:
            if self.km_transform:
                pos1 = pos1*6371.
                pos2 = pos2*6371.
                pos1[pos1.abs()<0.01]=0
                pos2[pos2.abs()<0.01]=0
            else:
                pos1[pos1.abs()<1e-6]=0
                pos2[pos2.abs()<1e-6]=0

            if self.transform is not None:
                pos1 = self.transform(pos1)
                pos2 = self.transform(pos2)
            
            return 16*self.proj_layer(torch.stack((pos1, pos2), dim=-1))    

        else:
            if self.km_transform:
                pos1 = pos1*6371.
                dist_0 = pos1 < 0.01
            else:
                dist_0 = pos1 < 1e-6

            if self.transform is not None:
                pos1 = self.transform(pos1)
            
            if isinstance(self.pos1_emb, nn.Sequential):
                pos1 = pos1.unsqueeze(dim=-1)

            pos1_emb = self.pos1_emb(pos1)
            pos2_emb = self.pos2_emb(pos2, special_token_mask=dist_0)

            if self.proj_layer is not None:
                return 16*self.proj_layer(torch.concat((pos1_emb, pos2_emb), dim=-1))
                        
            if self.operation == 'sum':
                return pos1_emb + pos2_emb
            
            elif self.operation == 'product':
                return pos1_emb * pos2_emb


def proj_data(data, weights):


    if weights.dim() - data.dim() == 2:
        data = data.unsqueeze(dim=2).unsqueeze(dim=2)

    projection = weights * data
    return projection.sum(dim=-2)


def get_spatial_projection_weights_dists(dists, dists_0, sigma):

    dist_weights = normal_dist(dists, sigma, dists_0)
    
    weights = F.softmax(dist_weights, dim=-1)

    #return weights/(weights_norm+1e-10)
    return weights

def get_spatial_projection_weights_n_dist(d_lons, d_lats, dlon_0, sigma_lon, dlat_0, sigma_lat):

    lon_weights = normal_dist(d_lons, sigma_lon, dlon_0)
    lat_weights = normal_dist(d_lats, sigma_lat, dlat_0)
    
    weights = lon_weights * lat_weights

    weights = F.softmax(weights, dim=-2)

    #return weights/(weights_norm+1e-10)
    return weights

def get_spatial_projection_weights_vm_dist(phis, dists, phi_0, kappa_vm, dists_0, sigma, mask=None):

    vm_weights = von_mises(phis, kappa_vm, phi_0)
    dist_weights = normal_dist(dists, sigma, dists_0)
    
    vm_weights[dist_weights[:,:,:,:,0]==1] = torch.exp(kappa_vm)

    weights = vm_weights * dist_weights

    if mask is not None:
        weights = weights.transpose(2,3)
        weights[mask] = -1e30 if weights.dtype == torch.float32 else -1e4
        weights = weights.transpose(2,3)

    weights = F.softmax(weights, dim=-2)

    return weights

def cosine(thetas, wavelengths, distances, theta_offsets=None):

    freq = 2*torch.pi/wavelengths

    if theta_offsets is not None:
        Z = torch.cos(freq * (torch.cos(thetas.unsqueeze(dim=-1)-theta_offsets)*distances.unsqueeze(dim=-1)).unsqueeze(dim=-1))
    else:
        Z = torch.cos(freq * distances)

    return Z


def von_mises(thetas, kappa, theta_offsets=None):

    if theta_offsets is not None:
        if not torch.is_tensor(theta_offsets):
            theta_offsets = torch.tensor(theta_offsets)
        vm = torch.exp(kappa * torch.cos(thetas.unsqueeze(dim=-1) - theta_offsets.unsqueeze(dim=-2)).unsqueeze(dim=-1))
    else:
        vm = torch.exp(kappa * torch.cos(thetas.unsqueeze(dim=-1)).unsqueeze(dim=-1))

    return vm


def normal_dist(distances, sigma, distances_offsets=None, sigma_cross=True):


    if distances_offsets is not None:
        if not torch.is_tensor(distances_offsets):
            distances_offsets = torch.tensor(distances_offsets)

        diff = distances.unsqueeze(dim=-1) - distances_offsets.unsqueeze(dim=1).unsqueeze(dim=-2)
    else:
        diff = distances
    
    if sigma_cross: 
        diff = diff.unsqueeze(dim=-1).unsqueeze(dim=-1)
    
    sigma = sigma.unsqueeze(dim=1).unsqueeze(dim=1)

    nd = torch.exp(-0.5 * (diff / sigma) ** 2)

    return nd

class angular_embedder(nn.Module):
    def __init__(self, n_bins, emb_dim) -> None: 
        super().__init__()
 
        self.thata_embedder = helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins, n_heads=emb_dim, special_token=True)

    def forward(self, thetas, dist_0_mask):
        return  self.thata_embedder(thetas, special_token_mask=dist_0_mask)

class multi_grid_encoder(nn.Module):
    def __init__(self, 
                 grid_layers:dict, 
                 global_level_in, 
                 global_levels_out, 
                 model_dim_in, 
                 model_dims: dict, 
                 n_vars_total, 
                 kernel_fcn='n', 
                 simultaneous=False, 
                 channel_attention=True,
                 n_chunks_channels=1,
                 n_heads=4) -> None: 
        
        super().__init__()

        self.grid_layers = grid_layers
        
        self.global_level_in = global_level_in
        self.global_levels = global_levels_out
        self.global_levels.sort()

        self.simultaneous = simultaneous

        self.aggregation_layers = nn.ModuleDict()

        for k, global_level in enumerate(self.global_levels):
            #nh_projection = True if global_level == global_level_in else False

            if global_level_in != global_level:

                global_level = int(global_level)
                
                model_dim_in = model_dim_in if simultaneous or 'model_dim_out' not in locals() else model_dim_out
                model_dim_out = model_dims[k]

                self.aggregation_layers[str(global_level)] = get_projection_layer(kernel_fcn, 
                                                                            model_dim_in, 
                                                                            model_dim_out, 
                                                                            grid_layers[str(global_level)], 
                                                                            n_vars_total,
                                                                            n_heads=n_heads, 
                                                                            channel_attention=channel_attention,
                                                                            n_chunks_channels=n_chunks_channels,
                                                                            nh_projection=True)
             

    def forward(self, x, indices_layers, drop_mask=None, coords_in=None, sample_dict=None):   
        
        x_levels={}
        drop_masks_level = {}
        #from fine to coarse

        for k, global_level in enumerate(self.global_levels):
            global_level = int(global_level)
            global_level_in = int(self.global_level_in) if k==0 else int(self.global_levels[k-1]) 
            global_level_in = int(self.global_level_in) if self.simultaneous else global_level_in
            
            x_in = x if self.simultaneous or k==0 else x_

            if str(global_level) in self.aggregation_layers.keys():
                if coords_in is None or k>0:
                    x_, drop_mask = self.aggregation_layers[str(global_level)](x_in, 
                                                                        indices_layer=indices_layers[global_level_in], 
                                                                        grid_layer=self.grid_layers[str(global_level_in)], 
                                                                        indices_layer_out=indices_layers[global_level], 
                                                                        grid_layer_out=self.grid_layers[str(global_level)],
                                                                        sample_dict = sample_dict,
                                                                        mask=drop_mask)
                else:
                    x_, drop_mask = self.aggregation_layers[str(global_level)](x_in,
                                                                        indices_layer_out=indices_layers[global_level], 
                                                                        grid_layer_out=self.grid_layers[str(global_level)],
                                                                        mask=drop_mask,
                                                                        sample_dict=sample_dict,
                                                                        coordinates=coords_in)
                
            else:
                x_ = x_in

            x_levels[global_level] = x_
            
            drop_masks_level[global_level] = drop_mask

        return x_levels, drop_masks_level


class multi_grid_decoder(nn.Module):
    def __init__(self, 
                 grid_layers:dict, 
                 global_levels_in: torch.tensor, 
                 global_levels: list, 
                 model_dims_in: list, 
                 model_dims_out: list,
                 n_vars_total,
                 kernel_fcn='n', 
                 channel_attention=True,
                 n_chunks_channels=4, 
                 n_heads=4,
                 output_layer=False) -> None: 
        
        super().__init__()

        self.grid_layers = grid_layers
        
        self.global_levels_in = global_levels_in = torch.tensor(global_levels_in)
        self.global_levels = global_levels = torch.tensor(global_levels)
        model_dims_in = torch.tensor(model_dims_in)
        model_dims_out = torch.tensor(model_dims_out)
 
        self.projection_layers = nn.ModuleDict()
        self.multi_grid_reduction_layers = nn.ModuleDict()

        for k, global_level_output in enumerate(global_levels):
            global_levels_in_step = global_levels_in[global_levels_in>=global_level_output]
            model_dims_in_step = model_dims_in[global_levels_in>=global_level_output]

            self.projection_layers_step = nn.ModuleDict()

            for j, global_level_in_step in enumerate(global_levels_in_step):
                if global_level_in_step != global_level_output:
                    self.projection_layers_step[str(int(global_level_in_step))] = (get_projection_layer(kernel_fcn, 
                                                                                                int(model_dims_in_step[j]), 
                                                                                                int(model_dims_in_step[j]), 
                                                                                                grid_layers[str(int(global_level_output))], 
                                                                                                n_vars_total,
                                                                                                n_heads=n_heads, 
                                                                                                channel_attention=channel_attention,
                                                                                                n_chunks_channels=n_chunks_channels,
                                                                                                nh_projection=True))
                else:
                    self.projection_layers_step[str(int(global_level_in_step))] = nn.Identity()

            model_dims_in = torch.concat((model_dims_out[k].view(-1), model_dims_in[global_levels_in<global_level_output]))
            global_levels_in = torch.concat((global_level_output.view(-1), global_levels_in[global_levels_in<global_level_output]))


            self.projection_layers[str(int(global_level_output))] = self.projection_layers_step
            self.multi_grid_reduction_layers[str(int(global_level_output))] = multi_grid_channel_attention(model_dims_in_step,
                                                                                                 int(model_dims_out[k]),
                                                                                                 n_heads=n_heads,
                                                                                                 output_layer=output_layer if k==len(global_levels)-1 else False)

    def forward(self, x_levels, indices_grid_layers, drop_mask_levels=None, sample_dict=None, coords_out=None):   
        

        for global_level_output, projection_layers_output in self.projection_layers.items():
            
            x_out = []
            for global_level_input, projection_layers_input in projection_layers_output.items():

                if drop_mask_levels is not None:
                    drop_mask_input = drop_mask_levels[int(global_level_input)] if int(global_level_input) in drop_mask_levels.keys() else None

                if global_level_input != global_level_output:
                    x, drop_mask_level = projection_layers_input(x_levels[int(global_level_input)], 
                                                        grid_layer=self.grid_layers[global_level_input], 
                                                        grid_layer_out=self.grid_layers[global_level_output], 
                                                        indices_layer=indices_grid_layers[int(global_level_input)],
                                                        indices_layer_out = indices_grid_layers[int(global_level_output)],
                                                        sample_dict=sample_dict,
                                                        mask=drop_mask_input)
                else:
                    x = x_levels[int(global_level_input)]
                    drop_mask_level = drop_mask_input

                x_out.append(x)

                if drop_mask_level is not None:
                    mask_shape = x.shape[:-1]

                    if int(global_level_output) in drop_mask_levels.keys() and drop_mask_input is not None:
                        drop_mask_levels[int(global_level_output)] = torch.logical_and(drop_mask_levels[int(global_level_output)].view(mask_shape), drop_mask_level.view(mask_shape))
                    else:
                        drop_mask_levels[int(global_level_output)] = drop_mask_level.view(x.shape[:-1])


            x_levels[int(global_level_output)] = self.multi_grid_reduction_layers[global_level_output](x_out)

    
        return x_levels[int(global_level_output)], drop_mask_levels[int(global_level_output)]


class processing_layer(nn.Module):
    #on each grid layer

    def __init__(self, grid_layers: dict,
                 global_levels, 
                 model_dims:list,
                 seq_level_attention: bool,
                 n_vars_total,
                 nh:int=1,
                 n_heads:int=4,
                 pos_emb_calc:str='cartesian_km',
                 emb_table_bins:int=16,
                 kernel_fcn:str='n_ca',
                 kernel_dim=4, 
                 n_chunks=4) -> None: 
        
        super().__init__()

        self.global_levels = global_levels
        self.mode = kernel_fcn

        self.processing_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()

        for k, global_level in enumerate(global_levels):
            global_level = str(int(global_level))

            if 'spatial' in kernel_fcn:
                nh_attention = 'nh' in kernel_fcn
                self.processing_layers[global_level] = icon_spatial_attention_ds(grid_layers, 
                                                                                global_level, 
                                                                                int(model_dims[k]),
                                                                                n_heads,
                                                                                seq_level_attention,
                                                                                nh=nh,
                                                                                pos_emb_calc=pos_emb_calc,
                                                                                emb_table_bins=emb_table_bins, 
                                                                                nh_attention=nh_attention, 
                                                                                continous_pos_embedding=True)
        self.grid_layers = grid_layers

            
    def forward(self, x_levels, indices_layers, batch_dict, drop_masks_levels=None):
        
        for k, global_level in enumerate(self.global_levels):
            global_level = int(global_level)
   
            x, mask = self.processing_layers[str(global_level)](x_levels[global_level], drop_masks_levels[global_level], indices_layers, batch_dict)

            x_levels[global_level] = x
            drop_masks_levels[global_level] = mask

        return x_levels, drop_masks_levels


class channel_attention(nn.Module):
    def __init__(self, grid_layer, model_hparams, model_dim=None, n_chunks=None) -> None: 
        super().__init__()

        self.grid_layer = grid_layer

        model_dim = model_hparams['model_dim']

        # channel attention is good here!
        self.n_chunks = 4

        input_model_dim = model_hparams['model_dim'] // self.n_chunks

        self.norm = nn.LayerNorm(input_model_dim, elementwise_affine=True)
        
        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim, input_model_dim, model_hparams['n_heads'], input_dim=input_model_dim, qkv_proj=True
            )   
        
        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )
        self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)

    def forward(self, x_level):
        b,n,f = x_level.shape

        x = x_level.view(b*n,f)

        x = x.view(b*n,self.n_chunks,-1)

        x = self.norm(x)

        x = self.MHA(q=x, k=x, v=x)

        x = x_level + self.gamma * x.view(b,n,f)
        x = x + self.gamma_mlp * self.mlp_layer(x)

        x = x.view(b,n,f)

        return x



# make similar hier nh grid embedding (hier vielleicht noch nicht wichtig)
class nh_pos_embedding(nn.Module):
    def __init__(self, grid_layer: grid_layer, nh, emb_dim, softmax=False) -> None: 
        super().__init__()
        #uses hierarical grid embeddings
        #vm to interpolate -> for nh attention

        self.n_nh  = [4,6,20]

        # number of bins per neighbours
        n_bins_nh = {1:4, 2:6, 3:12} 

        self.grid_layer = grid_layer
        self.nh = nh
        
        self.softmax = softmax

        self.angular_embedders= nn.ParameterList()
        for n in range(nh):
            self.angular_embedders.append(helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins_nh[n+1], n_heads=emb_dim, special_token=True, constant_init=False))


    def forward(self, x, indices_layer, batch_dict, add_to_x=True, drop_mask=None):
      
        x_nh, _, rel_coords = self.grid_layer.get_nh(x, indices_layer, batch_dict, coord_system='polar', relative_coordinates=True)
        rel_coords = torch.stack(rel_coords, dim=0)
        b,n,n_nh,f = x_nh.shape
        
        rel_coords = rel_coords.split(tuple(self.n_nh[:self.nh]), dim=-1)
        
        embeddings = []
        for i in range(self.nh):

            embeddings.append(self.angular_embedders[i](rel_coords[i][1], special_token_mask=rel_coords[i][0]<1e-6))

        embeddings = torch.concat(embeddings, dim=-2)

        if self.softmax:
            embeddings = F.softmax(embeddings, dim=-2)

        if add_to_x:
            x_nh = x_nh * embeddings  
            return x_nh.view(b,n,n_nh,f)
        else:
            return embeddings.view(b,n,n_nh,f)


class seq_grid_embedding2(nn.Module):
    def __init__(self, grid_layer: grid_layer, max_seq_level, n_bins, emb_dim, constant_init=False) -> None: 
        super().__init__()
        #uses hierarical grid embeddings
        #vm to interpolate -> for nh attention

        self.grid_layer = grid_layer
        self.max_seq_level = max_seq_level

        self.angular_embedders= nn.ParameterList()
        for _ in range(max_seq_level):
            self.angular_embedders.append(helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins, n_heads=emb_dim, special_token=True, constant_init=constant_init))

    def forward(self, x, indices_layer):
        b,n,f = x.shape
        
        seq_level = min([get_max_seq_level(x), self.max_seq_level])

        all_embeddings = []
        for i in range(seq_level):
            x, mask, rel_coords, indices_layer = self.grid_layer.get_sections(x, indices_layer, section_level=1, relative_coordinates=True, return_indices=True, coord_system="polar")
            
            embeddings = self.angular_embedders[i](rel_coords[1], special_token_mask=rel_coords[0]<1e-6)

            if i > 0:
                scale = scale.view(x.shape) + embeddings
            else:
                scale = embeddings

            if seq_level>1:
                indices_layer = indices_layer[:,:,[0]]

            all_embeddings.append(scale.view(b,n,f))
        return scale.view(b,n,f)

# make similar hier nh grid embedding (hier vielleicht noch nicht wichtig)
class seq_grid_embedding(nn.Module):
    def __init__(self, grid_layer: grid_layer, n_bins, seq_level, emb_dim, softmax=True,  constant_init=False) -> None: 
        super().__init__()
        #uses hierarical grid embeddings
        #vm to interpolate -> for nh attention


        self.grid_layer = grid_layer
        self.seq_level = seq_level

        self.softmax = softmax

        self.angular_embedders= nn.ParameterList()
        for _ in range(seq_level):
            self.angular_embedders.append(helpers.PositionEmbedder_phys(-torch.pi, torch.pi, n_bins, n_heads=emb_dim, special_token=True, constant_init=constant_init))
            #initi to 0.25

    def forward(self, x, indices_layer,  add_to_x=True, drop_mask=None):
        b,n,f = x.shape
        
        seq_level = min([get_max_seq_level(x), self.seq_level])

        for i in range(seq_level):
            x, mask, rel_coords, indices_layer = self.grid_layer.get_sections(x, indices_layer, section_level=1, relative_coordinates=True, return_indices=True, coord_system="polar")
            
            embeddings = self.angular_embedders[i](rel_coords[1], special_token_mask=rel_coords[0]<1e-6)

            if self.softmax:
                if drop_mask is not None and i==seq_level-1:
                    embeddings[drop_mask.view(x.shape[:-1],1)] = -100
                embeddings = F.softmax(embeddings, dim=-2-i)

            if i > 0:
                scale = scale.view(x.shape) + embeddings
            else:
                scale = embeddings

            if seq_level>1:
                # keep midpoints of previous
                indices_layer = indices_layer[:,:,[0]]

        if add_to_x:
            x = x * scale  
            return x.view(b,n,f)
        else:
            return scale.view(b,n,f)



def get_projection_layer(kernel_fcn, model_dim_in, model_dim_out, grid_layer, n_vars_total, n_heads=4, channel_attention=True, n_chunks_channels=1, nh_projection=False):

    if kernel_fcn == 'n':
        projection_layer = projection_layer_n_multi_ca(model_dim_in, 
                                                       model_dim_out, 
                                                       grid_layer.min_dist/2,
                                                       grid_layer.min_dist, 
                                                       n_vars_total,
                                                       n_heads=n_heads, 
                                                       channel_attention=channel_attention, 
                                                       nh_projection=nh_projection,
                                                       n_chunks_channels=n_chunks_channels,
                                                       )

    return projection_layer


class projection_layer(nn.Module):
    def __init__(self, 
                 model_dim,
                 n_vars_total,
                 polar=False, 
                 requires_arel_positions=True, 
                 channel_attention=True, 
                 n_heads=4, 
                 nh_projection=False, 
                 init_dist_range=[], 
                 init_sigma_range=[],
                 n_chunks_channels=1) -> None: 
        
        super().__init__()
        # use channel attention and residual here

        self.periodic_fov=None
        self.polar=polar
        self.coord_system = 'polar' if polar else 'cartesian'
        
        self.requires_arel_positions = requires_arel_positions
        self.channel_attention = channel_attention
        self.nh_projection = nh_projection
       
        dist = torch.linspace(0, init_dist_range[1], model_dim)
        self.emb_dist = nn.Parameter(dist.view(1,-1).repeat_interleave(n_vars_total, dim=0), requires_grad=True)

        sigma = torch.linspace(init_sigma_range[0], init_sigma_range[1], model_dim)
        self.emb_sigma = nn.Parameter(sigma.view(1,-1).repeat_interleave(n_vars_total, dim=0), requires_grad=True)
        self.n_vars_total = n_vars_total

        
        if channel_attention:
            n_chunks_channels = model_dim//(4*n_heads) if n_vars_total==1 else n_chunks_channels
            self.att_layer = channel_variable_attention(model_dim, n_chunks_channels, n_heads)
        else:
            self.att_layer = nn.Identity()
    
    def get_embedding(self, var_indices):

        if self.n_vars_total==1:
            var_indices = var_indices.clamp(max=self.n_vars_total-1)
            return self.emb_sigma[var_indices][:,[0]], self.emb_dist[var_indices][:,[0]]
        
        if self.emb_sigma is not None:
            return self.emb_sigma[var_indices], self.emb_dist[var_indices]

    def forward(self, 
                x, 
                grid_layer: grid_layer=None, 
                grid_layer_out: grid_layer=None, 
                indices_layer=None, 
                indices_layer_out=None,
                sample_dict=None,
                coordinates=None, 
                coordinates_out=None,
                mask=None):
        
        nh_projection = self.nh_projection

        emb_sigma, emb_dist = self.get_embedding(sample_dict['variables'])

        if coordinates is None and grid_layer is not None and not nh_projection:
            coordinates = grid_layer.get_coordinates_from_grid_indices(indices_layer)

        elif coordinates is None and grid_layer is not None and nh_projection:
            x, mask, coordinates = grid_layer.get_nh(x, indices_layer, sample_dict, relative_coordinates=False, coord_system=self.coord_system, mask=mask)
            
        if coordinates_out is None and grid_layer_out is not None:
            coordinates_out = grid_layer_out.get_coordinates_from_grid_indices(indices_layer_out)

        n_v, f = x.shape[-2:]
        n_c, b, seq_dim_in = coordinates.shape[:3]
        seq_dim_out = coordinates_out.shape[2]
        
        if seq_dim_in > seq_dim_out:
            coordinates = coordinates.view(n_c, b, seq_dim_out, -1)
            x = x.view(b, seq_dim_out, coordinates.shape[-1], n_v, f)
            if mask is not None:
                mask = mask.view(b, seq_dim_out, -1, n_v)
            coordinates_out = coordinates_out.view(n_c, b, seq_dim_out, -1)
        else:
            coordinates = coordinates.view(n_c, b, seq_dim_in,-1)
            coordinates_out = coordinates_out.view(n_c, b, seq_dim_in, -1)
            x = x.view(b, seq_dim_in, coordinates.shape[-1], n_v, f)
            if mask is not None:
                mask = mask.view(b, seq_dim_in, -1, n_v)

        if self.requires_arel_positions:
            coords_ref = coordinates[:,:,:,[0]]
            coordinates_rel = get_distance_angle(coords_ref[0], coords_ref[1], coordinates[0], coordinates[1], base=self.coord_system, periodic_fov=self.periodic_fov)
            coordinates_rel_out = get_distance_angle(coords_ref[0], coords_ref[1], coordinates_out[0], coordinates_out[1], base=self.coord_system, periodic_fov=self.periodic_fov)

            x, mask = self.project(x, coordinates_rel, coordinates_rel_out, mask=mask)
        else:
            coordinates = coordinates.unsqueeze(dim=-2)
            coordinates_out = coordinates_out.unsqueeze(dim=-1)
            coordinates_rel = get_distance_angle(coordinates[0], coordinates[1], coordinates_out[0], coordinates_out[1], base=self.coord_system, periodic_fov=self.periodic_fov)

            x, mask = self.project(x, coordinates_rel, emb_sigma, emb_dist, mask=mask)      

        b, n, nt, nv, f = x.shape
        x = x.view(b,n*nt,nv,f)

        if mask is not None:
            mask = mask.sum(dim=-2)==mask.shape[-2]
            if x.shape[1] > mask.shape[1]:
                mask = mask.unsqueeze(dim=-2).repeat_interleave(x.shape[1]//mask.shape[1], dim=-2)
            mask = mask.view(b,n*nt,nv)

        if self.channel_attention:
            x, mask = self.att_layer(x, x_res=x.view(b,n,nt,nv,f).mean(dim=2, keepdim=True), mask=mask)    
                
        return x, mask

    def project(self, x, coordinates_rel, coordinates_rel_out=None, mask=None):
        return x, mask


class channel_variable_attention(nn.Module):
    def __init__(self, model_dim, n_chunks_channel, n_heads):
        super().__init__()

        self.n_chunks_channels = n_chunks_channel
        model_dim_att = model_dim // n_chunks_channel

        self.layer_norm = nn.LayerNorm(model_dim_att, elementwise_affine=True)
        self.MHA = helpers.MultiHeadAttentionBlock(
            model_dim_att, model_dim_att, n_heads, input_dim=model_dim_att, qkv_proj=True
            )   

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim, bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )
    
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)

    def forward(self, x, x_res=None,mask=None):
        b,n,nv,f=x.shape
        x = x.view(b,n,nv,f)

        if x_res is None:
            x_res = x

        x = x.view(b*n,nv,f)
        x = x.view(b*n,nv*self.n_chunks_channels,-1)

        q = k = v = self.layer_norm(x)
        
        mask_chunk=mask
        if mask_chunk is not None:
            mask_chunk = mask_chunk.view(b*n,nv).repeat_interleave(self.n_chunks_channels,dim=1)
            
        x = self.MHA(q=q, k=k, v=v, mask=mask_chunk)
        x = x.view(b*n,nv,-1)
        x = x.view(b,n,nv,-1)

        x = x_res + self.gamma * x.view(b,x_res.shape[1],-1,nv,f)

        x = x.view(b,n,nv,-1)
        x = x + self.gamma_mlp * self.mlp_layer(x)

        if mask is not None:
            mask = mask.view(b, x.shape[1], -1)
            mask[mask.sum(dim=-1)!=mask.shape[-1]] = False
            mask = mask.view(b,n,nv)

        return x, mask

class projection_layer_n_multi_ca(projection_layer):
   
    def __init__(self, 
                 model_dim_in, 
                 model_dim_out, 
                 dist_min,
                 dist_max,  
                 n_vars_total, 
                 n_heads=4, 
                 channel_attention=True,
                 n_chunks_channels=1, 
                 nh_projection=False) -> None: 

        super().__init__(model_dim_out, 
                         n_vars_total, 
                         polar=True, 
                         requires_arel_positions=False, 
                         channel_attention=channel_attention, 
                         n_heads=n_heads, 
                         nh_projection=nh_projection,
                         init_sigma_range=[dist_min, dist_max],
                         init_dist_range=[0, dist_max],
                         n_chunks_channels=n_chunks_channels)

        if model_dim_in > 1 and model_dim_in != model_dim_out:
            self.lin_proj = nn.Linear(model_dim_in, model_dim_out, bias=False)
        else:
            self.lin_proj = nn.Identity()

        self.min_sigma = dist_min/2

        
    def project(self, x: torch.tensor, coordinates_rel: torch.tensor, sigma:torch.tensor, dist:torch.tensor, mask=None):
        
        x = self.lin_proj(x)

        min_val = 1e-10 if x.dtype == torch.float32 else 1e-4
        sigma = sigma.clamp(min=min_val)

        weights = normal_dist(coordinates_rel[0], sigma, dist, sigma_cross=False)
        
        if mask is not None:
            # weights should automatically 
            #weights = weights.unsqueeze(dim=-2).repeat_interleave(mask.shape[-1],dim=-2)
            weights = weights.masked_fill(mask.unsqueeze(dim=2), -1e30 if x.dtype == torch.float32 else -1e4)

        weights = F.softmax(weights, dim=-2)

        x = (x.unsqueeze(dim=2) * weights.unsqueeze(dim=-2)).sum(dim=-3)
        
        return x, mask
    


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


class MultiGridBlock(nn.Module):
    def __init__(self, 
                 grid_layers,
                 global_level_in,
                 global_levels_encode,
                 global_levels_decode, 
                 model_dim_in,
                 model_dims_encode,
                 model_dims_decode, 
                 n_vars_total,
                 encoder_kernel='n',
                 encoder_simul=False,
                 processing_method=None,
                 processing_min_lvl=2,
                 decoder_kernel='n',
                 seq_level_attention=2, 
                 nh=1,
                 n_heads=4, 
                 encoder_channel_attention=True,
                 decoder_channel_attention=True,
                 n_chunks_channels_encoder=1,
                 n_chunks_channels_decoder=4,
                 pos_emb_calc='cartesian_km',
                 emb_table_bins=16,
                 output_layer=False
                 ):
        super().__init__()      
        
        self.decomp_layer = multi_grid_encoder(grid_layers, 
                                               global_level_in, 
                                               global_levels_encode, 
                                               model_dim_in,
                                               model_dims_encode,
                                               n_vars_total, 
                                               kernel_fcn=encoder_kernel,
                                               channel_attention=encoder_channel_attention,
                                               n_chunks_channels=n_chunks_channels_encoder,
                                               n_heads=n_heads, 
                                               simultaneous=encoder_simul)


        if processing_method is not None:
            global_levels_process = torch.tensor(global_levels_encode)
            proc_idx = global_levels_process >= int(processing_min_lvl)
            global_levels_process = global_levels_process[proc_idx]
            model_dims_processing = torch.tensor(model_dims_encode)[proc_idx]

            self.processing_layer = processing_layer(grid_layers,
                                                     global_levels_process,  
                                                     model_dims_processing,
                                                     seq_level_attention,
                                                     n_vars_total,
                                                     nh=nh,
                                                     n_heads=n_heads,
                                                     pos_emb_calc=pos_emb_calc,
                                                     emb_table_bins=emb_table_bins,
                                                     kernel_fcn=processing_method)

        self.mg_layer = multi_grid_decoder(grid_layers,
                                           global_levels_encode,
                                           global_levels_decode, 
                                           model_dims_encode,
                                           model_dims_decode,
                                           n_vars_total,
                                           kernel_fcn=decoder_kernel,
                                           channel_attention=decoder_channel_attention,
                                           n_chunks_channels=n_chunks_channels_decoder,
                                           n_heads=n_heads,
                                           output_layer=output_layer)


    def forward(self, x, indices_layers, indices_batch_dict, mask=None, coords_in=None, coords_out=None):

        x_levels, mask_levels = self.decomp_layer(x, indices_layers, drop_mask=mask, coords_in=coords_in, sample_dict=indices_batch_dict)

        if hasattr(self, 'processing_layer'):
            x_levels, mask_levels = self.processing_layer(x_levels, indices_layers, indices_batch_dict, mask_levels)

        x, mask = self.mg_layer(x_levels, indices_layers, mask_levels, indices_batch_dict, coords_out=coords_out)

        return x, mask


def check_value(value, n_repeat):
    if not isinstance(value, list) and not isinstance(value, omegaconf.listconfig.ListConfig):
        value = [value]*n_repeat
 #   else:
 #       if len(value) != n_repeat and len(value)<=1:
 #           value = [value]*n_repeat
    return value


class ICON_Transformer(nn.Module):
    def __init__(self, 
                 icon_grid: str,
                 global_levels_block_encoder: list,
                 model_dims_encoder: list,
                 mg_encoder_kernel: list | str,
                 mg_encoder_simul: list | bool,
                 mg_decoder_kernel: list | str,
                 mg_spa_method: list | str = None,
                 mg_spa_min_lvl: list | str = None,
                 global_levels_block_decoder: list=[],
                 model_dims_decoder: list =[],
                 mg_encoder_channel_attention: list | bool = True,
                 mg_decoder_channel_attention: list | bool = True,
                 mg_encoder_n_chunks_channel: list | int = 1,
                 mg_decoder_n_chunks_channel: list | int = 1,
                 mg_encoder_spa_method: list | bool = False,
                 mg_decoder_spa_method: list | bool = False,
                 nh: int=1,
                 seq_lvl_att: int=2,
                 model_dim_in: int=1,
                 model_dim_out: int=1,
                 n_heads:int=2,
                 residual_cons: bool=True,
                 var_model: bool=False,
                 pos_emb_calc: str='cartesian_km',
                 kappa_vm: float=0.5,
                 load_strict: bool=True,
                 separate_variables=True,
                 input_dim=1,
                 n_vars_total=1
                 ) -> None: 
        
        super().__init__()
        
        global_levels_encode_flat = torch.concat([torch.tensor(l) for l in global_levels_block_encoder])
        global_levels_decode_flat = torch.concat([torch.tensor(l) for l in global_levels_block_decoder]) if len(global_levels_block_decoder)>0 else global_levels_encode_flat
        global_levels = torch.concat((global_levels_encode_flat, global_levels_decode_flat, torch.tensor(0).view(-1))).unique()
        self.register_buffer('global_levels', global_levels, persistent=False)

        mgrids = icon_grid_to_mgrid(xr.open_dataset(icon_grid),
                                    int(torch.tensor(global_levels).max()) + 1, 
                                    nh=nh)

        self.coord_system = "polar" if "polar" in  pos_emb_calc else "cartesian"
        
        self.register_buffer('global_indices', torch.arange(mgrids[0]['coords'].shape[1]).unsqueeze(dim=0), persistent=False)
        self.register_buffer('cell_coords_global', mgrids[0]['coords'], persistent=False)
        
        grid_layers = nn.ModuleDict()
        for global_level in global_levels:
            grid_layers[str(int(global_level))] = grid_layer(global_level, mgrids[global_level]['adjc_lvl'], mgrids[global_level]['adjc_mask'], mgrids[global_level]['coords'], coord_system=self.coord_system)
        
    
        n_blocks = len(global_levels_block_encoder)


        model_dims_encoder = check_value(model_dims_encoder, n_blocks)
        mg_encoder_kernel = check_value(mg_encoder_kernel, n_blocks)
        mg_encoder_simul = check_value(mg_encoder_simul, n_blocks)
        mg_spa_method = check_value(mg_spa_method, n_blocks)
        mg_spa_min_lvl = check_value(mg_spa_min_lvl, n_blocks)
        mg_decoder_kernel = check_value(mg_decoder_kernel, n_blocks)
        mg_encoder_spa_method = check_value(mg_encoder_spa_method, n_blocks)
        mg_decoder_spa_method = check_value(mg_decoder_spa_method, n_blocks)
        mg_encoder_n_chunks_channel = check_value(mg_encoder_n_chunks_channel, n_blocks)
        mg_decoder_n_chunks_channel = check_value(mg_decoder_n_chunks_channel, n_blocks)
        mg_encoder_channel_attention = check_value(mg_encoder_channel_attention, n_blocks)
        mg_decoder_channel_attention = check_value(mg_decoder_channel_attention, n_blocks)

        input_dim=1 if separate_variables else input_dim
        self.model_dim_in = model_dim_in

        if len(global_levels_block_decoder)==0:
            global_levels_block_decoder = list([[int(k)] for k in torch.tensor(global_levels_block_encoder)[:,0]])
            global_levels_block_decoder[-1]=[0]

            model_dims_decoder = list([[int(k)] for k in torch.tensor(model_dims_encoder)[:,0]])
            model_dims_decoder[-1] = model_dim_out
        else:
            global_levels_block_decoder = check_value(global_levels_block_decoder, n_blocks)
            model_dims_decoder = check_value(model_dims_decoder, n_blocks)



        self.MGBlocks = nn.ModuleList()

        for k in range(n_blocks):
            global_level_in = 0 if k==0 else global_levels_block_decoder[k-1][-1]
            model_dim_in = model_dim_in if k==0 else model_dims_decoder[k-1][-1]
            global_levels_encode = global_levels_block_encoder[k]
            global_levels_decode = global_levels_block_decoder[k]
            model_dims_encode = model_dims_encoder[k] 
            model_dims_decode = model_dims_decoder[k]

            if k==n_blocks-1:
                if global_levels_decode[-1]!=0:
                    global_levels_decode.append(0)
                    model_dims_decode.append(model_dim_out)
                else:
                    model_dims_decode[-1]=model_dim_out

            self.MGBlocks.append(MultiGridBlock(grid_layers,
                                                global_level_in,
                                                global_levels_encode,
                                                global_levels_decode, 
                                                model_dim_in,
                                                model_dims_encode,
                                                model_dims_decode, 
                                                n_vars_total,
                                                encoder_kernel = mg_encoder_kernel[k],
                                                encoder_simul = mg_encoder_simul[k],
                                                processing_method = mg_spa_method[k],
                                                processing_min_lvl = mg_spa_min_lvl[k],
                                                decoder_kernel = mg_decoder_kernel[k],
                                                seq_level_attention = seq_lvl_att, 
                                                nh = nh,
                                                n_heads=n_heads, 
                                                encoder_channel_attention=mg_encoder_channel_attention[k],
                                                decoder_channel_attention=mg_decoder_channel_attention[k],
                                                n_chunks_channels_encoder=mg_encoder_n_chunks_channel[k],
                                                n_chunks_channels_decoder=mg_decoder_n_chunks_channel[k],
                                                pos_emb_calc='cartesian_km',
                                                emb_table_bins=16,
                                                output_layer=True if k==n_blocks-1 else False))
        
        

       # output_projection = self.model_settings['output_projection']
        #self.output_layer = output_layer(output_mapping['cell']['cell'], output_coordinates['cell'], [self.global_levels[-1]], grid_layers, self.model_settings, mode=output_projection)

        """
        trained_iterations = None
        if "pretrained_path" in self.model_settings.keys():
            trained_iterations = self.check_pretrained(log_dir_check=self.model_settings['pretrained_path'], strict=strict)

        if "pretrained_pos_embeddings_path" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_pos_embeddings_path'], strict=False, match_list='pos_embedder')

        if "pretrained_model_wo_input" in self.model_settings.keys():
            self.check_pretrained(log_dir_check=self.model_settings['pretrained_model_wo_input'], strict=False, not_match='input')

        self.trained_iterations = trained_iterations
        """

    def forward(self, x, coords_input=None, coords_output=None, sampled_indices_batch_dict=None, drop_mask=None):
        # if global_indices are provided, batches in x are treated as independent

        b,n = x.shape[:2]
        x = x.view(b,n,-1,self.model_dim_in)

        if drop_mask is not None:
            drop_mask = drop_mask[:,:,:,:x.shape[2]]

        if sampled_indices_batch_dict is None:
            sampled_indices_batch_dict = {'global_cell': self.global_indices,
                                  'local_cell': self.global_indices,
                                   'sample': None,
                                   'sample_level': None,
                                   'output_indices': None}
        else:
            indices_layers = dict(zip(self.global_levels.tolist(),[self.get_global_indices_local(sampled_indices_batch_dict['sample'], sampled_indices_batch_dict['sample_level'], global_level) for global_level in self.global_levels]))
            #indices_layers['sample_level'] = sampled_indices_batch_dict['sample_level']

        if coords_input.numel()==0:
            coords_input = self.cell_coords_global[:,sampled_indices_batch_dict['local_cell']].unsqueeze(dim=-1)

        if coords_output.numel()==0:
            coords_output = self.cell_coords_global[:,sampled_indices_batch_dict['local_cell']].unsqueeze(dim=-1)

        for k, multi_grid_block in enumerate(self.MGBlocks):
            
            drop_mask = None if k>0 else drop_mask

            coords_in = coords_input if k==0 else None
            coords_out = coords_output if k==len(self.MGBlocks)-1  else None
            
            x, drop_mask = multi_grid_block(x, indices_layers, sampled_indices_batch_dict, mask=drop_mask, coords_in=coords_in, coords_out=coords_out)
       
        x = x.view(b,n,-1)
        return x



    def get_data_from_ds(self, ds, ts, variables_dict, global_level_start, global_indices):
        
        sampled_data = {}
        for key, variables in variables_dict.items():
            data_g = []
            for variable in variables:
                data = torch.tensor(ds[variable][ts].values)
                data = data[0] if data.dim() > 1  else data
                data_g.append(data)

            data_g = torch.stack(data_g, dim=-1)

            indices = global_indices

            data_g = data_g[self.input_mapping['cell']['cell'][indices]]
            data_g = data_g.view(indices.shape[0], indices.shape[1], -1, len(variables))

            sampled_data[key] = data_g

        return sampled_data        

    def get_grid_mappings(self, mgrid_coords_input, mgrid_coords_output):
        
        indices_path = os.path.join(self.model_settings["model_dir"],"indices_data.pickle")

        if not os.path.isfile(indices_path):

            input_mapping, input_in_range = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                        self.model_settings['input_grid'], self.input_data, 
                                        search_radius=self.model_settings['search_raadius'], 
                                        max_nh=self.model_settings['nh_input'], 
                                        lowest_level=0,
                                        coords_icon=mgrid_coords_input,
                                        scale_input = self.scale_input,
                                        periodic_fov= self.model_settings['clon_fov'] if ('input_periodicty' in self.model_settings.keys() and self.model_settings['input_periodicty']) else None
                                        )

            output_mapping, output_in_range = get_nh_variable_mapping_icon(self.model_settings['processing_grid'], ['cell'], 
                                        self.model_settings['output_grid'], self.output_data, 
                                        search_radius=self.model_settings['search_radius'], 
                                        max_nh=1, 
                                        lowest_level=0,
                                        reverse_last=False,
                                        coords_icon=mgrid_coords_output,
                                        scale_input = self.scale_output,
                                        periodic_fov= self.model_settings['clon_fov'] if ('input_periodicty' in self.model_settings.keys() and self.model_settings['input_periodicty']) else None
                                        )
            
        else:
            with open(indices_path, 'rb') as handle:
                indices_data = pickle.load(handle)
            
            input_mapping = mapping_to_(indices_data['input_mapping'], to='pytorch')
            input_in_range = mapping_to_(indices_data['input_in_range'], to='pytorch')

            output_mapping = mapping_to_(indices_data['output_mapping'], to='pytorch')
            output_in_range = mapping_to_(indices_data['output_in_range'], to='pytorch')

        input_coordinates = {}
        for grid_type in self.input_data.keys():
            input_coordinates[grid_type] = scale_coordinates(get_coords_as_tensor(xr.open_dataset(self.model_settings['input_grid']), grid_type=grid_type), self.scale_input)
            

        output_coordinates = {}
        for grid_type in self.output_data.keys():
            output_coordinates[grid_type] = scale_coordinates(get_coords_as_tensor(xr.open_dataset(self.model_settings['output_grid']), grid_type=grid_type), self.scale_output)

        
        return input_mapping, input_in_range, input_coordinates, output_mapping, output_in_range, output_coordinates


    def get_nh_indices(self, global_level, global_cell_indices=None, local_cell_indices=None, adjc_global=None):
        
        if adjc_global is not None:
            adjc_global = self.get_adjacent_global_cell_indices(global_level)

        if global_cell_indices is not None:
            local_cell_indices =  global_cell_indices // 4**global_level

        local_cell_indices_nh, mask = helpers.get_nh_of_batch_indices(local_cell_indices, adjc_global)

        return local_cell_indices_nh, mask



    def get_global_indices_global(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        
        return self.get_global_indices_relative(global_indices_sampled, global_level)
    
    def get_global_indices_local(self, batch_sample_indices, sampled_level_fov, global_level):

        global_indices_sampled  = self.global_indices.view(-1, 4**sampled_level_fov[0])[batch_sample_indices]
        global_indices_sampled = self.get_global_indices_relative(global_indices_sampled, global_level)    
        return global_indices_sampled // 4**global_level
    
    def get_global_indices_relative(self, sampled_indices, level):
        return sampled_indices.view(sampled_indices.shape[0], -1, 4**level)[:,:,0]
    

    def localize_global_indices(self, sample_indices_dict, level):
        
        b,n = sample_indices_dict['global_cell'].shape[:2]
        indices_offset_level = sample_indices_dict['sample']*4**(sample_indices_dict['sample_level']-level)
        indices_level = sample_indices_dict['global_cell'].view(b,n) - indices_offset_level.view(-1,1)

        return indices_level
    
    def coarsen_indices(self, global_level, coarsen_level=None, indices=None, nh=1):
        if indices is None:
            indices = self.global_indices

        global_cells, local_cells, cells_nh, out_of_fov_mask = helpers.coarsen_global_cells(indices, self.eoc, self.acoe, global_level=global_level, coarsen_level=coarsen_level, nh=nh)
        

        return global_cells, local_cells, cells_nh, out_of_fov_mask 
    

    def get_adjacent_global_cell_indices(self, global_level, nh=2):
        adjc, mask = icon_get_adjacent_cell_indices(self.acoe, self.eoc, nh=nh, global_level=global_level)

        return adjc, mask


def sequenize(tensor, max_seq_level):
    
    seq_level = min([get_max_seq_level(tensor), max_seq_level])
    
    if tensor.dim()==2:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level))
    elif tensor.dim()==3:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-1])
    elif tensor.dim()==4:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-2], tensor.shape[-1])
    elif tensor.dim()==5:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])
    elif tensor.dim()==6:
        tensor = tensor.view(tensor.shape[0], -1, 4**(seq_level), tensor.shape[-4], tensor.shape[-3], tensor.shape[-2], tensor.shape[-1])

    return tensor

def get_max_seq_level(tensor):
    seq_len = tensor.shape[1]
    max_seq_level_seq = int(math.log(seq_len)/math.log(4))
    return max_seq_level_seq


def get_nh_indices(adjc_global, global_level, global_cell_indices=None, local_cell_indices=None):
        
    if global_cell_indices is not None:
        local_cell_indices =  global_cell_indices // 4**global_level

    local_cell_indices_nh, mask = helpers.get_nh_of_batch_indices(local_cell_indices, adjc_global)

    return local_cell_indices_nh, mask


def gather_nh_data(x, local_cell_indices_nh, batch_sample_indices, sampled_level, global_level):
    # x in batches sampled from local_cell_indices_nh
    if x.dim()<4:
        x = x.unsqueeze(dim=-1)

    b,n,nv,e = x.shape
    nh = local_cell_indices_nh.shape[-1]

    local_cell_indices_nh_batch = local_cell_indices_nh - (batch_sample_indices*4**(sampled_level - global_level)).view(-1,1,1)

    return torch.gather(x.view(b,-1,nv,e),1, index=local_cell_indices_nh_batch.view(b,-1,1,1).repeat(1,1,nv,e)).view(b,n,nh,nv,e)
