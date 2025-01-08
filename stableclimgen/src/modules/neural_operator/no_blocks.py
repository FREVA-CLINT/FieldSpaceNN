from typing import List, Dict

import torch
import torch.nn.functional as F
import torch.nn as nn


from .neural_operator import NoLayer
from ..transformer.attention import ChannelVariableAttention, CrossChannelVariableAttention, ResLayer, AdaptiveLayerNorm
from ...models.mgno_transformer.mg_attention import MultiGridAttention

from ..icon_grids.grid_attention import GridAttention
from ...modules.embedding.embedder import EmbedderSequential, EmbedderManager, BaseEmbedder

class NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 att_block_types:list,
                 n_params=list,
                 n_head_channels = 16,
                 att_dim=None,
                 p_dropout=0.,
                 no_layer_nh: NoLayer=None,
                 is_decode:list=[],
                 embed_names: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum",
                 spatial_attention_config={}
                ) -> None: 
      
        super().__init__()

        self.model_dim_in = model_dim_in
        self.n_params = n_params
        self.no_layer = no_layer
        self.att_block_types_encode = nn.ModuleList()
        self.att_block_types_decode = nn.ModuleList()

        self.prepare_coordinates = False

        for k, att_block_type in enumerate(att_block_types):

            if len(embed_names[k])>0 and not 'trans' in att_block_type:
                emb_dict = nn.ModuleDict()
                for embed_name in embed_names[k]:
                    emb: BaseEmbedder = EmbedderManager().get_embedder(embed_name, **embed_confs[embed_name])
                    emb_dict[emb.name] = emb     
                embedder_seq = EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = 1)
                embedder_mlp = EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = 1)

                if 'CoordinateEmbedder' in emb_dict.keys():
                    self.prepare_coordinates = True
                    self.grid_layer = no_layer.grid_layers[str(no_layer.global_level_no)]
                    self.global_level = no_layer.global_level_no
            else:
                embedder_seq = None
                embedder_mlp = None

            if 'param' in att_block_type:
                param_att_idx = int(att_block_type.replace('param',''))
                layer = ParamAttention(model_dim_in, 
                                       n_head_channels,
                                       att_dim=att_dim,
                                       p_dropout=p_dropout,
                                       n_params=n_params, 
                                       param_idx_att=param_att_idx,
                                       embedder=embedder_seq,
                                       embedder_mlp=embedder_mlp)
            elif 'var' in att_block_type:

                if 'self_cross_var' in att_block_type:
                    block = SelfCrossAttention
                elif 'cross' in att_block_type:
                    block = CrossAttention
                else:
                    block = ParamAttention
    
                layer = block(model_dim_in, 
                                n_head_channels,
                                att_dim=att_dim,
                                p_dropout=p_dropout,
                                n_params=n_params, 
                                param_idx_att=None,
                                embedder=embedder_seq,
                                embedder_mlp=embedder_mlp)

            elif 'nh' in att_block_type:
                layer = NHAttention(no_layer_nh,
                                    model_dim_in, 
                                    n_head_channels,
                                    p_dropout=p_dropout,
                                    n_params=n_params,
                                    embedder=embedder_seq,
                                    embedder_mlp=embedder_mlp)
            
            elif 'trans' in att_block_type:
                spatial_attention_config['embedder_names'] = [embed_names[k], []]
                spatial_attention_config['embed_confs'] = embed_confs
                spatial_attention_config['embed_mode'] = embed_mode
                layer = SpatialAttention(no_layer,
                                         model_dim_in, 
                                         n_head_channels, 
                                         n_params,
                                         spatial_attention_config,
                                         p_dropout=p_dropout)

            if len(is_decode)>0 and is_decode[k]:
                self.att_block_types_decode.append(layer)
            else:
                self.att_block_types_encode.append(layer)


    def check_add_coordinate_embeddings(self, emb, indices_sample):
  
        coords = self.grid_layer.get_coordinates_from_grid_indices(
            indices_sample['indices_layers'][self.global_level] if indices_sample is not None else None)
        emb['CoordinateEmbedder'] = coords
        return emb


    def encode(self, x, coords_in=None, indices_sample=None, mask=None, emb=None):

        x, mask = self.no_layer.transform(x, coordinates=coords_in, indices_sample=indices_sample, mask=mask)

        for layer in self.att_block_types_encode:
            if isinstance(layer, SpatialAttention):
                x = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            elif isinstance(layer, NHAttention):
                x, mask = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            else:
                emb = self.check_add_coordinate_embeddings(emb, indices_sample) if self.prepare_coordinates else emb
                x, mask = layer(x, mask=mask, emb=emb)

        return x, mask


    def decode(self, x, coords_out=None, indices_sample=None, mask=None, emb=None, inv_transform_mask=None):
        
        for layer in self.att_block_types_decode:
            if isinstance(layer, SpatialAttention):
                x = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            elif isinstance(layer, NHAttention):
                x, mask = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            else:
                emb = self.check_add_coordinate_embeddings(emb, indices_sample) if self.prepare_coordinates else emb
                x, mask = layer(x, mask=mask, emb=emb)

        if mask is not None:
            if inv_transform_mask is None:
                inv_transform_mask = mask 

        x, mask = self.no_layer.inverse_transform(x, coordinates=coords_out, indices_sample=indices_sample, mask=mask)

        return x, mask
    

    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):

        x, mask = self.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        x, mask = self.decode(x, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

        return x, mask



class Serial_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dims_out,
                 no_layers: list[NoLayer],
                 att_block_types_encode:list,
                 att_block_types_decode:list,
                 n_params=list,
                 n_head_channels = 16,
                 att_dims: list=[],
                 no_layers_nh: list[NoLayer]=[None],
                 multi_grid_attention: bool=False,
                 embed_names_encode: List[List[str]] = None,
                 embed_names_decode: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum",
                 spatial_attention_configs: dict = {},
                 p_dropout=0.
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention
        self.NO_Blocks = nn.ModuleList()

        for n, no_layer in enumerate(no_layers): 

            att_block_types = att_block_types_encode[n] + att_block_types_decode[n]
            embed_names = embed_names_encode[n] + embed_names_decode[n]

            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                model_dim_in=model_dim_in,
                model_dim_out=model_dims_out[n],
                att_block_types=att_block_types,
                n_params=n_params[n],
                n_head_channels=n_head_channels,
                att_dim=att_dims[n],
                no_layer_nh=no_layers_nh[n],
                embed_names=embed_names,
                embed_confs=embed_confs[n],
                embed_mode=embed_mode[n],
                spatial_attention_config=spatial_attention_configs[n],
                p_dropout=p_dropout
            ))

        if multi_grid_attention:
            self.mga_layer = MultiGridAttention(model_dim_in, model_dims_out[n], len(no_layers), att_dim=att_dims[-1], n_head_channels=n_head_channels)
    
    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        x_mg = []
        mask_mg = []
        for layer in self.NO_Blocks:
            x, mask = layer(x, coords_in=coords_in, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)
        
            if self.multi_grid_attention:
                x_mg.append(x)
                mask_mg.append(mask)
        
        if self.multi_grid_attention:
            x = self.mga_layer(x_mg, mask_mg)
        
        return x, mask

class Stacked_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dims_out,
                 no_layers: list[NoLayer],
                 att_block_types_encode:list,
                 att_block_types_decode:list,
                 n_params=list,
                 n_head_channels = 16,
                 att_dims: list=[],
                 no_layers_nh: list[NoLayer]=[None],
                 multi_grid_attention: bool=False,
                 embed_names_encode: List[List[str]] = None,
                 embed_names_decode: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum",
                 spatial_attention_configs = {},
                 p_dropout=0.
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention
        self.NO_Blocks = nn.ModuleList()

        for n, no_layer in enumerate(no_layers): 
            
            is_decode_encode = [False for _ in range(len(att_block_types_encode[n]))]
            is_decode_decode = [True for _ in range(len(att_block_types_decode[n]))]
            is_decode = is_decode_encode + is_decode_decode
            embed_names = embed_names_encode[n] + embed_names_decode[n]

            att_block_types = att_block_types_encode[n] + att_block_types_decode[n]

            model_dim_in = model_dim_in if n==0 else model_dim_out_encode
            model_dim_out_encode = int(model_dim_in*torch.tensor(n_params[n]).prod())

            
            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                model_dim_in=model_dim_in,
                model_dim_out=model_dims_out[n],
                att_block_types=att_block_types,
                is_decode=is_decode,
                n_params=n_params[n],
                n_head_channels=n_head_channels,
                att_dim=att_dims[n],
                no_layer_nh=no_layers_nh[n],
                embed_names=embed_names,
                embed_confs=embed_confs[n],
                embed_mode=embed_mode[n],
                spatial_attention_config=spatial_attention_configs[n],
                p_dropout=p_dropout
            ))

        self.model_dim_out_encode = model_dim_out_encode

    def encode(self, x, coords_in=None, indices_sample=None, mask=None, emb=None):

        for layer in self.NO_Blocks:
            x, mask = layer.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)
            x, mask = shape_to_att(x, mask=mask)

        return x, mask
    
    def decode(self, x, coords_out=None, indices_sample=None, mask=None, emb=None):

        for layer_idx in range(len(self.NO_Blocks)-1,-1,-1):
            layer = self.NO_Blocks[layer_idx]
            x_shape_origin = (*x.shape[:3], *layer.n_params, -1)
            x = shape_to_x(x, x_shape_origin)
            x, mask = layer.decode(x, coords_out=coords_out, indices_sample=indices_sample, emb=emb, mask=mask)

        return x, mask


    
    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        x, mask = self.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        x, mask = self.decode(x, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

        return x, mask


class UNet_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dims_out,
                 no_layers: list[NoLayer],
                 att_block_types_encode:list,
                 att_block_types_decode:list,
                 n_params=list,
                 n_head_channels = 16,
                 att_dims: list=[],
                 no_layers_nh: list[NoLayer]=[None],
                 multi_grid_attention: bool=False,
                 embed_names_encode: List[List[str]] = None,
                 embed_names_decode: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum",
                 spatial_attention_configs = {},
                 p_dropout=0.
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention
        self.NO_Blocks = nn.ModuleList()

        for n, no_layer in enumerate(no_layers): 
            
            is_decode_encode = [False for _ in range(len(att_block_types_encode[n]))]
            is_decode_decode = [True for _ in range(len(att_block_types_decode[n]))]
            is_decode = is_decode_encode + is_decode_decode

            embed_names = embed_names_encode[n] + embed_names_decode[n]
            att_block_types = att_block_types_encode[n] + att_block_types_decode[n]

            model_dim_in = model_dim_in if n==0 else model_dim_out_encode

            model_dim_out_encode = int(model_dim_in*torch.tensor(n_params[n]).prod())

            
            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                model_dim_in=model_dim_in,
                model_dim_out=model_dims_out[n],
                att_block_types=att_block_types,
                is_decode=is_decode,
                n_params=n_params[n],
                n_head_channels=n_head_channels,
                att_dim=att_dims[n],
                no_layer_nh=no_layers_nh[n],
                embed_names=embed_names,
                embed_confs=embed_confs[n],
                embed_mode=embed_mode[n],
                spatial_attention_config=spatial_attention_configs[n],
                p_dropout=p_dropout
            ))

        self.model_dim_out_encode = model_dim_out_encode

    def encode(self, x, coords_in=None, indices_sample=None, mask=None, emb=None):
        
        x_skip = []
        mask_skip = []
        for layer in self.NO_Blocks:
            x, mask = layer.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)
            x, mask = shape_to_att(x, mask=mask)

            x_skip.append(x)
            mask_skip.append(mask)

        return x_skip, mask_skip
    
    def decode(self, x_skip, masks_skip, coords_out=None, indices_sample=None, emb=None):
        
        mask = masks_skip[-1] 
        for layer_idx in range(len(self.NO_Blocks)-1,-1,-1):
            
            layer = self.NO_Blocks[layer_idx]
            x_shape_origin = (*x_skip[layer_idx].shape[:3], *layer.n_params, -1)

            if layer_idx<len(self.NO_Blocks)-1:
               x = torch.cat((shape_to_x(x, x_shape_origin),
                               shape_to_x(x_skip[layer_idx], x_shape_origin)), dim=2)
               mask = torch.cat([mask, masks_skip[layer_idx]], dim=2)
            else:
                x = shape_to_x(x_skip[layer_idx], x_shape_origin)
            
            x, mask = layer.decode(x, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb, inv_transform_mask=mask)

        return x


    
    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        x_skip, mask_skip = self.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        x = self.decode(x_skip, mask_skip, coords_out=coords_out, indices_sample=indices_sample, emb=emb)

        return x, mask


class Parallel_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dims_out,
                 no_layers: list[NoLayer],
                 att_block_types_encode:list,
                 att_block_types_decode:list,
                 n_params=list,
                 n_head_channels = 16,
                 att_dims: list=[],
                 no_layers_nh: list[NoLayer]=[None],
                 multi_grid_attention: bool=False,
                 embed_names_encode: List[List[str]] = None,
                 embed_names_decode: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum",
                 spatial_attention_configs = {},
                 p_dropout=0.
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention
        self.NO_Blocks = nn.ModuleList()

        for n, no_layer in enumerate(no_layers): 

            att_block_types = att_block_types_encode[n] + att_block_types_decode[n]
            embed_names = embed_names_encode[n] + embed_names_decode[n]

            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                model_dim_in=model_dim_in,
                model_dim_out=model_dims_out[n],
                att_block_types=att_block_types,
                n_params=n_params[n],
                n_head_channels=n_head_channels,
                att_dim=att_dims[n],
                no_layer_nh=no_layers_nh[n],
                embed_names=embed_names,
                embed_confs=embed_confs[n],
                embed_mode=embed_mode[n],
                spatial_attention_config = spatial_attention_configs[n],
                p_dropout=p_dropout
            ))

        if multi_grid_attention:
            self.mga_layer = MultiGridAttention(model_dim_in, model_dims_out[n], len(no_layers), att_dim=att_dims[-1], n_head_channels=n_head_channels)
    
    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        x_mg = []
        mask_mg = []
        for layer in self.NO_Blocks:
            x_enc, mask_enc = layer.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)
            x_dec, mask_dec = layer.decode(x_enc, coords_out=coords_out, indices_sample=indices_sample, mask=mask_enc, emb=emb)

            if self.multi_grid_attention:
                x_mg.append(x_dec)
                mask_mg.append(mask_dec)
        
        if self.multi_grid_attention:
            x = self.mga_layer(x_mg, mask_mg)
        
        return x, mask


class SpatialAttention(nn.Module):
  
    def __init__(self,
                 no_layer: NoLayer,
                 model_dim_in, 
                 n_head_channels,
                 n_params, 
                 spatial_attention_configs = None,
                 p_dropout=0.
                ) -> None: 
      
        super().__init__()

        ch_in = int(model_dim_in*torch.tensor(n_params).prod())

        self.grid_attention_layer = GridAttention(
            no_layer.grid_layers[str(no_layer.global_level_no)],
            ch_in,
            ch_in,
            n_head_channels=n_head_channels,
            spatial_attention_configs=spatial_attention_configs,
            rotate_coord_system=no_layer.rotate_coord_system
        )
       
    def forward(self, x, indices_sample=None, mask=None, emb=None):

        x_shape = x.shape
        x, mask = shape_to_att(x, mask=mask)
        x = x.unsqueeze(dim=1)
        mask = mask.unsqueeze(dim=1) if mask is not None else mask
        x = self.grid_attention_layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
        x = x.squeeze(dim=1)
        x = shape_to_x(x, x_shape)
        return x


class ParamAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in, 
                 n_head_channels, 
                 att_dim=None,
                 p_dropout = 0,
                 n_params = [],
                 param_idx_att = None,
                 embedder: BaseEmbedder=None,
                 embedder_mlp: BaseEmbedder=None
                ) -> None: 
      
        super().__init__()

        model_dim_total = model_dim_in*torch.tensor(n_params).prod()
        model_dim_param = model_dim_total/float(n_params[param_idx_att]) if param_idx_att is not None else model_dim_total
        model_dim_param = int(model_dim_param)

        att_dim = att_dim if att_dim is not None and model_dim_param < att_dim else model_dim_param

        self.attention_layer = ChannelVariableAttention(model_dim_param, 1, n_head_channels, att_dim=att_dim, with_res=False)

        self.ada_ln = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_total, embedder=embedder)
        self.ada_ln_mlp = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_total, embedder=embedder_mlp)
       
        self.gamma = nn.Parameter(torch.ones(model_dim_param)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim_param)*1e-6, requires_grad=True)

        self.res_layer = ResLayer(model_dim_param, with_res=False, p_dropout=p_dropout)

        self.param_idx_att = param_idx_att

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()


    def forward(self, x, mask=None, emb=None):
        x_shape = x.shape

        x_res, mask = shape_to_att(x, self.param_idx_att, mask=mask)

        x = self.ada_ln(x, emb=emb)

        x, _ = shape_to_att(x, self.param_idx_att)
        
        x, mask  = self.attention_layer(x, mask=mask)

        x = self.dropout(x)

        x = x_res + self.gamma*x

        x_res = x

        x = shape_to_x(x, x_shape, self.param_idx_att)  
        x = self.ada_ln_mlp(x, emb=emb)
        x, _ = shape_to_att(x, self.param_idx_att)

        x = self.res_layer(x)

        x = x_res + self.gamma_mlp*x

        x = shape_to_x(x, x_shape, self.param_idx_att)

        if mask is not None:
            mask = mask.view(*x.shape[:3],-1)
            mask = mask.sum(dim=-1).bool()
            

        return x, mask
    
class SelfCrossAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in, 
                 n_head_channels, 
                 att_dim=None,
                 p_dropout = 0,
                 n_params = [],
                 param_idx_att = None,
                 embedder: BaseEmbedder=None,
                 embedder_mlp: BaseEmbedder=None
                ) -> None: 
      
        super().__init__()

        model_dim_total = model_dim_in*torch.tensor(n_params).prod()
        model_dim_param = model_dim_total/float(n_params[param_idx_att]) if param_idx_att is not None else model_dim_total
        model_dim_param = int(model_dim_param)

        att_dim = att_dim if att_dim is not None and model_dim_param < att_dim else model_dim_param

        self.attention_layer = CrossChannelVariableAttention(model_dim_param, 1, n_head_channels, att_dim=att_dim, with_res=False)

        self.ada_ln_q = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_param, embedder=embedder)
        self.ada_ln_kv1 = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_param, embedder=embedder)
        self.ada_ln_kv2 = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_param, embedder=embedder)

        self.ada_ln_mlp = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_param, embedder=embedder_mlp)

        self.gamma = nn.Parameter(torch.ones(model_dim_param)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim_param)*1e-6, requires_grad=True)

        self.res_layer = ResLayer(model_dim_param, with_res=False, p_dropout=p_dropout)

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()

        self.res_gamma = nn.Parameter(torch.tensor(1e-6), requires_grad=True)

        self.res_gamma_activation = nn.Sigmoid()

        self.param_idx_att = param_idx_att

    def forward(self, x, mask=None, emb=None):
  
        x_shape = x.shape

        res_gamma = self.res_gamma_activation(self.res_gamma)

        if mask is not None:
            f_mask = torch.stack((mask==False).chunk(2,dim=2), dim=3) * torch.cat([1-res_gamma.view(-1), res_gamma.view(-1)])
            f_mask = f_mask/(f_mask+1e-10).sum(dim=-1, keepdim=True)
            x_q = f_mask.view(*f_mask.shape,1,1,1) * torch.stack(x.chunk(2,dim=2), dim=3)
        else:
            x_q = torch.stack(x.chunk(2,dim=2), dim=3) * torch.cat([1-res_gamma.view(-1), res_gamma.view(-1)]).view(1,1,2,1,1,1)

        x_q = x_q.sum(dim=3)

        x_shape = x_q.shape

        x_res = x_q

        x_q = self.ada_ln_q(x_q, emb=emb).contiguous()

        x1, x2 = x.chunk(2, dim=2) 
        x1 = self.ada_ln_kv1(x1, emb=emb).contiguous()
        x2 = self.ada_ln_kv2(x2, emb=emb).contiguous()
        x = torch.concat((x1,x2), dim=2)
        
        x_res , _ = shape_to_att(x_res, self.param_idx_att)
        x_q, _ = shape_to_att(x_q, self.param_idx_att, mask=mask)
        x, mask = shape_to_att(x, self.param_idx_att, mask=mask)
        
        x, _  = self.attention_layer(x_q, x, mask=mask)

        x = self.dropout(x)

        x = x_res + self.gamma*x

        x_res = x

        x = shape_to_x(x, x_shape, self.param_idx_att)
        x = self.ada_ln_mlp(x, emb=emb)
        x, _ = shape_to_att(x, self.param_idx_att)
    
        x = self.res_layer(x)

        x = x_res + self.gamma_mlp*x

        x = shape_to_x(x, x_shape, self.param_idx_att)

        if mask is not None:
            mask = mask.chunk(2,dim=-1)[0]

        return x, mask
    
class CrossAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in, 
                 n_head_channels, 
                 att_dim=None,
                 p_dropout = 0,
                 n_params = [],
                 param_idx_att = None,
                 embedder: BaseEmbedder=None,
                 embedder_mlp: BaseEmbedder=None
                ) -> None: 
      
        super().__init__()

        model_dim_total = model_dim_in*torch.tensor(n_params).prod()
        model_dim_param = model_dim_total/float(n_params[param_idx_att]) if param_idx_att is not None else model_dim_total
        model_dim_param = int(model_dim_param)

        att_dim = att_dim if att_dim is not None and model_dim_param < att_dim else model_dim_param

        self.attention_layer = CrossChannelVariableAttention(model_dim_param, 1, n_head_channels, att_dim=att_dim, with_res=False)

        self.ada_ln = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_param, embedder=embedder)
        self.ada_ln_cross = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_param, embedder=embedder)

        self.ada_ln_mlp = AdaptiveLayerNorm(n_params + [int(model_dim_in)], model_dim_param, embedder=embedder_mlp)

        self.gamma = nn.Parameter(torch.ones(model_dim_param)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim_param)*1e-6, requires_grad=True)

        self.res_layer = ResLayer(model_dim_param, with_res=False, p_dropout=p_dropout)

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()

        self.param_idx_att = param_idx_att

    def forward(self, x, mask=None, emb=None):
        x, x_cross = x.chunk(2,dim=2)
        x_shape = x.shape

        if mask is not None:
            _, mask = mask.chunk(2,dim=2)

        x_res, mask = shape_to_att(x.contiguous(), self.param_idx_att, mask=mask)

        x = self.ada_ln(x, emb=emb)
        x_cross_v = x_cross
        x_cross = self.ada_ln_cross(x_cross, emb=emb)
        
        x, _ = shape_to_att(x.contiguous(), self.param_idx_att)
        x_cross, _ = shape_to_att(x_cross.contiguous(), self.param_idx_att)
        x_cross_v, _ = shape_to_att(x_cross_v.contiguous(), self.param_idx_att)

        x, _  = self.attention_layer(x, x_cross, x_cross_v=x_cross_v, mask=mask)

        x = self.dropout(x)

        x = x_res + self.gamma*x

        x_res = x

        x = shape_to_x(x, x_shape, self.param_idx_att)
        x = self.ada_ln_mlp(x, emb=emb)
        x, _ = shape_to_att(x, self.param_idx_att)
    
        x = self.res_layer(x)

        x = x_res + self.gamma_mlp*x

        x = shape_to_x(x, x_shape, self.param_idx_att)

        return x, mask


class NHAttention(nn.Module):
  
    def __init__(self,
                 no_layer_nh: NoLayer,
                 model_dim_in, 
                 n_head_channels, 
                 p_dropout = 0,
                 n_params = [],
                 embedder: BaseEmbedder=None,
                 embedder_mlp: BaseEmbedder=None
                ) -> None: 
      
        super().__init__()

        model_dim_in = model_dim_in*torch.tensor(n_params).prod()

        self.no_layer_nh = no_layer_nh

        self.attention_layer = ParamAttention(model_dim_in, 
                                              n_head_channels, 
                                              att_dim=model_dim_in, 
                                              p_dropout=p_dropout, 
                                              param_idx_att=None, 
                                              n_params=no_layer_nh.n_params, 
                                              embedder=embedder, 
                                              embedder_mlp=embedder_mlp)

    def forward(self, x, indices_sample=None, mask=None, emb=None):
        x_shape = x.shape
        x, mask = shape_to_att(x, mask=mask)

        x, mask = self.no_layer_nh.transform(x, indices_sample=indices_sample, mask=mask)

        x, mask = self.attention_layer(x, mask=mask, emb=emb)
 
        x, mask = self.no_layer_nh.inverse_transform(x, indices_sample=indices_sample, mask=mask)

        x = shape_to_x(x, x_shape)

        return x, mask



def shape_to_att(x, param_att: int=None, mask=None):
    b,n,nv = x.shape[:3]
    param_shape = x.shape[3:]
    n_c = x.shape[-1]

    if param_att == 0:
        x =  x.view(b,n*nv,param_shape[0],-1)
    
    elif param_att==1:
        x = x.transpose(3,4).contiguous()
        x = x.view(b,n*nv,param_shape[1],-1)
    
    elif param_att==2:
        x = x.transpose(3,5)
        x = x.view(b,n*nv,param_shape[2],-1)
    else:
        x = x.view(b,n,nv,-1)

    if mask is not None:
        mask = mask.view(b,x.shape[1],-1)
        mask = mask.repeat_interleave(x.shape[2]//mask.shape[-1],dim=-1)

    return x, mask
    

def shape_to_x(x, x_shape_orig:tuple, param_att: int=None):

    if param_att == 0 or param_att is None:
        x =  x.view(x_shape_orig)
    
    elif param_att==1:
        x_shape_flat = (*x_shape_orig[:3],x_shape_orig[4], x_shape_orig[3], *x_shape_orig[5:])   
        x = x.view(x_shape_flat).transpose(3,4).contiguous()
        
    elif param_att==2:
        x_shape_flat = (*x_shape_orig[:3], x_shape_orig[5], x_shape_orig[4], x_shape_orig[3], *x_shape_orig[6:])   
        x = x.view(x_shape_flat).transpose(3,5).contiguous()

    return x  