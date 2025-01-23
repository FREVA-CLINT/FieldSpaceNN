from typing import List, Dict

import torch
import torch.nn.functional as F
import torch.nn as nn


from .neural_operator import NoLayer, polNormal_NoLayer
from ..transformer.attention import ChannelVariableAttention, CrossChannelVariableAttention, ResLayer, AdaptiveLayerNorm
from ...models.mgno_transformer.mg_attention import MultiGridAttention

from ..icon_grids.grid_attention import GridAttention
from ...modules.embedding.embedder import EmbedderSequential, EmbedderManager, BaseEmbedder

class NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim,
                 no_layer: NoLayer,
                 att_block_types:list,
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

        self.model_dim = model_dim
        self.n_params = no_layer.n_params_no
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
                layer = ParamAttention(self.n_params, 
                                       n_head_channels,
                                       att_dim=att_dim,
                                       p_dropout=p_dropout,
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
    
                layer = block(self.n_params, 
                                n_head_channels,
                                att_dim=att_dim,
                                p_dropout=p_dropout,
                                param_idx_att=None,
                                embedder=embedder_seq,
                                embedder_mlp=embedder_mlp)

            elif 'nh' in att_block_type:
                layer = NHAttention(no_layer_nh,
                                    model_dim, 
                                    n_head_channels,
                                    p_dropout=p_dropout,
                                    n_params=self.n_params,
                                    embedder=embedder_seq,
                                    embedder_mlp=embedder_mlp)
            
            elif 'trans' in att_block_type:
                spatial_attention_config['embedder_names'] = [embed_names[k], []]
                spatial_attention_config['embed_confs'] = embed_confs
                spatial_attention_config['embed_mode'] = embed_mode
                layer = SpatialAttention(no_layer,
                                         model_dim, 
                                         n_head_channels, 
                                         self.n_params,
                                         spatial_attention_config,
                                         p_dropout=p_dropout)

            if len(is_decode)>0 and is_decode[k]:
                self.att_block_types_decode.append(layer)
            else:
                self.att_block_types_encode.append(layer)

    def squeeze_no_dims(self, x):
        x = x.view(*x.shape[:3],-1,self.n_params[-1])
        return x

    def unsqueeze_no_dims(self, x):
        x = x.view(*x.shape[:3], *self.n_params[:-1], -1, x.shape[-1])
        return x

    def check_add_coordinate_embeddings(self, emb, indices_sample):
  
        coords = self.grid_layer.get_coordinates_from_grid_indices(
            indices_sample['indices_layers'][self.global_level] if indices_sample is not None else None)
        emb['CoordinateEmbedder'] = coords
        return emb


    def encode(self, x, coords_in=None, indices_sample=None, mask=None, emb=None, squeeze_no_dims=True):

        x, mask = self.no_layer.transform(x, coordinates=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        for layer in self.att_block_types_encode:
            if isinstance(layer, SpatialAttention):
                x = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            elif isinstance(layer, NHAttention):
                x, mask = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            else:
                emb = self.check_add_coordinate_embeddings(emb, indices_sample) if self.prepare_coordinates else emb
                x, mask = layer(x, mask=mask, emb=emb)
        
        if squeeze_no_dims:
            x = self.squeeze_no_dims(x)

        return x, mask


    def decode(self, x, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        if x.dim()==5:
            x = self.unsqueeze_no_dims(x)

        for layer in self.att_block_types_decode:
            if isinstance(layer, SpatialAttention):
                x = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            elif isinstance(layer, NHAttention):
                x, mask = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            else:
                emb = self.check_add_coordinate_embeddings(emb, indices_sample) if self.prepare_coordinates else emb
                x, mask = layer(x, mask=mask, emb=emb)


        x, mask = self.no_layer.inverse_transform(x, coordinates=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

        return x, mask
    

    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):

        x, mask = self.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        x, mask = self.decode(x, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

        return x, mask


class UNet_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 layer_settings: list,     
                 n_head_channels = 16,
                 multi_grid_attention: bool=False,
                 p_dropout=0.,
                 global_res=False,
                 skip_mode='amp'
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention

        self.Skip_Blocks = nn.ModuleList()
        self.NO_Blocks = nn.ModuleList()
        self.global_res=global_res

        encoding_dims = []
        decoding_dims = []
        for n, layer_setting in enumerate(layer_settings):
            encoding_dims.append(layer_setting['amplitude_dim_encode'])
            decoding_dims.append(layer_setting['amplitude_dim_decode'])

        for n, layer_setting in enumerate(layer_settings):
            
            no_layer = polNormal_NoLayer(
                layer_setting['grid_layer_in'],
                layer_setting['grid_layer_no'],
                n_amplitudes_in=model_dim_in if n==0 else encoding_dims[n-1],
                n_amplitudes_out=encoding_dims[n],
                n_amplitdues_inv_in= encoding_dims[-1] if n==len(layer_settings) -1 else decoding_dims[-(n+1)],
                n_amplitudes_inv_out=decoding_dims[-n],
                n_phi=layer_setting["n_params"][0],
                n_dist=layer_setting["n_params"][1],
                n_sigma=layer_setting["n_params"][2],
                avg_phi=layer_setting["avg_params"][0],
                avg_dist=layer_setting["avg_params"][1],
                avg_sigma=layer_setting["avg_params"][2],
                dist_learnable=layer_setting["global_params_learnable"][0],
                sigma_learnable=layer_setting["global_params_learnable"][1],
                amplitudes_learnable=layer_setting["global_params_learnable"][2],
                nh_projection=layer_setting["nh_transformation"], 
                nh_backprojection=layer_setting["nh_inverse_transformation"],
                precompute_coordinates=layer_setting["precompute_coordinates"],
                rotate_coord_system=layer_setting["rotate_coordinate_system"],
                pretrained_weights=layer_setting["pretrained_weights"],
                n_var_amplitudes=layer_setting["n_var_amplitudes"],
                with_res=layer_setting["with_res"]
            )
            
            is_decode_encode = [False for _ in range(len(layer_setting['block_types_encode']))]
            is_decode_decode = [True for _ in range(len(layer_setting['block_types_decode']))]
            is_decode = is_decode_encode + is_decode_decode

            embed_names = layer_setting['embed_names_encode'] + layer_setting['embed_names_decode']
            att_block_types = layer_setting['block_types_encode'] + layer_setting['block_types_decode']

            model_dim = int(torch.tensor(no_layer.n_params_no).prod())
            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                model_dim=model_dim,
                att_block_types=att_block_types,
                is_decode=is_decode,
                n_head_channels=n_head_channels,
                att_dim=layer_setting["att_dim"],
                no_layer_nh=None,
                embed_names=embed_names,
                embed_confs=layer_setting['embed_confs'] ,
                embed_mode=layer_setting['embed_mode'] ,
                spatial_attention_config=layer_setting.get("spatial_attention_configs",{}),
                p_dropout=p_dropout
            ))

        self.skip = False if len(skip_mode)==0 else True
        for n, layer_setting in enumerate(layer_settings):
            if n != len(layer_settings)-1:
                x_dims = self.NO_Blocks[-(n+1)].no_layer.n_params_out
                x_skip_dims = self.NO_Blocks[n].no_layer.n_params_no

                if skip_mode=='amp':
                    self.Skip_Blocks.append(
                        SkipAdd_Layer([x_dims[-1], x_skip_dims[-1]], x_dims[-1], n_var_amplitudes=layer_setting["n_var_amplitudes"])
                        )
                else:
                    self.Skip_Blocks.append(nn.Identity())
        
        if model_dim_out is None:
            model_dim_out = decoding_dims[0]
        
        if global_res:
            self.out_layer = SkipAdd_Layer([decoding_dims[0], model_dim_in], model_dim_out, n_var_amplitudes=layer_settings[-1]['n_var_amplitudes'])
        elif decoding_dims[0]!=model_dim_out:
            self.out_layer = VarLin_Layer(decoding_dims[0], model_dim_out, layer_settings[-1]['n_var_amplitudes'])
        else:
            self.out_layer = nn.Identity()

        self.model_dim_out = model_dim_out

    def encode(self, x, coords_in=None, indices_sample=None, mask=None, emb=None):
        
        x_skip = []
        mask_skip = []
        for layer in self.NO_Blocks:

            x_skip.append(x)
            mask_skip.append(mask)

            x, mask = layer.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        return x, mask, x_skip, mask_skip
    
    def decode(self, x, mask, x_skip=None, masks_skip=None, coords_out=None, indices_sample=None, emb=None):
        
        for layer_idx in range(len(self.NO_Blocks)-1,-1,-1):

            if layer_idx<len(self.NO_Blocks)-1 and x_skip is not None and self.skip:
                x = self.NO_Blocks[layer_idx].unsqueeze_no_dims(x)
                x_skip_ = self.NO_Blocks[layer_idx].unsqueeze_no_dims(x_skip[layer_idx+1])
                x = self.Skip_Blocks[layer_idx]([x, x_skip_], masks=[mask,masks_skip[layer_idx+1]], emb=emb)[0]
            
            x, mask = self.NO_Blocks[layer_idx].decode(x, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

        return x, mask


    
    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        if self.global_res:
            x_res = x #if isinstance(self.out_layer_res, nn.Identity) else self.out_layer_res(x, emb=emb)

        x, mask, x_skip, mask_skip = self.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        x, mask_out = self.decode(x, mask, x_skip, mask_skip, coords_out=coords_out, indices_sample=indices_sample, emb=emb)

        mask_out = mask_out.view(*x.shape[:-1])

        if self.global_res:
            x = self.out_layer([x, x_res.view(*x.shape[:-1],x_res.shape[-1])], masks=[mask_out, mask.view(*x.shape[:-1])], emb=emb)[0]

        elif not isinstance(self.out_layer, nn.Identity):
            x = self.out_layer(x, emb=emb)

        return x, mask_out


class VarLin_Layer(nn.Module):

    def __init__(self,
                 model_dim_in: list,
                 model_dim_out: int,
                 n_var_amplitudes=1
                ) -> None: 
    
        super().__init__()
        
        self.weights = nn.Parameter(torch.empty((n_var_amplitudes, model_dim_in, model_dim_out)), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weights)


    def get_params(self, emb, weight_or_bias):

        if self.weights.shape[0]>1:
            amps =  weight_or_bias[emb['VariableEmbedder']]
        else:
            amps = weight_or_bias
        amps = amps.view(amps.shape[0],1,*amps.shape[1:])
    
        return amps
    
    def forward(self, x, emb=None):
        
        x = torch.matmul(x, self.get_params(emb, self.weights))
       
        return x 


class SkipAdd_Layer(nn.Module):

    def __init__(self,
                 n_amplitudes_ins: list,
                 n_amplitudes_out: int,
                 n_var_amplitudes=1
                ) -> None: 
    
        super().__init__()
        
        self.weights = nn.ParameterList()
        for n_amplitudes_in in n_amplitudes_ins:
            weights = nn.Parameter(torch.empty(n_var_amplitudes, n_amplitudes_in, n_amplitudes_out), requires_grad=True)
            torch.nn.init.xavier_uniform_(weights)

            self.weights.append(weights)


    def get_weights(self, emb, idx):
        amps = self.weights[idx]

        if amps.shape[0]>1:
            amps =  amps[emb['VariableEmbedder']]
    
        amps = amps.view(amps.shape[0],1,*amps.shape[1:])
    
        return amps
    
    def forward(self, xs, masks=None, emb=None):
        
        x_out = []
        masks_out = []
        for k, x in enumerate(xs):
            w_x = self.get_weights(emb, k)
            w_x = w_x.view(*w_x.shape[:3],*(1,)*(x.dim()-w_x.dim()),*w_x.shape[3:])
            x_out.append(torch.matmul(x, w_x))

            if masks[k] is not None:
                masks_out.append(masks[k].view(*masks[k].shape,*(1,)*(x.dim()-masks[k].dim())))
        
        x = torch.concat(x_out, dim=-2).sum(dim=-2, keepdim=True)

        if len(masks_out)>0:
            norm = (torch.concat(masks_out, dim=-1)==False).sum(dim=-1, keepdim=True)
            x = x/(norm + 1e-10)
            x = x.masked_fill_(norm==0, 0.0)
        else:
            x = x/len(xs)

        return x, norm==0


class Serial_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 layer_settings: list,     
                 n_head_channels = 16,
                 multi_grid_attention: bool=False,
                 p_dropout=0.,
                 global_res=False,
                 skip_mode='amp'
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention
        self.skip = False if len(skip_mode)==0 else True

        self.Skip_Blocks = nn.ModuleList()
        self.NO_Blocks = nn.ModuleList()
        self.global_res=global_res

        encoding_dims = []
        decoding_dims = []
        for n, layer_setting in enumerate(layer_settings):
            encoding_dims.append(layer_setting['amplitude_dim_encode'])
            decoding_dims.append(layer_setting['amplitude_dim_decode'])

   
        for n, layer_setting in enumerate(layer_settings):
            
            model_d_in = model_dim_in if n==0 else decoding_dims[n-1]
            

            no_layer = polNormal_NoLayer(
                layer_setting['grid_layer_in'],
                layer_setting['grid_layer_no'],
                n_amplitudes_in=model_d_in,
                n_amplitudes_out=encoding_dims[n],
                n_amplitdues_inv_in = encoding_dims[n],
                n_amplitudes_inv_out = decoding_dims[n],
                n_phi=layer_setting["n_params"][0],
                n_dist=layer_setting["n_params"][1],
                n_sigma=layer_setting["n_params"][2],
                avg_phi=layer_setting["avg_params"][0],
                avg_dist=layer_setting["avg_params"][1],
                avg_sigma=layer_setting["avg_params"][2],
                dist_learnable=layer_setting["global_params_learnable"][0],
                sigma_learnable=layer_setting["global_params_learnable"][1],
                amplitudes_learnable=layer_setting["global_params_learnable"][2],
                nh_projection=layer_setting["nh_transformation"], 
                nh_backprojection=layer_setting["nh_inverse_transformation"],
                precompute_coordinates=layer_setting["precompute_coordinates"],
                rotate_coord_system=layer_setting["rotate_coordinate_system"],
                pretrained_weights=layer_setting["pretrained_weights"],
                n_var_amplitudes=layer_setting["n_var_amplitudes"],
                with_res=layer_setting["with_res"]
            )
            
            is_decode_encode = [False for _ in range(len(layer_setting['block_types_encode']))]
            is_decode_decode = [True for _ in range(len(layer_setting['block_types_decode']))]
            is_decode = is_decode_encode + is_decode_decode

            embed_names = layer_setting['embed_names_encode'] + layer_setting['embed_names_decode']
            att_block_types = layer_setting['block_types_encode'] + layer_setting['block_types_decode']

            model_dim = int(torch.tensor(no_layer.n_params_no).prod())
            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                model_dim=model_dim,
                att_block_types=att_block_types,
                is_decode=is_decode,
                n_head_channels=n_head_channels,
                att_dim=layer_setting["att_dim"],
                no_layer_nh=None,
                embed_names=embed_names,
                embed_confs=layer_setting['embed_confs'] ,
                embed_mode=layer_setting['embed_mode'] ,
                spatial_attention_config=layer_setting.get("spatial_attention_configs",{}),
                p_dropout=p_dropout
            ))

            model_dim_out_layer = decoding_dims[n] if n < len(layer_settings)-1 and model_dim_out is not None else model_dim_out

            if skip_mode=='amp':
                self.Skip_Blocks.append(SkipAdd_Layer([decoding_dims[n], model_d_in], model_dim_out_layer, n_var_amplitudes=layer_setting["n_var_amplitudes"]))
            elif decoding_dims[n]!=model_dim_out_layer:
                self.Skip_Blocks.append(VarLin_Layer(decoding_dims[n], model_dim_out_layer, layer_settings[-1]['n_var_amplitudes']))
            else:
                self.Skip_Blocks.append(nn.Identity())

        self.model_dim_out = model_dim_out

    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        

        for layer_idx, layer in enumerate(self.NO_Blocks):
            x_res, mask_res = x, mask
            x, mask = layer(x, coords_in=coords_in, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)
            
            mask = mask.view(*x.shape[:-1])
            mask_res = mask_res.view(*x.shape[:-1])
            x_res = x_res.view(*x.shape[:-1],x_res.shape[-1])

            if self.skip:
                x, mask = self.Skip_Blocks[layer_idx]([x, x_res], masks=[mask, mask_res], emb=emb)

            elif not isinstance(self.Skip_Blocks[layer_idx], nn.Identity):
                x = self.Skip_Blocks[layer_idx](x, emb=emb)
                 
        return x, mask

class Parallel_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 layer_settings: list,     
                 n_head_channels = 16,
                 multi_grid_attention: bool=False,
                 p_dropout=0.,
                 global_res=False,
                 skip_mode='amp'
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention
        self.skip = False if len(skip_mode)==0 else True

        self.NO_Blocks = nn.ModuleList()
        self.global_res=global_res

        encoding_dims = []
        decoding_dims = []
        for n, layer_setting in enumerate(layer_settings):
            encoding_dims.append(layer_setting['amplitude_dim_encode'])
            decoding_dims.append(layer_setting['amplitude_dim_decode'])

   
        for n, layer_setting in enumerate(layer_settings):           

            no_layer = polNormal_NoLayer(
                layer_setting['grid_layer_in'],
                layer_setting['grid_layer_no'],
                n_amplitudes_in=model_dim_in,
                n_amplitudes_out=encoding_dims[n],
                n_amplitdues_inv_in = encoding_dims[n],
                n_amplitudes_inv_out = decoding_dims[n],
                n_phi=layer_setting["n_params"][0],
                n_dist=layer_setting["n_params"][1],
                n_sigma=layer_setting["n_params"][2],
                avg_phi=layer_setting["avg_params"][0],
                avg_dist=layer_setting["avg_params"][1],
                avg_sigma=layer_setting["avg_params"][2],
                dist_learnable=layer_setting["global_params_learnable"][0],
                sigma_learnable=layer_setting["global_params_learnable"][1],
                amplitudes_learnable=layer_setting["global_params_learnable"][2],
                nh_projection=layer_setting["nh_transformation"], 
                nh_backprojection=layer_setting["nh_inverse_transformation"],
                precompute_coordinates=layer_setting["precompute_coordinates"],
                rotate_coord_system=layer_setting["rotate_coordinate_system"],
                pretrained_weights=layer_setting["pretrained_weights"],
                n_var_amplitudes=layer_setting["n_var_amplitudes"],
                with_res=layer_setting["with_res"]
            )
            
            is_decode_encode = [False for _ in range(len(layer_setting['block_types_encode']))]
            is_decode_decode = [True for _ in range(len(layer_setting['block_types_decode']))]
            is_decode = is_decode_encode + is_decode_decode

            embed_names = layer_setting['embed_names_encode'] + layer_setting['embed_names_decode']
            att_block_types = layer_setting['block_types_encode'] + layer_setting['block_types_decode']

            model_dim = int(torch.tensor(no_layer.n_params_no).prod())
            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                model_dim=model_dim,
                att_block_types=att_block_types,
                is_decode=is_decode,
                n_head_channels=n_head_channels,
                att_dim=layer_setting["att_dim"],
                no_layer_nh=None,
                embed_names=embed_names,
                embed_confs=layer_setting['embed_confs'] ,
                embed_mode=layer_setting['embed_mode'] ,
                spatial_attention_config=layer_setting.get("spatial_attention_configs",{}),
                p_dropout=p_dropout
            ))

        model_dim_out = decoding_dims[n] if model_dim_out is None else model_dim_out

        if global_res:
            input_dims = decoding_dims + [model_dim_in]
        else:
            input_dims = decoding_dims

        self.Skip_Layer = SkipAdd_Layer(input_dims, model_dim_out, n_var_amplitudes=layer_setting["n_var_amplitudes"])
    
        self.model_dim_out = model_dim_out

    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        #x_res, mask_res = x, mask
        xs = []
        masks = []
        x_in, mask_in = x, mask

        for layer_idx, layer in enumerate(self.NO_Blocks):
            x, mask = layer(x_in, coords_in=coords_in, coords_out=coords_out, indices_sample=indices_sample, mask=mask_in, emb=emb)
            
            mask = mask.view(*x.shape[:-1])
            xs.append(x)
            masks.append(mask)

        if self.global_res:
            mask_in = mask_in.view(*x.shape[:-1])
            x_in = x_in.view(*x.shape[:-1],x_in.shape[-1])
            xs.append(x_in)
            masks.append(mask_in)

        
        x, mask = self.Skip_Layer(xs, masks=masks, emb=emb)

                 
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
                 model_dims, 
                 n_head_channels, 
                 att_dim=None,
                 p_dropout = 0,
                 param_idx_att = None,
                 embedder: BaseEmbedder=None,
                 embedder_mlp: BaseEmbedder=None
                ) -> None: 
      
        super().__init__()

        model_dim_total = torch.tensor(model_dims).prod()
        model_dim_param = model_dim_total/float(model_dims[param_idx_att]) if param_idx_att is not None else model_dim_total
        model_dim_param = int(model_dim_param)

        self.ada_ln = AdaptiveLayerNorm(model_dims, model_dim_total, embedder=embedder)
        self.ada_ln_mlp = AdaptiveLayerNorm(model_dims, model_dim_total, embedder=embedder_mlp)

        att_dim = att_dim if att_dim is not None else model_dim_param

        self.attention_layer = ChannelVariableAttention(model_dim_param, 1, n_head_channels, att_dim=att_dim, with_res=False)

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

        self.param_idx_att = param_idx_att

    def forward(self, x, mask=None, emb=None):
  
        x_shape = x.shape

        res_gamma = self.res_gamma

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

class NH_skip(nn.Module):
  
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