from typing import List, Dict

import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import math

from .neural_operator import NoLayer, polNormal_NoLayer,ReshapeAtt
from ..transformer.attention import ChannelVariableAttention, ResLayer, AdaptiveLayerNorm

from ..icon_grids.grid_attention import GridAttention
from ...modules.embedding.embedder import EmbedderSequential, EmbedderManager, BaseEmbedder


class NOBlock(nn.Module):
  
    def __init__(self,
                 x_dims,
                 no_layer: NoLayer,
                 att_block_types:list,
                 x_dims_decode=None,
                 n_head_channels = 16,
                 att_dim=None,
                 p_dropout=0.,
                 is_decode:list=[],
                 embed_names: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum",
                 spatial_attention_config={}
                ) -> None: 
      
        super().__init__()


       # self.n_params = no_layer.n_params_no
        self.x_dims = x_dims
        x_dims_decode = self.x_dims_decode = x_dims_decode if x_dims_decode is not None else x_dims

        self.no_layer = no_layer
        self.att_block_types_encode = nn.ModuleList()
        self.att_block_types_decode = nn.ModuleList()

        self.prepare_coordinates = False

        for k, att_block_type in enumerate(att_block_types):

            x_dims_layer = x_dims_decode if len(is_decode)>0 and is_decode[k] else x_dims

            if len(embed_names[k])>0 and not 'trans' in att_block_type:
                emb_dict = nn.ModuleDict()
                for embed_name in embed_names[k]:
                    emb: BaseEmbedder = EmbedderManager().get_embedder(embed_name, **embed_confs[embed_name])
                    emb_dict[emb.name] = emb     
                if 'param' in att_block_type or 'var' in att_block_type:
                    embedder_seq = EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = 1)
                embedder_mlp = EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = 1)

                if 'CoordinateEmbedder' in emb_dict.keys():
                    self.prepare_coordinates = True
                    self.grid_layer = no_layer.grid_layers[str(no_layer.global_level_no)]
                    self.global_level = no_layer.global_level_no
            else:
                embedder_seq = None
                embedder_mlp = None

            if 'param' in att_block_type or "var" in att_block_type:
                param_idx_att = [c for c in att_block_type if c.isnumeric()]
                param_idx_att = None if len(param_idx_att)==0 else int(param_idx_att[0])
                cross_var = "var" in att_block_type
                v_proj = "hole" not in att_block_type

                layer = ParamAttention(x_dims_layer, 
                                       n_head_channels,
                                       att_dim=att_dim,
                                       cross_var=cross_var,
                                       p_dropout=p_dropout,
                                       param_idx_att=param_idx_att,
                                       embedder=embedder_seq,
                                       embedder_mlp=embedder_mlp,
                                       v_proj=v_proj)
            elif 'mlp' in att_block_type:
                layer = ParamMLP(x_dims_layer, 
                                p_dropout=p_dropout,
                                embedder=embedder_mlp)
            
            elif 'trans' in att_block_type:
                spatial_attention_config['embedder_names'] = [embed_names[k], []]
                spatial_attention_config['embed_confs'] = embed_confs
                spatial_attention_config['embed_mode'] = embed_mode
                layer = SpatialAttention(x_dims_layer, 
                                         no_layer.grid_layer_no,
                                         n_head_channels, 
                                         spatial_attention_config,
                                         p_dropout=p_dropout,
                                         rotate_coord_system=no_layer.rotate_coord_system)

            if len(is_decode)>0 and is_decode[k]:
                self.att_block_types_decode.append(layer)
            else:
                self.att_block_types_encode.append(layer)

    def squeeze_no_dims(self, x):
        x = x.reshape(*x.shape[:3],-1, x.shape[-1])
        return x

    def unsqueeze_no_dims(self, x):
        x = x.reshape(*x.shape[:3], *self.x_dims[:-2], -1, x.shape[-1])
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
                 p_dropout=0.,
                 global_res=False,
                 skip_mode='amp_sum'
                ) -> None: 
      
        super().__init__()


        self.Skip_Blocks = nn.ModuleList()
        self.NO_Blocks = nn.ModuleList()
        self.global_res=global_res

        encoding_dims = []
        decoding_dims = []
        no_dims = []

        x_dims_no = []
        no_dim_total_prev = 1
        for n, layer_setting in enumerate(layer_settings):
            encoding_dims.append(layer_setting['amplitude_dim_encode'])
            decoding_dims.append(layer_setting['amplitude_dim_decode'])

            no_dim = [layer_setting["n_params"][k] if not layer_setting["avg_params"][k] else 1 
                for k in range(len(layer_setting["n_params"]))]
    
            if n==0:
                x_dims_no.append(no_dim + [1]) 
            else:
                x_dims_no.append(no_dim + [no_dim_total_prev]) 
            no_dim_total_prev *= int(torch.tensor(no_dim).prod())

        encoding_dims_in = []
        encoding_dims_out = []
        decoding_dims_in = []
        decoding_dims_out = []

        for n, layer_setting in enumerate(layer_settings):
            
            encoding_dim_in = model_dim_in if n==0 else encoding_dims[n-1]
            encoding_dim_out = encoding_dims[n]
      
            if n==len(layer_settings) -1:
                decoding_dim_in = encoding_dims[n]
            elif skip_mode=='concat':
                decoding_dim_in = decoding_dims[n+1] + encoding_dims[n]
            else:
                decoding_dim_in = decoding_dims[n+1]
            
            decoding_dim_out = decoding_dims[n]

            encoding_dims_in.append(encoding_dim_in)
            decoding_dims_in.append(decoding_dim_in)
            encoding_dims_out.append(encoding_dim_out)
            decoding_dims_out.append(decoding_dim_out)


            no_layer = get_no_layer(layer_setting, 
                                    encoding_dim_in, 
                                    encoding_dim_out, 
                                    decoding_dim_in, 
                                    decoding_dim_out)
            
            is_decode_encode = [False for _ in range(len(layer_setting['block_types_encode']))]
            is_decode_decode = [True for _ in range(len(layer_setting['block_types_decode']))]
            is_decode = is_decode_encode + is_decode_decode

            embed_names = layer_setting['embed_names_encode'] + layer_setting['embed_names_decode']
            att_block_types = layer_setting['block_types_encode'] + layer_setting['block_types_decode']
                        
            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                x_dims=x_dims_no[n] + [encoding_dims_out[n]],
                x_dims_decode= x_dims_no[n] + [decoding_dims_in[n]],
                att_block_types=att_block_types,
                is_decode=is_decode,
                n_head_channels=n_head_channels,
                att_dim=layer_setting["att_dim"],
                embed_names=embed_names,
                embed_confs=layer_setting['embed_confs'],
                embed_mode=layer_setting['embed_mode'],
                spatial_attention_config=layer_setting.get("spatial_attention_configs",{}),
                p_dropout=p_dropout
            ))

        self.skip = False if len(skip_mode)==0 else True
        for n, layer_setting in enumerate(layer_settings):
            if n != len(layer_settings)-1:
                x_skip_dims = x_dims_no[n] + [encoding_dims_out[n]]
                x_dims = x_dims_no[n] + [decoding_dims_out[n+1]]

                if skip_mode=='amp_sum':
                    self.Skip_Blocks.append(
                        SkipAdd_Layer([x_dims[-1], x_skip_dims[-1]], x_dims[-1], n_var_amplitudes=layer_setting["n_var_amplitudes"])
                        )
                elif skip_mode == 'var_att':
                    emb_dict = nn.ModuleDict()
                    emb: BaseEmbedder = EmbedderManager().get_embedder("VariableEmbedder", **layer_setting['embed_confs']["VariableEmbedder"])
                    emb_dict[emb.name] = emb     
                    embedder_seq = EmbedderSequential(emb_dict, mode=layer_setting['embed_mode'], spatial_dim_count = 1)
                    embedder_mlp = EmbedderSequential(emb_dict, mode=layer_setting['embed_mode'], spatial_dim_count = 1)

                    self.Skip_Blocks.append(
                        MGParamAttention([x_dims, x_skip_dims], 
                                         x_dims,
                                         n_head_channels=n_head_channels,
                                         cross_var=True,
                                         p_dropout=p_dropout,
                                         param_idx_att=None,
                                         embedder=embedder_seq,
                                         embedder_mlp=embedder_mlp
                                         )
                        )

                else:
                    self.Skip_Blocks.append(nn.Identity())
        
        if model_dim_out is None:
            model_dim_out = decoding_dims[0]
        
        if global_res:
            self.out_layer = SkipAdd_Layer([decoding_dims[0], model_dim_in], model_dim_out, n_var_amplitudes=layer_settings[-1]['n_var_amplitudes'])
            self.out_layer_decoder = nn.Identity()
        elif decoding_dims[0]!=model_dim_out:
            self.out_layer = nn.Identity()
            self.out_layer_decoder = VarLin_Layer(decoding_dims[0], model_dim_out, layer_settings[-1]['n_var_amplitudes'])
        else:
            self.out_layer = nn.Identity()
            self.out_layer_decoder = nn.Identity()
            
        self.model_dim_out = model_dim_out

    def encode(self, x, coords_in=None, indices_sample=None, mask=None, emb=None):
        
        x_skip = []
        mask_skip = []
        for layer in self.NO_Blocks:

            x_skip.append(x)
            mask_skip.append(mask)

            x, mask = layer.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        return x, mask, x_skip, mask_skip
    
    def decode(self, x, mask=None, x_skip=None, masks_skip=None, coords_out=None, indices_sample=None, emb=None):
        
        for layer_idx in range(len(self.NO_Blocks)-1,-1,-1):

            if layer_idx<len(self.NO_Blocks)-1 and x_skip is not None and self.skip:
                x = self.NO_Blocks[layer_idx].unsqueeze_no_dims(x)
                x_skip_ = self.NO_Blocks[layer_idx].unsqueeze_no_dims(x_skip[layer_idx+1])

                if not isinstance(self.Skip_Blocks[layer_idx], nn.Identity):
                    x = self.Skip_Blocks[layer_idx]([x, x_skip_], masks=[mask, masks_skip[layer_idx+1]], emb=emb)[0]
                else:
                    x = torch.concat((x, x_skip_), dim=-1)
            
            x, mask = self.NO_Blocks[layer_idx].decode(x, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

        if not isinstance(self.out_layer_decoder, nn.Identity):
            x = self.out_layer_decoder(x, emb=emb)
        return x, mask


    
    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        if self.global_res:
            x_res = x #if isinstance(self.out_layer_res, nn.Identity) else self.out_layer_res(x, emb=emb)

        x, mask_enc, x_skip, mask_skip = self.encode(x, coords_in=coords_in, indices_sample=indices_sample, mask=mask, emb=emb)

        x, mask_out = self.decode(x, mask_enc, x_skip, mask_skip, coords_out=coords_out, indices_sample=indices_sample, emb=emb)

        mask_out = mask_out.view(*x.shape[:-1])

        if self.global_res:
            x = self.out_layer([x, x_res.view(*x.shape[:-1],x_res.shape[-1])], masks=[mask_out, mask.view(*x.shape[:-1])], emb=emb)[0]

        return x, mask_out


class VarLin_Layer(nn.Module):

    def __init__(self,
                 model_dim_in: list,
                 model_dim_out: int,
                 n_var_amplitudes=1
                ) -> None: 
    
        super().__init__()
        
     #   self.weights = nn.Parameter(torch.empty((n_var_amplitudes, model_dim_in, model_dim_out)), requires_grad=True)

        self.lin_layer = nn.Linear(model_dim_in, model_dim_out)
        #torch.nn.init.xavier_uniform_(self.weights)
       # stdv = 1. / math.sqrt(model_dim_in)
       # self.weights.data.uniform_(-stdv, stdv)


    def get_params(self, emb, weight_or_bias):

        if self.weights.shape[0]>1:
            amps =  weight_or_bias[emb['VariableEmbedder']]
        else:
            amps = weight_or_bias
        amps = amps.view(amps.shape[0],1,*amps.shape[1:])
    
        return amps
    
    def forward(self, x, emb=None):
        
       # x = torch.matmul(x, self.get_params(emb, self.weights))

        x = self.lin_layer(x)
       
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
        else:
            amps = amps.unsqueeze(dim=0)
    
        amps = amps.view(amps.shape[0],1,*amps.shape[1:])
    
        return amps
    
    def forward(self, xs, masks=None, emb=None):
        
        x_out = []
        masks_out = []
        for k, x in enumerate(xs):
            x = x.unsqueeze(dim=-2)
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

        return x.squeeze(dim=-2), norm==0


class Serial_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 layer_settings: list,     
                 n_head_channels = 16,
                 multi_grid_attention: bool=False,
                 p_dropout=0.,
                 global_res=False,
                 skip_mode='amp_sum'
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
            
            no_layer = get_no_layer(layer_setting, model_d_in, encoding_dims[n], encoding_dims[n], decoding_dims[n])
            
            is_decode_encode = [False for _ in range(len(layer_setting['block_types_encode']))]
            is_decode_decode = [True for _ in range(len(layer_setting['block_types_decode']))]
            is_decode = is_decode_encode + is_decode_decode

            embed_names = layer_setting['embed_names_encode'] + layer_setting['embed_names_decode']
            att_block_types = layer_setting['block_types_encode'] + layer_setting['block_types_decode']

            #model_dim = int(torch.tensor(no_layer.n_params_no).prod())
            x_dims = copy.deepcopy(no_layer.n_params_no)
            x_dims.insert(-1,1)
            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                x_dims=x_dims,
                att_block_types=att_block_types,
                is_decode=is_decode,
                n_head_channels=n_head_channels,
                att_dim=layer_setting["att_dim"],
                embed_names=embed_names,
                embed_confs=layer_setting['embed_confs'] ,
                embed_mode=layer_setting['embed_mode'] ,
                spatial_attention_config=layer_setting.get("spatial_attention_configs",{}),
                p_dropout=p_dropout
            ))

            model_dim_out_layer = decoding_dims[n] if n < len(layer_settings)-1 and model_dim_out is not None else model_dim_out

            if skip_mode=='amp_sum':
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
                 skip_mode='amp_sum'
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

       # x_dims_list = []
        for n, layer_setting in enumerate(layer_settings):           
            
            no_layer = get_no_layer(layer_setting, 
                                    model_dim_in, 
                                    encoding_dims[n], 
                                    encoding_dims[n], 
                                    decoding_dims[n])
            
            is_decode_encode = [False for _ in range(len(layer_setting['block_types_encode']))]
            is_decode_decode = [True for _ in range(len(layer_setting['block_types_decode']))]
            is_decode = is_decode_encode + is_decode_decode

            embed_names = layer_setting['embed_names_encode'] + layer_setting['embed_names_decode']
            att_block_types = layer_setting['block_types_encode'] + layer_setting['block_types_decode']

            x_dims = copy.deepcopy(no_layer.n_params_no)
            x_dims.insert(-1,1)
          #  x_dims_list.append([x_dims[-1]])
            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                x_dims=x_dims,
                att_block_types=att_block_types,
                is_decode=is_decode,
                n_head_channels=n_head_channels,
                att_dim=layer_setting["att_dim"],
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

        if skip_mode=='amp_sum':
            self.Skip_Layer = SkipAdd_Layer(input_dims, model_dim_out, n_var_amplitudes=layer_setting["n_var_amplitudes"])

        elif skip_mode=='var_att':
            emb_dict = nn.ModuleDict()
            emb: BaseEmbedder = EmbedderManager().get_embedder("VariableEmbedder", **layer_setting['embed_confs']["VariableEmbedder"])
            emb_dict[emb.name] = emb     
            embedder_seq = EmbedderSequential(emb_dict, mode=layer_setting['embed_mode'], spatial_dim_count = 1)
            embedder_mlp = EmbedderSequential(emb_dict, mode=layer_setting['embed_mode'], spatial_dim_count = 1)

            self.Skip_Layer = MGParamAttention([[input_dim] for input_dim in input_dims], 
                                                [1, model_dim_out],
                                                n_head_channels=n_head_channels,
                                                cross_var=True,
                                                p_dropout=p_dropout,
                                                param_idx_att=None,
                                                embedder=embedder_seq,
                                                embedder_mlp=embedder_mlp
                                                )
                
    
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
                 x_dims, 
                 grid_layer,
                 n_head_channels,
                 spatial_attention_configs = None,
                 p_dropout=0.,
                 rotate_coord_system=True
                ) -> None: 
      
        super().__init__()

        self.reshaper = ReshapeAtt(x_dims, None, cross_var=True)

        self.grid_attention_layer = GridAttention(
            grid_layer,
            int(torch.tensor(x_dims).prod()),
            int(torch.tensor(x_dims).prod()),
            n_head_channels=n_head_channels,
            spatial_attention_configs=spatial_attention_configs,
            rotate_coord_system=rotate_coord_system
        )
       
    def forward(self, x, indices_sample=None, mask=None, emb=None):
        nv = x.shape[2]
        x, mask = self.reshaper.shape_to_att(x, mask=mask)

        # insert time dimension
        x = x.unsqueeze(dim=1)
        mask = mask.unsqueeze(dim=1) if mask is not None else mask

        x = self.grid_attention_layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
        x = x.squeeze(dim=1)
        x = self.reshaper.shape_to_x(x, nv_dim=nv)
        return x

class ParamMLP(nn.Module):
  
    def __init__(self,
                 model_dims,
                 p_dropout = 0,
                 embedder: BaseEmbedder=None
                ) -> None: 
      
        super().__init__()

        model_dim_total = int(torch.tensor(model_dims).prod())

        self.reshaper = ReshapeAtt(model_dims, None, cross_var=True)

        self.ada_ln = AdaptiveLayerNorm(model_dims, model_dim_total, embedder=embedder)

        self.gamma = nn.Parameter(torch.ones(model_dim_total)*1e-6, requires_grad=True)

        self.res_layer = ResLayer(model_dim_total, with_res=False, p_dropout=p_dropout)

        self.cross_var = True

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()


    def forward(self, x, mask=None, emb=None):
        nv = x.shape[2]

        x_res, _ = self.reshaper.shape_to_att(x)

        x = self.ada_ln(x, emb=emb)
        x, _ = self.reshaper.shape_to_att(x)

        x = self.res_layer(x)

        x = x_res + self.gamma*x

        x = self.reshaper.shape_to_x(x, nv)
        
        return x, mask


class ParamAttention(nn.Module):
  
    def __init__(self,
                 model_dims, 
                 n_head_channels,
                 cross_var=False, 
                 att_dim=None,
                 p_dropout = 0,
                 param_idx_att = None,
                 embedder: BaseEmbedder=None,
                 embedder_mlp: BaseEmbedder=None,
                 v_proj=True
                ) -> None: 
      
        super().__init__()

        model_dim_total = torch.tensor(model_dims).prod()
        model_dim_param = model_dim_total/float(model_dims[param_idx_att]) if param_idx_att is not None else model_dim_total
        model_dim_param = int(model_dim_param)

        self.reshaper = ReshapeAtt(model_dims, param_idx_att, cross_var=cross_var)

        self.ada_ln = AdaptiveLayerNorm(model_dims, model_dim_total, embedder=embedder)
        self.ada_ln_mlp = AdaptiveLayerNorm(model_dims, model_dim_total, embedder=embedder_mlp)

        att_dim = att_dim if att_dim is not None else model_dim_param

        self.attention_layer = ChannelVariableAttention(model_dim_param, 1, n_head_channels, att_dim=att_dim, with_res=False, v_proj=v_proj)

        self.gamma = nn.Parameter(torch.ones(model_dim_param)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim_param)*1e-6, requires_grad=True)

        self.res_layer = ResLayer(model_dim_param, with_res=False, p_dropout=p_dropout)

        self.param_idx_att = param_idx_att
        self.cross_var = cross_var
        self.v_proj = v_proj

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()


    def forward(self, x, mask=None, emb=None):
        nv = x.shape[2]

        x_res, mask_att = self.reshaper.shape_to_att(x, mask=mask)

        x = self.ada_ln(x, emb=emb)

        x, _ = self.reshaper.shape_to_att(x)
        
        v = x if self.v_proj else x_res

        x, _ = self.attention_layer(x, v=v, mask=mask_att)

        x = self.dropout(x)
        
        residual = x_res + self.gamma*x

       # if mask is not None:
       #     x = torch.where(mask.unsqueeze(-1), x, residual)
       # else:
        x = residual

        x_res = x

        x = self.reshaper.shape_to_x(x, nv)  
        x = self.ada_ln_mlp(x, emb=emb)
        x, _ = self.reshaper.shape_to_att(x)

        x = self.res_layer(x)

        x = x_res + self.gamma_mlp*x

        x = self.reshaper.shape_to_x(x, nv)

        #if mask is not None:
        #    mask = mask_update.view(*x.shape[:3],-1)
        #    mask_update = mask_update.sum(dim=-1).bool()
            

        return x, mask

class MGParamAttention(nn.Module):
  
    def __init__(self,
                 x_dims: list, 
                 x_dim_out,
                 n_head_channels,
                 cross_var=False, 
                 att_dim=None,
                 p_dropout = 0,
                 param_idx_att = None,
                 embedder: BaseEmbedder=None,
                 embedder_mlp: BaseEmbedder=None
                ) -> None: 
      
        super().__init__()

        self.reshaper = []
        self.ada_lns = nn.ModuleList()

        model_dim_total_out = int(torch.tensor(x_dim_out).prod())
        self.x_dim_out = x_dim_out

        model_dim = 0
        res_dim = 0
        for x_dim in x_dims:

            model_dim_total = torch.tensor(x_dim).prod()
            model_dim_param = model_dim_total
            model_dim += int(model_dim_param)
            res_dim += x_dim[-1]

            self.reshaper.append(ReshapeAtt(x_dim, param_idx_att, cross_var=cross_var))

            self.ada_lns.append(AdaptiveLayerNorm(x_dim, model_dim_total, embedder=embedder))

        self.reshaper_out = ReshapeAtt(x_dim_out, param_idx_att, cross_var=cross_var)
        self.ada_ln_mlp = AdaptiveLayerNorm(x_dim_out, model_dim_total_out, embedder=embedder_mlp)

        #self.lin_res_layer = nn.Linear(model_dim, model_dim_total_out, bias=False)
        self.lin_res_layer = VarLin_Layer(res_dim, x_dim_out[-1])

        self.attention_layer = ChannelVariableAttention(model_dim, 1, n_head_channels, att_dim=model_dim, model_dim_out=model_dim_total_out, with_res=False)

        self.gamma = nn.Parameter(torch.ones(model_dim_total_out)*1e-6, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(model_dim_total_out)*1e-6, requires_grad=True)

        self.res_layer = ResLayer(model_dim_total_out, with_res=False, p_dropout=p_dropout)

        self.param_idx_att = param_idx_att
        self.cross_var = cross_var

        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()


    def forward(self, xs, masks=None, emb=None):
        
        x_res_ = []
        x_ = []
        for k, x in enumerate(xs):
            x_res_.append(x)

            x = self.ada_lns[k](x, emb=emb)
            x = self.reshaper[k].shape_to_att(x)[0]
            x_.append(x)
        
        x_res = self.lin_res_layer(torch.concat(x_res_, dim=-1), emb=emb)
        x_res, _ = self.reshaper[k].shape_to_att(x_res)

        x = torch.concat(x_, dim=-1)

        if masks[0] is not None:
            mask = torch.stack(masks, dim=-1).sum(dim=-1).bool()

        x, _ = self.attention_layer(x, mask=mask)

        x = self.dropout(x)
        
        x = x_res + self.gamma*x

        x_res = x

        x = self.reshaper_out.shape_to_x(x, x.shape[-2])  
        x = self.ada_ln_mlp(x, emb=emb)
        x, _ = self.reshaper_out.shape_to_att(x)

        x = self.res_layer(x)

        x = x_res + self.gamma_mlp*x

        x = self.reshaper_out.shape_to_x(x, x.shape[-2])

       # if mask_update is not None:
       #     mask_update = mask_update.view(*x.shape[:3],-1)
       #     mask_update = mask_update.sum(dim=-1).bool()
            

        return x, mask

def get_no_layer(layer_setting, dim_in, dim_out, inv_dim_in, inv_dim_out):
    no_layer = polNormal_NoLayer(
            layer_setting['grid_layer_in'],
            layer_setting['grid_layer_no'],
            n_amplitudes_in=dim_in,
            n_amplitudes_out=dim_out,
            n_amplitdues_inv_in = inv_dim_in,
            n_amplitudes_inv_out = inv_dim_out,
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
            pretrained_weights=layer_setting.get("pretrained_weights",None),
            n_var_amplitudes=layer_setting["n_var_amplitudes"],
            non_linear_encode=layer_setting.get("non_linear_encode", False),
           non_linear_decode=layer_setting.get("non_linear_decode", False),
           cross_no=layer_setting.get("cross_no", False)
        )
    return no_layer