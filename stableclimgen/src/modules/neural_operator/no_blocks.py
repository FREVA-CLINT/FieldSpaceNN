import torch
import torch.nn.functional as F
import torch.nn as nn


from .neural_operator import NoLayer
from ..transformer.attention import ChannelVariableAttention
from ...models.mgno_transformer.mg_attention import MultiGridAttention

class NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 att_block_types:list,
                 n_params=list,
                 n_head_channels = 16,
                 att_dim=None,
                 with_res=False,
                 no_layer_nh: NoLayer=None,
                 is_decode:list=[],
                ) -> None: 
      
        super().__init__()
        
        self.n_params = n_params
        self.no_layer = no_layer
        self.with_res = with_res

        self.att_block_types_encode = nn.ModuleList()
        self.att_block_types_decode = nn.ModuleList()

        for k, att_block_type in enumerate(att_block_types):
            if 'param' in att_block_type:
                param_att_idx = int(att_block_type.replace('param',''))
                layer = ParamAttention(model_dim_in, 
                                       n_head_channels,
                                       att_dim=att_dim, 
                                       n_params=n_params, 
                                       param_idx_att=param_att_idx)
            elif 'var' in att_block_type:
                layer = ParamAttention(model_dim_in, 
                                       n_head_channels,
                                       att_dim=att_dim, 
                                       n_params=n_params, 
                                       param_idx_att=None)

            elif 'nh' in att_block_type:
                layer = NHAttention(no_layer_nh,
                                    model_dim_in, 
                                    n_head_channels,
                                    n_params=n_params)

            if len(is_decode)>0 and is_decode[k]:
                self.att_block_types_decode.append(layer)
            else:
                self.att_block_types_encode.append(layer)

        self.with_res = with_res

        if with_res:
            self.gamma = nn.Parameter(torch.tensor(1e-6), requires_grad=True)
        

    def encode(self, x, indices_layers=None, sample_dict=None, mask=None, coords_in=None):

        x, mask = self.no_layer.transform(x, indices_layers=indices_layers, coordinates=coords_in, sample_dict=sample_dict, mask=mask)

        for layer in self.att_block_types_encode:
            if isinstance(layer, NHAttention):
                x = layer(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask)
            else:    
                x = layer(x, mask=mask)

        return x, mask


    def decode(self, x, indices_layers=None, sample_dict=None, mask=None, coords_in=None, coords_out=None):

        for layer in self.att_block_types_decode:
            if isinstance(layer, NHAttention):
                x = layer(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask)
            else:    
                x = layer(x, mask=mask)

        x, mask = self.no_layer.inverse_transform(x, indices_layers=indices_layers, coordinates=coords_in, sample_dict=sample_dict, mask=mask)

        return x, mask
    

    def forward(self, x, indices_layers=None, sample_dict=None, mask=None, coords_in=None, coords_out=None):
        
        if self.res_mode is not None:
            x_res = x

        x, mask = self.encode(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask, coords_in=coords_in)

        x, mask = self.decode(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask, coords_in=coords_in)

        if self.with_res:
            x = (1-self.gamma)*x_res + (self.gamma)*x

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
                 bottle_neck_dims:list=[],
                 att_dims: list=[],
                 with_res: list=[],
                 no_layers_nh: list[NoLayer]=[None],
                 multi_grid_attention: bool=False
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention
        self.NO_Blocks = nn.ModuleList()

        for n, no_layer in enumerate(no_layers): 

            att_block_types = att_block_types_encode[n] + att_block_types_decode[n]

            self.NO_Blocks.append(NOBlock(
                no_layer=no_layer,
                model_dim_in=model_dim_in,
                model_dim_out=model_dims_out[n],
                att_block_types=att_block_types,
                n_params=n_params[n],
                n_head_channels=n_head_channels,
                att_dim=att_dims[n],
                with_res=with_res[n],
                no_layer_nh=no_layers_nh[n]
            ))

        if multi_grid_attention:
            self.mga_layer = MultiGridAttention(model_dim_in, model_dims_out[n], len(no_layers), att_dim=att_dims[-1], n_head_channels=n_head_channels)
    
    def forward(self, x, indices_layers=None, sample_dict=None, mask=None, coords_in=None, coords_out=None):
        
        x_mg = []
        mask_mg = []
        for layer in self.NO_Blocks:
            x, mask = layer.encode(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask, coords_in=coords_in)
            x, mask = layer.decode(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask, coords_in=coords_in)

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
                 bottle_neck_dims:list=[],
                 att_dims: list=[],
                 with_res: list=[],
                 no_layers_nh: list[NoLayer]=[None],
                 multi_grid_attention: bool=False
                ) -> None: 
      
        super().__init__()

        self.multi_grid_attention = multi_grid_attention
        self.NO_Blocks = nn.ModuleList()

        for n, no_layer in enumerate(no_layers): 
            
            is_decode_encode = [False for _ in range(len(att_block_types_encode[n]))]
            is_decode_decode = [True for _ in range(len(att_block_types_decode[n]))]
            is_decode = is_decode_encode + is_decode_decode

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
                with_res=with_res[n],
                no_layer_nh=no_layers_nh[n]
            ))

        self.model_dim_out_encode = model_dim_out_encode

    def encode(self, x, indices_layers=None, sample_dict=None, mask=None, coords_in=None):

        for layer in self.NO_Blocks:
            x, mask = layer.encode(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask, coords_in=coords_in)
            x, mask = shape_to_att(x, mask=mask)

        return x
    
    def decode(self, x, indices_layers=None, sample_dict=None, mask=None,  coords_out=None):

        for layer_idx in range(len(self.NO_Blocks)-1,-1,-1):
            layer = self.NO_Blocks[layer_idx]
            x_shape_origin = (*x.shape[:3], *layer.n_params, -1)
            x = shape_to_x(x, x_shape_origin)
            x, mask = layer.decode(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask, coords_out=coords_out)

        return x


    
    def forward(self, x, indices_layers=None, sample_dict=None, mask=None, coords_in=None, coords_out=None):
        
        x, shapes = self.encode(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask, coords_in=coords_in)

        x = self.decode(x,shapes, indices_layers=indices_layers, sample_dict=sample_dict, mask=None, coords_out=coords_out)

        return x, mask



class Parallel_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layers: list[NoLayer],
                 att_block_types_encode:list,
                 att_block_types_decode:list,
                 has_bottle_neck:list,
                 n_params=list,
                 n_head_channels = 16,
                 att_dim=None,
                 res_mode=True,
                 no_layers_nh: list[NoLayer]=[None]
                ) -> None: 
      
        super().__init__()

class ParamAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in, 
                 n_head_channels, 
                 att_dim=None,
                 n_params = [],
                 param_idx_att = None
                ) -> None: 
      
        super().__init__()

        model_dim_total = model_dim_in*torch.tensor(n_params).prod()
        model_dim_param = model_dim_total/float(n_params[param_idx_att]) if param_idx_att is not None else model_dim_total

        att_dim = model_dim_param if att_dim is None else att_dim
        self.attention_layer = ChannelVariableAttention(model_dim_param, 1, n_head_channels, att_dim=att_dim)
        self.param_idx_att = param_idx_att

    def forward(self, x, mask=None):
        x_shape = x.shape
        x, mask = shape_to_att(x, self.param_idx_att, mask=mask)
        x, _  = self.attention_layer(x, mask=mask)
        x = shape_to_x(x, x_shape, self.param_idx_att)
        return x


class NHAttention(nn.Module):
  
    def __init__(self,
                 no_layer_nh: NoLayer,
                 model_dim_in, 
                 n_head_channels, 
                 n_params = [], 
                ) -> None: 
      
        super().__init__()

        model_dim_total = model_dim_in*torch.tensor(no_layer_nh.n_params).prod()*torch.tensor(n_params).prod()

        self.no_layer_nh = no_layer_nh

        self.attention_layer = ParamAttention(model_dim_total, n_head_channels, att_dim=model_dim_total, param_idx_att=None)

    def forward(self, x, indices_layers=None, sample_dict=None, mask=None):
        x_shape = x.shape
        x, mask = shape_to_att(x, mask=mask)

        x, mask = self.no_layer_nh.transform(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask)

        x = self.attention_layer(x, mask=mask)
 
        x, mask = self.no_layer_nh.inverse_transform(x, indices_layers=indices_layers, sample_dict=sample_dict, mask=mask)

        x = shape_to_x(x, x_shape)
        return x



def shape_to_att(x, param_att: int=None, mask=None):
    b,n,nv = x.shape[:3]
    param_shape = x.shape[3:]
    n_c = x.shape[-1]

    if param_att == 0:
        x =  x.view(b,n,nv*param_shape[0],-1)
    
    elif param_att==1:
        x = x.transpose(3,4).contiguous()
        x = x.view(b,n,nv*param_shape[1],-1)
    
    elif param_att==2:
        x = x.transpose(3,5)
        x = x.view(b,n,nv*param_shape[2],-1)
    else:
        x = x.view(b,n,nv,-1)

    if mask is not None:
        mask = mask.view(b,n,nv).repeat_interleave(x.shape[2]//nv,dim=2)

    return x, mask
    

def shape_to_x(x, x_shape_orig:tuple, param_att: int=None):

    if param_att == 0 or param_att is None:
        x =  x.view(x_shape_orig)
    
    elif param_att==1:
        x_shape_flat = (*x_shape_orig[:3],x_shape_orig[4], x_shape_orig[3], *x_shape_orig[5:])   
        x = x.view(x_shape_flat)
        x = x.view(x_shape_orig)
        
    elif param_att==2:
        x_shape_flat = (*x_shape_orig[:3], x_shape_orig[5], x_shape_orig[4], x_shape_orig[3], *x_shape_orig[6:])   
        x = x.view(x_shape_flat)
        x = x.view(x_shape_orig)

    return x  