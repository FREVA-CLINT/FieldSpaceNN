
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
import copy
import math

#import tensorly as tl
#from tensorly import einsum

from .neural_operator import NoLayer
from.no_helpers import add_coordinates_to_emb_dict, add_mask_to_emb_dict, update_mask

from ..transformer.attention import ChannelVariableAttention, ResLayer, AdaptiveLayerNorm
from ..transformer.transformer_modules import MultiHeadAttentionBlock

from ..icon_grids.grid_attention import GridAttention
from ...modules.embedding.embedder import EmbedderSequential

#tl.set_backend('pytorch')
einsum_dims = 'pqrstuvw'


def get_residual(x, level_diff, mask=None):
    if level_diff > 0:
        x = x.view(x.shape[0], -1, 4**level_diff, x.shape[-2], x.shape[-1])

        if mask is not None:
            weights = mask.view(x.shape[0], -1, 4**level_diff, x.shape[-2],1)==False
            weights = weights.sum(dim=-3, keepdim=True)
            x = (x/(weights+1e-10)).sum(dim=-3)
            x = x * (weights.sum(dim=-3)!=0)

        else:
            x = x.mean(dim=-3)

    elif level_diff < 0:
        x = x.unsqueeze(dim=2).repeat_interleave(4**(-1*level_diff), dim=2)
        x = x.view(x.shape[0],-1,x.shape[-2],x.shape[-1])
        
    return x


class residual_layer(nn.Module):
  
    def __init__(self,
                 level_diff,
                 model_dim_in,
                 model_dim_out
                ) -> None: 
      
        super().__init__() 

        self.level_diff = level_diff
        self.lin_layer = nn.Linear(model_dim_in, model_dim_out, bias=True) if model_dim_in!= model_dim_out else nn.Identity()

    def forward(self, x, mask=None):

        return self.lin_layer(get_residual(x, self.level_diff, mask=mask))


def get_layer(no_dims_in, input_dim, output_dim, no_dims_out=None, rank=4, no_rank_decay=0, layer_type='Dense'):
    if layer_type == "Dense":
        layer = DenseLayer
        
    elif layer_type == "Tucker":
        layer = TuckerLayer
    
    elif layer_type == "CrossTucker":
        layer = CrossTuckerLayer

    elif layer_type == "CrossDense":
        layer = CrossDenseLayer
    
    elif layer_type == "PathAttention":
        layer = PathAttentionLayer


    return  layer(no_dims_in, 
                input_dim, 
                output_dim, 
                no_dims_out=no_dims_out, 
                rank=rank,
                no_rank_decay=no_rank_decay)

class Stacked_NOBlock(nn.Module):
  
    def __init__(self,
                 Stacked_NOConv_layer,
                 layer_type = 'Tucker',
                 rank = 4,
                 embedder: EmbedderSequential=None,
                 with_gamma = False,
                 p_dropout=0,
                 grid_layers: List = None
                ) -> None: 
      
        super().__init__()

        self.no_conv = Stacked_NOConv_layer
        
        self.input_levels = Stacked_NOConv_layer.input_levels
        self.output_levels = Stacked_NOConv_layer.global_levels_decode
        model_dims_in = Stacked_NOConv_layer.model_dims_in
        model_dims_out = Stacked_NOConv_layer.model_dims_out

        if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
            self.grid_layers = nn.ModuleDict()

        self.lin_skip_inner = nn.ModuleDict()
        self.lin_skip_outer = nn.ModuleDict()
        self.layer_norms1 = nn.ModuleDict()
        self.layer_norms2 = nn.ModuleDict()
        self.gammas1 = nn.ParameterDict()
        self.gammas2 = nn.ParameterDict()
        self.mlp_layers = nn.ModuleDict()

        for output_level, model_dim_out in model_dims_out.items():

            if output_level in self.input_levels:
                model_dim_in = model_dims_in[output_level]
                level_diff = 0
            else:
                level_diff = output_level
                model_dim_in = model_dims_in[0]

            self.lin_skip_inner[str(output_level)] = residual_layer(level_diff, model_dim_in, model_dim_out)
            self.lin_skip_outer[str(output_level)] = residual_layer(level_diff, model_dim_in, model_dim_out)
                
            self.layer_norms1[str(output_level)] = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder)
            self.layer_norms2[str(output_level)] = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder)


            if with_gamma:
                self.gammas1[str(output_level)] = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
                self.gammas2[str(output_level)] = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
            else:
                self.gammas1[str(output_level)] = nn.Parameter(torch.ones(1), requires_grad=False)
                self.gammas2[str(output_level)] = nn.Parameter(torch.ones(1), requires_grad=False)

            if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
                self.grid_layers[str(output_level)] = grid_layers[str(output_level)]

            self.mlp_layers[str(output_level)] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_dim_out, model_dim_out),
                nn.Dropout(p_dropout) if p_dropout>0 else nn.Identity(),
                nn.Linear(model_dim_out, model_dim_out)
            )

        self.activation = nn.SiLU()
              

    def forward(self, x_levels, coords_encode=None, coords_decode=None, indices_sample=None, mask_levels=None, emb=None):
        
        x_levels_input = dict(zip(self.input_levels, x_levels))

        x_levels_out, mask_levels_out = self.no_conv(x_levels, 
                             coords_encode=coords_encode, 
                             coords_decode=coords_decode, 
                             indices_sample=indices_sample, 
                             mask_levels=mask_levels, 
                             emb=emb)

        for k, x in enumerate(x_levels_out):
            output_level = int(self.output_levels[k])

            if output_level in self.input_levels:
                x_res = x_levels_input[output_level]
            else:
                x_res = x_levels_input[0]

            if mask_levels_out[k] is not None:
                emb = add_mask_to_emb_dict(emb, mask_levels_out[k])

            if hasattr(self, 'grid_layers'):
                emb = add_coordinates_to_emb_dict(self.grid_layers[str(output_level)], indices_sample["indices_layers"] if indices_sample else None, emb)

            x = self.gammas1[str(output_level)] * self.layer_norms1[str(output_level)](x, emb=emb) + self.lin_skip_inner[str(output_level)](x_res)

            x = self.mlp_layers[str(output_level)](x)  
        
            x = self.layer_norms2[str(output_level)](x, emb=emb)

            x = self.gammas2[str(output_level)] * x + self.lin_skip_outer[str(output_level)](x_res)

            x = self.activation(x)

            x_levels_out[k] = x
    
        return x_levels_out, mask_levels_out


class Stacked_PreActivationNOBlock(nn.Module):
  
    def __init__(self,
                 Stacked_NOConv_layer,
                 layer_type = 'Tucker',
                 rank = 4,
                 embedder: EmbedderSequential=None,
                 with_gamma = False,
                 p_dropout=0,
                 grid_layers: List = None
                ) -> None: 
      
        super().__init__()

        self.no_conv = Stacked_NOConv_layer
        
        self.input_levels = Stacked_NOConv_layer.input_levels
        self.output_levels = Stacked_NOConv_layer.global_levels_decode
        model_dims_in = Stacked_NOConv_layer.model_dims_in
        model_dims_out = Stacked_NOConv_layer.model_dims_out

        if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
            self.grid_layers = nn.ModuleDict()

        self.lin_skip_inner = nn.ModuleDict()
        self.lin_skip_outer = nn.ModuleDict()
        self.layer_norms1 = nn.ModuleDict()
        self.layer_norms2 = nn.ModuleDict()
        self.gammas1 = nn.ParameterDict()
        self.gammas2 = nn.ParameterDict()
        self.mlp_layers = nn.ModuleDict()

        for input_level, model_dim_in in model_dims_in.items():
            self.layer_norms1[str(input_level)] = AdaptiveLayerNorm([model_dim_in], model_dim_in, embedder=embedder)

        for output_level, model_dim_out in model_dims_out.items():

            if output_level in self.input_levels:
                model_dim_in = model_dims_in[output_level]
                level_diff = 0
            else:
                level_diff = output_level
                model_dim_in = model_dims_in[0]

            self.lin_skip_inner[str(output_level)] = residual_layer(level_diff, model_dim_in, model_dim_out)
            self.lin_skip_outer[str(output_level)] = residual_layer(level_diff, model_dim_in, model_dim_out)
                
            self.layer_norms2[str(output_level)] = AdaptiveLayerNorm([model_dim_out], model_dim_out)


            if with_gamma:
                self.gammas1[str(output_level)] = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
                self.gammas2[str(output_level)] = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
            else:
                self.gammas1[str(output_level)] = nn.Parameter(torch.ones(1), requires_grad=False)
                self.gammas2[str(output_level)] = nn.Parameter(torch.ones(1), requires_grad=False)

            if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
                self.grid_layers[str(output_level)] = grid_layers[str(output_level)]

            self.mlp_layers[str(output_level)] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_dim_out, model_dim_out),
                nn.Dropout(p_dropout) if p_dropout>0 else nn.Identity(),
                nn.Linear(model_dim_out, model_dim_out)
            )

        for input_level in self.input_levels:
            if str(input_level) not in self.grid_layers.keys():
                self.grid_layers[str(input_level)] = grid_layers[str(input_level)]

        self.activation = nn.SiLU()
              

    def forward(self, x_levels, coords_encode=None, coords_decode=None, indices_sample=None, mask_levels=None, emb=None):
        
        x_levels_input = dict(zip(self.input_levels, x_levels))
        mask_levels_input = dict(zip(self.input_levels, mask_levels))

        x_levels = []
        for input_level in self.input_levels:
            input_level = int(input_level)
            
            if hasattr(self, 'grid_layers'):
                emb = add_coordinates_to_emb_dict(self.grid_layers[str(input_level)], indices_sample["indices_layers"] if indices_sample else None, emb)

            if mask_levels_input[input_level] is not None:
                emb = add_mask_to_emb_dict(emb, mask_levels_input[input_level])

            x_levels.append(self.layer_norms1[str(input_level)](x_levels_input[input_level], emb=emb))


        x_levels_out, mask_levels_out = self.no_conv(x_levels, 
                             coords_encode=coords_encode, 
                             coords_decode=coords_decode, 
                             indices_sample=indices_sample, 
                             mask_levels=mask_levels, 
                             emb=emb)

        for k, x in enumerate(x_levels_out):
            output_level = int(self.output_levels[k])

            if output_level in self.input_levels:
                x_res = x_levels_input[output_level]
            else:
                x_res = x_levels_input[0]

            if mask_levels_out[k] is not None:
                emb = add_mask_to_emb_dict(emb, mask_levels_out[k])

            if hasattr(self, 'grid_layers'):
                emb = add_coordinates_to_emb_dict(self.grid_layers[str(output_level)], indices_sample["indices_layers"] if indices_sample else None, emb)

            x = self.gammas1[str(output_level)] * x + self.lin_skip_inner[str(output_level)](x_res)

            x = self.layer_norms2[str(output_level)](x, emb=emb)

            x = self.mlp_layers[str(output_level)](x)  

            x = self.gammas2[str(output_level)] * x + self.lin_skip_outer[str(output_level)](x_res)

            x = self.activation(x)

            x_levels_out[k] = x
    
        return x_levels_out, mask_levels_out

class Stacked_PreActivationAttNOBlock(nn.Module):
  
    def __init__(self,
                 Stacked_NOConv_layer,
                 layer_type = 'Tucker',
                 rank = 4,
                 embedder: EmbedderSequential=None,
                 with_gamma = False,
                 p_dropout=0,
                 grid_layers: List = None
                ) -> None: 
      
        super().__init__()

        self.no_conv = Stacked_NOConv_layer
        
        self.input_levels = Stacked_NOConv_layer.input_levels
        self.output_levels = Stacked_NOConv_layer.global_levels_decode
        model_dims_in = Stacked_NOConv_layer.model_dims_in
        model_dims_out = Stacked_NOConv_layer.model_dims_out

        if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
            self.grid_layers = nn.ModuleDict()

        self.MHA = nn.ModuleDict()
        self.lin_skip = nn.ModuleDict()
        self.lin_proj_in = nn.ModuleDict()
        self.layer_norms1 = nn.ModuleDict()
        self.layer_norms2 = nn.ModuleDict()
        self.layer_norms3 = nn.ModuleDict()
        self.gammas1 = nn.ParameterDict()
        self.gammas2 = nn.ParameterDict()
        self.mlp_layers = nn.ModuleDict()

        self.seq_level = 2
        for input_level, model_dim_in in model_dims_in.items():
            self.layer_norms1[str(input_level)] = AdaptiveLayerNorm([model_dim_in], model_dim_in, embedder=embedder)

        for output_level, model_dim_out in model_dims_out.items():

            if output_level in self.input_levels:
                model_dim_in = model_dims_in[output_level]
                level_diff = 0
            else:
                level_diff = output_level
                model_dim_in = model_dims_in[0]

            n_heads = max([1, model_dim_in // 16])

            self.MHA[str(output_level)] = MultiHeadAttentionBlock(model_dim_out, model_dim_out, n_heads, qkv_proj=False)

            self.lin_proj_in[str(output_level)] = residual_layer(level_diff, model_dim_in, model_dim_out)

            self.lin_skip[str(output_level)] = residual_layer(level_diff, model_dim_in, model_dim_out)
                
            self.layer_norms2[str(output_level)] = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder)

            self.layer_norms3[str(output_level)] = AdaptiveLayerNorm([model_dim_out], model_dim_out)


            if with_gamma:
                self.gammas1[str(output_level)] = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
                self.gammas2[str(output_level)] = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
            else:
                self.gammas1[str(output_level)] = nn.Parameter(torch.ones(1), requires_grad=False)
                self.gammas2[str(output_level)] = nn.Parameter(torch.ones(1), requires_grad=False)

            if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
                self.grid_layers[str(output_level)] = grid_layers[str(output_level)]

            self.mlp_layers[str(output_level)] = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_dim_out, model_dim_out),
                nn.Dropout(p_dropout) if p_dropout>0 else nn.Identity(),
                nn.Linear(model_dim_out, model_dim_out)
            )

        for input_level in self.input_levels:
            if str(input_level) not in self.grid_layers.keys():
                self.grid_layers[str(input_level)] = grid_layers[str(input_level)]

        self.activation = nn.SiLU()
              

    def forward(self, x_levels, coords_encode=None, coords_decode=None, indices_sample=None, mask_levels=None, emb=None):
        
        x_levels_input = dict(zip(self.input_levels, x_levels))
        mask_levels_input = dict(zip(self.input_levels, mask_levels))

        x_levels = []
        for input_level in self.input_levels:
            input_level = int(input_level)
            
            if hasattr(self, 'grid_layers'):
                emb = add_coordinates_to_emb_dict(self.grid_layers[str(input_level)], indices_sample["indices_layers"] if indices_sample else None, emb)

            if mask_levels_input[input_level] is not None:
                emb = add_mask_to_emb_dict(emb, mask_levels_input[input_level])

            x_levels.append(self.layer_norms1[str(input_level)](x_levels_input[input_level], emb=emb))


        x_levels_out, mask_levels_out = self.no_conv(x_levels, 
                             coords_encode=coords_encode, 
                             coords_decode=coords_decode, 
                             indices_sample=indices_sample, 
                             mask_levels=mask_levels, 
                             emb=emb)

        x_levels = dict(zip(self.input_levels, x_levels))
        
        for k, x in enumerate(x_levels_out):
            output_level = int(self.output_levels[k])
            mask = mask_levels_out[k]

            if output_level in self.input_levels:
                x_res = x_levels_input[output_level]
            else:
                x_res = x_levels_input[0]

            if mask_levels_out[k] is not None:
                emb = add_mask_to_emb_dict(emb, mask_levels_out[k])

            if hasattr(self, 'grid_layers'):
                emb = add_coordinates_to_emb_dict(self.grid_layers[str(output_level)], indices_sample["indices_layers"] if indices_sample else None, emb)


            if output_level in self.input_levels:
                x_in = x_levels[output_level]
            else:
                x_in = x_levels[0]

            x_in = self.lin_proj_in[str(output_level)](x_in)

            x = self.layer_norms2[str(output_level)](x, emb=emb)

            b,n,v,c = x.shape

            if n == x_in.shape[1]:
                x_in = x_in.view(b,-1,4**self.seq_level,c)
                x = x.view(b,-1,4**self.seq_level,c)

                x_in = x_in.view(-1,4**self.seq_level,c)
                x = x.view(-1,4**self.seq_level,c)
            else:
                x_in = x_in.view(n*b*v,-1,c)
                x = x.view(n*b*v,-1,c)

            mask = mask.view(x.shape[:-1]) if mask is not None else None

            x_mha = self.MHA[str(output_level)](q=x_in, k=x, v=x, mask=mask)

            x_mha = x_mha.view(b,n,v,c)
            x = x.view(b,n,v,c)

            x =  self.lin_skip[str(output_level)](x_res) + self.gammas1[str(output_level)] * x_mha 

            x_res = x 

            x = self.layer_norms3[str(output_level)](x, emb=emb)

            x = self.gammas2[str(output_level)] * self.mlp_layers[str(output_level)](x) + x_res

            x = self.activation(x)
    
            x_levels_out[k] = x
        return x_levels_out, mask_levels_out
    

class NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 embedder: EmbedderSequential=None,
                 with_gamma = False,
                 mask_as_embedding=False,
                 OW_zero = False,
                 layer_type='Dense',
                 rank = 4
                ) -> None: 
      
        super().__init__()

        self.mask_as_embedding = mask_as_embedding

        global_level_decode = int(no_layer.global_level_decode)

        self.level_diff = (global_level_decode - no_layer.global_level_encode)

        level_diff_no_out = (no_layer.global_level_no - global_level_decode)

        if level_diff_no_out == 0 and OW_zero:
            self.no_conv = O_NOConv(model_dim_in, 
                                    model_dim_out, 
                                    no_layer, 
                                    layer_type=layer_type, 
                                    rank=rank)
        else:
            self.no_conv = NOConv(model_dim_in, 
                                  model_dim_out, 
                                  no_layer, 
                                  layer_type=layer_type, 
                                  rank=rank)
        
        self.lin_skip_outer = nn.Linear(model_dim_in, model_dim_out, bias=True)
        self.lin_skip_inner = nn.Linear(model_dim_in, model_dim_out, bias=True)

        self.layer_norm1 = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder)
        self.layer_norm2 = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder)

        if with_gamma:
            self.gamma1 = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
        else:
            self.register_buffer('gamma1', torch.ones(model_dim_out))
            self.register_buffer('gamma2', torch.ones(model_dim_out))

        self.mlp_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim_out, model_dim_out),
            nn.Dropout(p_dropout) if p_dropout>0 else nn.Identity(),
            nn.Linear(model_dim_out, model_dim_out)
        )

        if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
            self.grid_layer_1 = no_layer.rcm.grid_layers[str(no_layer.global_level_decode)]

        self.activation = nn.SiLU()
              

    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        
        x_res = x
        
        x_res = get_residual(x, self.level_diff, mask=mask if not self.mask_as_embedding else None)

        x_conv, mask = self.no_conv(x, 
                                coords_encode=coords_encode, 
                                coords_decode=coords_decode, 
                                indices_sample=indices_sample, 
                                mask=mask, 
                                emb=emb)

        if mask is not None:
            emb = add_mask_to_emb_dict(emb, mask)

        if hasattr(self, 'grid_layer_1'):
            emb = add_coordinates_to_emb_dict(self.grid_layer_1, indices_sample["indices_layers"] if indices_sample else None, emb)

        x = self.gamma1 * self.layer_norm1(x_conv, emb=emb) + self.lin_skip_inner(x_res)

        x = self.mlp_layer(x) 
        
        x = self.layer_norm2(x, emb=emb)

        x = self.gamma2 * x + self.lin_skip_outer(x_res)

        x = self.activation(x)
    
        return x, mask

class PreActivation_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 embedder: EmbedderSequential=None,
                 p_dropout=0.,
                 with_gamma=False,
                 mask_as_embedding=False,
                 OW_zero = False,
                 layer_type = "Dense",
                 rank = 4
                ) -> None: 
      
        super().__init__()

        self.mask_as_embedding = mask_as_embedding
        global_level_decode = int(no_layer.global_level_decode)

        level_diff_no_out = (no_layer.global_level_no - global_level_decode)

        if level_diff_no_out == 0 and OW_zero:
            self.no_conv = O_NOConv(model_dim_in, 
                                    model_dim_out, 
                                    no_layer, 
                                    layer_type=layer_type, 
                                    rank=rank)
        else:
            self.no_conv = NOConv(model_dim_in, 
                                  model_dim_out, 
                                  no_layer, 
                                  layer_type=layer_type, 
                                  rank=rank)
        
        self.level_diff = (global_level_decode - no_layer.global_level_encode)

        self.lin_skip_outer = nn.Linear(model_dim_in, model_dim_out, bias=True)
        self.lin_skip_inner = nn.Linear(model_dim_in, model_dim_out, bias=True)

        self.layer_norm1 = AdaptiveLayerNorm([model_dim_in], model_dim_in, embedder=embedder)
        self.layer_norm2 = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder=embedder)

        if with_gamma:
            self.gamma1 = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
        else:
            self.register_buffer('gamma1', torch.ones(model_dim_out))
            self.register_buffer('gamma2', torch.ones(model_dim_out))

        self.mlp_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim_out, model_dim_out),
            nn.Dropout(p_dropout) if p_dropout>0 else nn.Identity(),
            nn.Linear(model_dim_out, model_dim_out),
        )

        self.activation = nn.SiLU()
        
        if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
            self.grid_layer_1 = no_layer.rcm.grid_layers[str(no_layer.global_level_encode)]
            self.grid_layer_2 = no_layer.rcm.grid_layers[str(no_layer.global_level_decode)]

    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        
        x_res = x
        
        x_res = get_residual(x, self.level_diff, mask=mask if not self.mask_as_embedding else None)
        
        if hasattr(self, 'grid_layer_1'):
            emb = add_coordinates_to_emb_dict(self.grid_layer_1, indices_sample["indices_layers"] if indices_sample else None, emb)

        if mask is not None:
            emb = add_mask_to_emb_dict(emb, mask)

        x = self.layer_norm1(x, emb=emb)

        x_conv, mask = self.no_conv(x, 
                                 coords_encode=coords_encode, 
                                 coords_decode=coords_decode, 
                                 indices_sample=indices_sample, 
                                 mask=mask, 
                                 emb=emb)

        x = self.gamma1 * x_conv + self.lin_skip_inner(x_res)
        
        if hasattr(self, 'grid_layer_2'):
            emb = add_coordinates_to_emb_dict(self.grid_layer_2, indices_sample["indices_layers"] if indices_sample else None, emb)

        if mask is not None:
            emb = add_mask_to_emb_dict(emb, mask)

        x = self.layer_norm2(x, emb=emb)

        x = self.gamma2 * self.mlp_layer(x) + self.lin_skip_outer(x_res)

        x = self.activation(x)
    
        return x, mask


class CrossTuckerLayer(nn.Module):
    def __init__(self, 
                 no_dims_in, 
                 input_dim, 
                 output_dim, 
                 no_dims_out=None, 
                 rank=2, 
                 const_str="bnv",
                 n_vars_total=1,
                 no_rank_decay=0, 
                 norm=True):
        """
        Args:
            in_shape (tuple of int): Non-batch input shape, e.g. (n1, n1, n1, c).
            out_shape (tuple of int): Desired output shape, e.g. (n2, n2, n2, c).
                The length of out_shape must equal len(in_shape).
            rank (int): Tucker rank (all modes use the same rank for simplicity).
            const_str (str): Label for the batch dimension (default "b").
        """
        super().__init__()

        in_shape = (*no_dims_in, input_dim)

        in_shape = [in_sh for in_sh in in_shape if in_sh>1]
        
        reduction = False

        if no_dims_out is None:
            no_dims_out = no_dims_in

        elif int(torch.tensor(no_dims_out).prod()) == 1:
            reduction=True
            no_dims_out = (1,)*len(no_dims_in)

        out_shape = (*no_dims_out, output_dim)

        out_shape = [out_sh for out_sh in out_shape if out_sh>1]

        d_in = len(in_shape)  
        d_out = len(out_shape)
        self.in_shape = in_shape

        

        ranks_in = get_ranks(in_shape, rank, no_rank_decay=no_rank_decay)
        ranks_out = get_ranks(out_shape, rank, no_rank_decay=no_rank_decay)

        if norm:
            norm_val = torch.tensor((in_shape)).prod()
        else:
            norm_val = 1.


       # self.core = nn.Parameter(torch.randn(*(*ranks_out,*ranks_in))/norm_val)

        fan_in = torch.tensor(ranks_in).prod()
        fan_out = torch.tensor(ranks_out).prod()
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        self.core = nn.Parameter(torch.randn(*ranks_out, *ranks_in) * scale)

        self.out_factors = nn.ParameterList()
        for i, rank in enumerate(ranks_out):
            param = torch.empty(out_shape[i], rank)
            nn.init.xavier_uniform_(param)
            self.out_factors.append(nn.Parameter(param))
        
        self.in_factors = nn.ParameterList()
        for i, rank in enumerate(ranks_in):
            param = torch.empty(in_shape[i], rank)
            nn.init.xavier_uniform_(param)
            self.in_factors.append(nn.Parameter(param))

        in_letters = "adefghijklmß$§%"   
        out_letters = "opqrstuwxyz§%"        
        core_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ12345678" 

        if d_in > len(in_letters) or d_out > len(out_letters) or d_in+d_out > len(core_letters):
            raise ValueError("Not enough letters to label all dimensions. Increase the letter set strings.")

      
        self.x_subscript = const_str + in_letters[:d_in]
 
        core_subscript = core_letters[:d_out] + core_letters[d_out:d_out+d_in]
  
        out_factor_subscripts = [out_letters[i] + core_letters[i] for i in range(d_out)]

        in_factor_subscripts = [in_letters[i] + core_letters[d_out+ i] for i in range(d_in)]

        output_subscript = const_str + out_letters[:d_out]

        if reduction:
            output_subscript = const_str + out_letters[d_out-1]
        else:
            output_subscript = const_str + out_letters[:d_out]

        self.einsum_eq = (
            f"{self.x_subscript},"
            f"{core_subscript},"
            f"{','.join(out_factor_subscripts)},{','.join(in_factor_subscripts)}"
            f"->{output_subscript}"
        )

    def forward(self, x):

        x = x.view(*x.shape[:3],*self.in_shape)

        x = torch.einsum(self.einsum_eq,
            x,
            self.core,
            *self.out_factors,
            *self.in_factors)
        x = x.reshape(*x.shape[:3],-1,x.shape[-1])
        return x


def get_ranks(shape, rank, no_rank_decay=0):
    rank_ = []
    for k in range(len(shape)):
        r = rank * (1 - no_rank_decay*k/(max([1,len(shape)-1]))) 
        if k < len(shape)-1:
            rank_.append(r)
        else:
            if len(rank_)>0:
                rank_.append(float(torch.tensor(rank_).mean()))
            else:
                rank_.append(float(rank))

    if rank > 1:
        ranks = [min([dim, rank_[k]]) for k, dim in enumerate(shape)]
    else:
        ranks = [max([1,int(dim * rank_[k])]) for k, dim in enumerate(shape)]
    
    return ranks

class TuckerLayer(nn.Module):
    def __init__(self, 
                 no_dims, 
                 input_dim, 
                 output_dim, 
                 no_dims_out=None, 
                 rank=2, 
                 const_str="bnv", 
                 norm=True, 
                 no_rank_decay=0,
                 n_vars_total=1,
                 **kwargs):

        super().__init__()
        # Number of dimensions to transform.
        in_shape = (*no_dims, input_dim)

        self.in_shape = in_shape
        weight_shape = (*in_shape, output_dim)

        d = len(weight_shape)

        ranks_in = get_ranks(in_shape, rank, no_rank_decay=no_rank_decay)
        ranks_out = get_ranks((output_dim,), rank, no_rank_decay=no_rank_decay)

        ranks = core_shape = (*ranks_in,*ranks_out)

        reduction = False

        if no_dims_out is not None:
            no_dims_tot_out_out = int(torch.tensor(no_dims_out).prod())
            reduction = no_dims_tot_out_out == 1
  

        fan_in = torch.tensor(ranks_in).prod()
        fan_out = torch.tensor(ranks_out).prod()
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        self.core = nn.Parameter(torch.randn(*ranks_in, *ranks_out) * scale)


        self.factors = nn.ParameterList()
        for i, rank in enumerate(ranks):
            param = torch.empty(weight_shape[i], rank)
            nn.init.xavier_uniform_(param)
            self.factors.append(nn.Parameter(param))


        factor_letters = "adefghijklmopqrstuwxyz"   # For input dimensions.

        core_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # For the core tensor; must have at least 2*d letters.

        x_subscript = const_str + factor_letters[:(d-1)]

        core_subscript = core_letters[:d]

        factor_subscripts = [factor_letters[i] + core_letters[i] for i in range(d)]


        if reduction:
            output_subscript = const_str + factor_letters[d-1]
        else:
            output_subscript = const_str + factor_letters[:(d-2)] + factor_letters[d-1]

        self.einsum_eq = (
            f"{x_subscript},"
            f"{core_subscript},"
            f"{','.join(factor_subscripts)}"
            f"->{output_subscript}"
        )

    def forward(self, x):

        x = x.view(*x.shape[:3],*self.in_shape)
        x = torch.einsum(
            self.einsum_eq,
            x,
            self.core,
            *self.factors,
        )
        x = x.reshape(*x.shape[:3],-1,x.shape[-1])
        return x

class LinConcatLayer(nn.Module):
  
    def __init__(self,
                 no_dims: list,
                 input_dim: int,
                 concat_model_dim: int,
                 **kwargs
                ) -> None: 
      
        super().__init__()
        
        target_dim = torch.tensor(no_dims).prod()
        self.no_dims = no_dims

        self.lin_proj = nn.Linear(input_dim, target_dim*concat_model_dim, bias=False)

    def forward(self, x, x_c):
       
        x_c = self.lin_proj(x_c)

        x_c = x_c.view(*x.shape[:3],*self.no_dims,-1)
        x = x.view(*x.shape[:3],*self.no_dims,-1)
        x = torch.concat((x, x_c), dim=-1)

        return x
    

class CrossTuckerConcatLayer(nn.Module):
  
    def __init__(self,
                 no_dims: list,
                 input_dim: int,
                 output_dim: int,
                 rank=2,
                 no_rank_decay=0,
                ) -> None: 
      
        super().__init__()
        
        self.no_dims = no_dims
        
        if len(no_dims)==0:
            no_dims_ = (1,)
        else:
            no_dims_ = no_dims

        self.layer = CrossTuckerLayer((1,)*len(no_dims_), input_dim, output_dim, no_dims_out=no_dims_, rank=rank, no_rank_decay=no_rank_decay)

    def forward(self, x, x_c):
       
        x_c = self.layer(x_c)

        x_c = x_c.view(*x.shape[:3],*self.no_dims,-1)
        x = x.view(*x.shape[:3],*self.no_dims,-1)
        x = torch.concat((x, x_c), dim=-1)

        return x


class TuckerConcatLayer(nn.Module):
  
    def __init__(self,
                 no_dims: list,
                 input_dim: int,
                 output_dim: int,
                 rank=3,
                 no_rank_decay=0
                ) -> None: 
      
        super().__init__()
        
        self.no_dims=no_dims
        no_dims_tot = int(torch.tensor(no_dims).prod())

        self.layer = TuckerLayer((1,), input_dim, no_dims_tot*output_dim, rank=rank, no_rank_decay=no_rank_decay)

    def forward(self, x, x_c):
       
        x_c = self.layer(x_c)

        x_c = x_c.view(*x.shape[:3],*self.no_dims,-1)
        x = x.view(*x.shape[:3],*self.no_dims,-1)
        x = torch.concat((x, x_c), dim=-1)

        return x


class PathAttentionLayer(nn.Module):
    def __init__(self,
                 no_dims,
                 input_dim,
                 output_dim,
                 n_vars_total=1,
                 rank = 0.5,
                 no_rank_decay = 0,
                 **kwargs
                ) -> None: 
         
        super().__init__()
        
        no_dims_out = get_ranks((*no_dims, input_dim), rank, no_rank_decay=no_rank_decay)[:-1]

        self.no_dims = no_dims
        self.no_dims_out = no_dims_out
        self.no_dims_tot = int(torch.tensor(no_dims).prod())
        self.no_dims_out_tot = int(torch.tensor(no_dims_out).prod())

        self.path_layer = CrossTuckerLayer(no_dims, input_dim, input_dim, no_dims_out=no_dims_out, rank=rank, no_rank_decay=no_rank_decay)

        #self.skip_layer = TuckerLayer(no_dims, input_dim, output_dim, no_dims_out=no_dims, rank=rank, no_rank_decay=no_rank_decay)     
        self.layer_norm1 = nn.LayerNorm([input_dim]) 
        self.layer_norm2 = nn.LayerNorm([input_dim]) 

        n_heads = max([1, input_dim // 16])
        self.MHA = MultiHeadAttentionBlock(
            output_dim, output_dim, n_heads, input_dim=input_dim, qkv_proj=True
            )   
        
        self.grid_embedding1 = nn.Parameter(torch.randn(self.no_dims_tot, input_dim), requires_grad=True)
        self.grid_embedding2 = nn.Parameter(torch.randn(self.no_dims_out_tot, input_dim), requires_grad=True)

    def forward(self, x, mask=None):
        
        b,n,v = x.shape[:3]

        x_cross = self.path_layer(x)

       # x_res = self.skip_layer(x)

        x_cross = x_cross.view(b*n*v, self.no_dims_out_tot, -1)
        x = x.view(b*n*v, self.no_dims_tot, -1)

        x = self.layer_norm1(x) + self.grid_embedding1
        x_crossk = self.layer_norm2(x_cross) + self.grid_embedding2

        x = self.MHA(q=x, k=x_crossk, v=x_cross)

        x = x.view(b,n,v,self.no_dims_tot,-1)

      #  x = x + x_res

        return x

class DenseLayer(nn.Module):
    def __init__(self,
                 no_dims,
                 input_dim,
                 output_dim,
                 no_dims_out=None,
                 n_vars_total=1,
                 **kwargs
                ) -> None: 
         
        super().__init__()

        self.no_dims = no_dims
        self.n_dim_tot = int(torch.tensor(no_dims).prod())

        if no_dims_out is not None and int(torch.tensor(no_dims_out).prod()) == 1:
            self.einsum_eq = "nbvmi,mij->nbvj"
        else:
            no_dims_out = 1
            self.einsum_eq = "nbvmi,mij->nbvmj"

        no_dims_out = int(torch.tensor(no_dims_out).prod())


        self.weights = torch.empty(self.n_dim_tot, input_dim, output_dim)
        for i in range(self.n_dim_tot):
            nn.init.kaiming_uniform_(self.weights[i], a=0, nonlinearity='relu')


    def forward(self, x):

        x = x.view(*x.shape[:3],self.n_dim_tot,-1)

        x = torch.einsum(self.einsum_eq, x, self.weights)

        return x
    
class CrossDenseLayer(nn.Module):
    def __init__(self,
                 no_dims,
                 input_dim,
                 output_dim,
                 no_dims_out,
                 n_vars_total=1,
                 **kwargs
                ) -> None: 
         
        super().__init__()

        self.n_dim_tot = int(torch.tensor(no_dims).prod())
        self.n_dim_tot_out = int(torch.tensor(no_dims_out).prod())

        if no_dims_out is not None and self.n_dim_tot_out == 1:
            self.einsum_eq = "nbvmi,oipj->nbvj"
        else:
            self.einsum_eq = "nbvmi,oipj->nbvpj"
        
        #self.linear = nn.Linear(self.n_dim_tot*input_dim, self.n_dim_tot_out*output_dim, bias=False)

        self.weights = torch.empty(self.n_dim_tot, input_dim, self.n_dim_tot_out, output_dim)
        for i in range(self.n_dim_tot):
            for j in range(self.n_dim_tot_out):
                nn.init.kaiming_uniform_(self.weights[i, :, j, :], a=0, nonlinearity='relu')

    def forward(self, x):

        x = x.view(*x.shape[:3],self.n_dim_tot,-1)
        x = torch.einsum(self.einsum_eq, x, self.weights)
        
        return x

class Stacked_NOConv(nn.Module):
  
    def __init__(self,
                 input_levels: List,
                 model_dims_in,
                 model_dims_out,
                 no_layers: List[NoLayer],
                 global_levels_decode: list,
                 layer_type = 'CrossTucker',
                 concat_layer_type = 'CrossTucker',
                 concat_model_dim = 1,
                 output_reduction_layer_type = 'Dense',
                 rank = 4,
                 rank_cross=2,
                 no_rank_decay=0
                ) -> None: 
      
        super().__init__()

        self.register_buffer("global_levels_decode", torch.tensor(global_levels_decode))
       
        self.input_levels = input_levels

        self.no_layers = no_layers
        max_level = no_layers[-1].global_level_no
        level_step = no_layers[1].global_level_no - no_layers[0].global_level_no
        
        self.no_dims = {0:[]}

        for k, no_layer in enumerate(no_layers):
            no_dims = copy.deepcopy(no_layer.n_params_no)
            if k>0:
                no_dims += self.no_dims[no_layer.global_level_no-level_step]

            self.no_dims[no_layer.global_level_no] = no_dims

        self.no_dim = copy.deepcopy(no_layer.n_params_no)

        model_dim_no_max_in = model_dims_in[0]
        total_concat_model_dim = 0
        model_dims_in = dict(zip(input_levels, model_dims_in))
        self.model_dims_in = model_dims_in

        self.level_concat_layers = nn.ModuleDict()
        for global_level_input in input_levels[1:]:
            
            no_dims = self.no_dims[global_level_input]

            if concat_layer_type == 'Dense':
                layer = LinConcatLayer
            
            elif concat_layer_type == 'Tucker':
                layer = TuckerConcatLayer
            
            elif concat_layer_type == 'CrossTucker':
                layer = CrossTuckerConcatLayer
                rank = rank_cross

            self.level_concat_layers[str(int(global_level_input))] = layer(no_dims,
                    model_dims_in[global_level_input],
                    concat_model_dim,
                    rank=rank,
                    no_rank_decay=no_rank_decay)
            
            total_concat_model_dim += concat_model_dim

        model_dim_no_max_in += total_concat_model_dim
        model_dim_max = model_dims_out[-1]
        model_dims_out = dict(zip(global_levels_decode, model_dims_out))
        self.model_dims_out = model_dims_out

        self.level_reduction_layers = nn.ModuleDict()
        self.skip_gammas = nn.ParameterDict()

        for global_level_decode in global_levels_decode:

            global_level_decode = int(global_level_decode)

            no_dims = self.no_dims[global_level_decode]

            layer = get_layer(no_dims, 
                            model_dim_max, 
                            model_dims_out[global_level_decode], 
                            no_dims_out=(1,), 
                            rank=rank_cross if output_reduction_layer_type=='CrossTucker' else rank,
                            no_rank_decay=no_rank_decay,
                            layer_type=output_reduction_layer_type)

            self.level_reduction_layers[str(global_level_decode)] = layer

               
        self.layer = get_layer(self.no_dims[max_level],
                               model_dim_no_max_in,
                               model_dim_max,
                               no_dims_out=self.no_dims[max_level],
                               rank=rank_cross if layer_type == 'CrossTucker' else rank,
                               no_rank_decay=no_rank_decay,
                               layer_type=layer_type)



    def forward(self, x_levels, coords_encode=None, coords_decode=None, indices_sample=None, mask_levels=None, emb=None):
        
        x = x_levels[0]
        mask = mask_levels[0] if mask_levels is not None else None

        x_levels = dict(zip(self.input_levels, x_levels))

        for k, no_layer in enumerate(self.no_layers):

            x, mask = no_layer.transform(x, coords_encode=coords_encode, indices_sample=indices_sample, mask=mask, emb=emb)

            global_level_no = int(no_layer.global_level_no)
            if global_level_no in self.input_levels:
                x = self.level_concat_layers[str(global_level_no)](x, x_levels[global_level_no])

            x = x.view(*x.shape[:3],-1)

        x = self.layer(x)

        x_levels_out = []
        mask_levels_out = []
        
        if self.no_layers[-1].global_level_no in self.global_levels_decode:
            x_out = self.level_reduction_layers[str(self.no_layers[-1].global_level_no)](x)
            x_levels_out.append(x_out.view(*x_out.shape[:3],-1))
            mask_levels_out.append(mask)

        for k, no_layer in enumerate(self.no_layers[::-1]):

            x = x.reshape(*x.shape[:3], *self.no_dim, -1)
            x, mask = no_layer.inverse_transform(x, coords_decode=coords_decode, indices_sample=indices_sample, mask=mask, emb=emb)

            if no_layer.global_level_decode in self.global_levels_decode:
                x_out = self.level_reduction_layers[str(no_layer.global_level_decode)](x)
          
                x_levels_out.insert(0,x_out.view(*x_out.shape[:3],-1))
                mask_levels_out.insert(0, mask)

        return x_levels_out, mask_levels_out

class NOConv(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 layer_type="Dense",
                 rank=4
                ) -> None: 
      
        super().__init__()

        self.no_layer = no_layer

        self.no_dims = self.no_layer.n_params_no 
        self.global_level_in = self.no_layer.global_level_encode
        self.global_levels_out = self.no_layer.global_level_decode

        self.layer = get_layer(self.no_dims, model_dim_in, model_dim_out, no_dims_out=self.no_dims, layer_type=layer_type, rank=rank)

    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        x, mask = self.no_layer.transform(x, coords_encode=coords_encode, indices_sample=indices_sample, mask=mask, emb=emb)

        x = self.layer(x)
        
        x = x.view(*x.shape[:3],*self.no_dims,-1)

        x, mask = self.no_layer.inverse_transform(x, coords_decode=coords_decode, indices_sample=indices_sample, mask=mask, emb=emb)

        x = x.contiguous()

        return x, mask

    
class O_NOConv(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 layer_type="Dense",
                 rank=4
                ) -> None: 
      
        super().__init__()

        self.no_layer = no_layer

        self.no_dims = self.no_layer.n_params_no 
        self.global_level_in = self.no_layer.global_level_encode
        self.global_level_out = int(self.no_layer.global_level_decode)

        self.layer = get_layer(self.no_dims, model_dim_in, model_dim_out, no_dims_out=(1,), rank=rank, layer_type=layer_type)


    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        x, mask = self.no_layer.transform(x, coords_encode=coords_encode, indices_sample=indices_sample, mask=mask, emb=emb)

        x = self.layer(x)
    
        x = x.view(*x.shape[:3],-1)

        return x, mask

class SpatialAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out, 
                 grid_layer,
                 n_head_channels,
                 p_dropout=0.,
                 rotate_coord_system=True,
                 spatial_attention_configs = None,
                 mask_as_embedding=False
                ) -> None: 
      
        super().__init__()

        self.grid_attention_layer = GridAttention(
            grid_layer,
            model_dim_in,
            model_dim_out,
            n_head_channels=n_head_channels,
            spatial_attention_configs=spatial_attention_configs,
            rotate_coord_system=rotate_coord_system
        )

        self.mask_as_embedding = mask_as_embedding
       
    def forward(self, x, indices_sample=None, mask=None, emb=None):
        nv = x.shape[2]

        if mask is not None:
            emb = add_mask_to_emb_dict(emb, mask)

        # insert time dimension
        x = x.unsqueeze(dim=1)
        mask = mask.unsqueeze(dim=1) if mask is not None else mask
        
      #      emb['MaskEmbedder'] = emb['MaskEmbedder'].unsqueeze(dim=1)

        x = self.grid_attention_layer(x, 
                                      indices_sample=indices_sample, 
                                      mask=mask if not self.mask_as_embedding else None, 
                                      emb=emb)
        x = x.squeeze(dim=1)
        return x



class VariableAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in, 
                 model_dim_out,
                 grid_layer,
                 n_head_channels,
                 att_dim=None,
                 p_dropout = 0,
                 embedder: EmbedderSequential=None,
                 embedder_mlp: EmbedderSequential=None,
                 mask_as_embedding=False
                ) -> None: 
      
        super().__init__()

        self.mask_as_embedding = mask_as_embedding

        if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
            self.grid_layer = grid_layer

        if model_dim_in != model_dim_out:
            self.lin_skip_outer = nn.Linear(model_dim_in, model_dim_out)
        else:
            self.lin_skip_outer = nn.Identity()

        self.attention_ada_ln = AdaptiveLayerNorm([model_dim_in], model_dim_in, embedder=embedder)
        self.attention_ada_ln_mlp = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder=embedder_mlp)

        att_dim = att_dim if att_dim is not None else model_dim_in
        self.attention_layer = ChannelVariableAttention(model_dim_in, 
                                                        1, 
                                                        n_head_channels, 
                                                        att_dim=att_dim, 
                                                        with_res=False,
                                                        model_dim_out=model_dim_out)

        self.attention_gamma = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
        self.attention_gamma_mlp = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)

        self.attention_res_layer = ResLayer(model_dim_out, with_res=False, p_dropout=p_dropout)

        self.attention_dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()


    def forward(self, x, mask=None, emb=None):
        nv = x.shape[2]

        if mask is not None:
            emb = add_mask_to_emb_dict(emb, mask)

        x_res = self.lin_skip_outer(x)

        x = self.attention_ada_ln(x, emb=emb)

        x, _ = self.attention_layer(x, 
                                    mask=mask if not self.mask_as_embedding else None
                                    )

        x = self.attention_dropout(x)
        
        x = x_res + self.attention_gamma*x

        x_res = x

        x = self.attention_ada_ln_mlp(x, emb=emb)

        x = self.attention_res_layer(x)

        x = x_res + self.attention_gamma_mlp*x

        return x