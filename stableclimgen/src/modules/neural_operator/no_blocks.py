
import torch
import torch.nn.functional as F
import torch.nn as nn

from .neural_operator import NoLayer
from.no_helpers import add_coordinates_to_emb_dict, add_mask_to_emb_dict, update_mask

from ..transformer.attention import ChannelVariableAttention, ResLayer, AdaptiveLayerNorm

from ..icon_grids.grid_attention import GridAttention
from ...modules.embedding.embedder import EmbedderSequential


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

class NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 embedder: EmbedderSequential=None,
                 cross_no = False,
                 with_gamma = False,
                 mask_as_embedding=False
                ) -> None: 
      
        super().__init__()

        self.mask_as_embedding = mask_as_embedding
        if cross_no:
            self.no_conv = CrossNOConv(model_dim_in, model_dim_out, no_layer)
        else:
            self.no_conv = NOConv(model_dim_in, model_dim_out, no_layer)
        
        self.level_diff = (no_layer.global_level_decode - no_layer.global_level_encode)

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

        self.activation = nn.SiLU()

        if embedder is not None and 'CoordinateEmbedder' in embedder.embedders.keys():
            self.grid_layer = no_layer.grid_layer_decode
        

    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        
        x_res = x
        
        x_res = get_residual(x, self.level_diff, mask=mask if not self.mask_as_embedding else None)

        x_conv, mask = self.no_conv(x, 
                                coords_encode=coords_encode, 
                                coords_decode=coords_decode, 
                                indices_sample=indices_sample, 
                                mask=mask, 
                                emb=emb)

        if hasattr(self, 'grid_layer'):
            emb = add_coordinates_to_emb_dict(self.grid_layer, indices_sample["indices_layers"] if indices_sample else None, emb)

        if self.mask_as_embedding and mask is not None:
            emb = add_mask_to_emb_dict(emb, mask)

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
                 cross_no = False,
                 with_gamma=False,
                 mask_as_embedding=False
                ) -> None: 
      
        super().__init__()

        self.mask_as_embedding = mask_as_embedding

        if cross_no:
            self.no_conv = CrossNOConv(model_dim_in, model_dim_out, no_layer)
        else:
            self.no_conv = NOConv(model_dim_in, model_dim_out, no_layer)
        
        self.level_diff = (no_layer.global_level_decode - no_layer.global_level_encode)

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
            self.grid_layer_1 = no_layer.grid_layer_encode
            self.grid_layer_2 = no_layer.grid_layer_decode


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

class NOConv(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 p_dropout=0.
                ) -> None: 
      
        super().__init__()

        self.no_layer = no_layer

        self.no_dims = self.no_layer.n_params_no 
        self.global_level_in = self.no_layer.global_level_encode
        self.global_level_out = self.no_layer.global_level_decode

        self.weights = nn.Parameter(torch.randn(*self.no_dims, model_dim_in, model_dim_out), requires_grad=True)
      #  self.weights = nn.Parameter(torch.ones(*self.no_dims, model_dim_in, model_dim_out), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(1)*1e-6, requires_grad=True)

    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        x, mask = self.no_layer.transform(x, coords_encode=coords_encode, indices_sample=indices_sample, mask=mask, emb=emb)

        x = torch.einsum("nbvpqrx, pqrxy -> nbvpqry", x, (1+self.gamma*self.weights))
    
        x, mask = self.no_layer.inverse_transform(x, coords_decode=coords_decode, indices_sample=indices_sample, mask=mask, emb=emb)

        x = x.contiguous()

        return x, mask

class CrossNOConv(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 p_dropout=0.
                ) -> None: 
      
        super().__init__()

        self.no_layer = no_layer

        self.no_dims = self.no_layer.n_params_no 
        self.global_level_in = self.no_layer.global_level_encode
        self.global_level_out = self.no_layer.global_level_decode

        self.linear_layer = nn.Linear(model_dim_in*int(torch.tensor(self.no_dims).prod()), model_dim_out*int(torch.tensor(self.no_dims).prod()), bias=False)

    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        x, mask = self.no_layer.transform(x, coords_encode=coords_encode, indices_sample=indices_sample, mask=mask, emb=emb)

        x_shape = x.shape
        x = x.view(*x.shape[:3],-1)
        x = self.linear_layer(x)
        x = x.view(*x_shape[:-1],-1)
    
        x, mask = self.no_layer.inverse_transform(x, coords_decode=coords_decode, indices_sample=indices_sample, mask=mask, emb=emb)

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
                                                        with_res=False)

        self.attention_gamma = nn.Parameter(torch.ones(model_dim_in)*1e-6, requires_grad=True)
        self.attention_gamma_mlp = nn.Parameter(torch.ones(model_dim_in)*1e-6, requires_grad=True)

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