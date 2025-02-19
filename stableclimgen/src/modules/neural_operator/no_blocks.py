from typing import List, Dict

import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import einops
import math
from ...modules.icon_grids.grid_layer import GridLayer

from .neural_operator import NoLayer, get_no_layer
from ..transformer.attention import ChannelVariableAttention, ResLayer, AdaptiveLayerNorm

from ..icon_grids.grid_attention import GridAttention
from ...modules.embedding.embedder import EmbedderSequential, EmbedderManager, BaseEmbedder

from ...utils.helpers import check_get_missing_key

class NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 p_dropout=0.,
                 embedder: EmbedderSequential=None,
                 cross_no = False,
                ) -> None: 
      
        super().__init__()

        if cross_no:
            self.no_conv = CrossNOConv(model_dim_in, model_dim_out, no_layer)
        else:
            self.no_conv = NOConv(model_dim_in, model_dim_out, no_layer)
        
        self.level_diff = (no_layer.global_level_decode - no_layer.global_level_encode)

        self.lin_skip_outer = nn.Linear(model_dim_in, model_dim_out, bias=True)
        self.lin_skip_inner = nn.Linear(model_dim_in, model_dim_out, bias=True)

        self.layer_norm1 = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder)
        self.layer_norm2 = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder)

        self.mlp_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim_out, model_dim_out , bias=False),
            nn.Dropout(p_dropout) if p_dropout>0 else nn.Identity(),
            nn.Linear(model_dim_out, model_dim_out, bias=False)
        )

        self.activation = nn.SiLU()
        

    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        
        x_res = x
        
        x_res = get_residual(x, self.level_diff, mask=mask)

        x_conv, mask = self.no_conv(x, coords_encode=coords_encode, coords_decode=coords_decode, indices_sample=indices_sample, mask=mask, emb=emb)

        x = self.layer_norm1(x_conv, emb=emb) + self.lin_skip_inner(x_res)

        x = self.mlp_layer(x) 
        
        x = self.layer_norm2(x, emb=emb)

        x = x + self.lin_skip_outer(x_res)

        x = self.activation(x)
    
        return x, mask


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


class PreActivation_NOBlock(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 level_data_in,
                 no_layer: NoLayer,
                 p_dropout=0.,
                ) -> None: 
      
        super().__init__()

        self.no_conv = NOConv(model_dim_in, model_dim_out, no_layer)
        
        self.level_diff = (self.no_conv.global_level_out - level_data_in)

        self.lin_skip_outer = nn.Linear(model_dim_in, model_dim_out, bias=False)
        self.lin_skip_inner = nn.Linear(model_dim_in, model_dim_out, bias=False)

        self.layer_norm1 = AdaptiveLayerNorm([model_dim_in], model_dim_out)

        self.mlp_layer = nn.Sequential(
            nn.SiLU(),
            AdaptiveLayerNorm([model_dim_out], model_dim_out),
            nn.Linear(model_dim_out, model_dim_out , bias=False),
            nn.Dropout(p_dropout) if p_dropout>0 else nn.Identity(),
            nn.Linear(model_dim_out, model_dim_out, bias=False),
        )

        self.activation = nn.SiLU()
        

    def forward(self, x, coords_encode=None, coords_decode=None, indices_sample=None, mask=None, emb=None):
        
        x_res = x
        
        x_res = get_residual(x, self.level_diff, mask=mask)
        
        x = self.layer_norm1(x)

        x_conv, mask = self.no_conv(x, coords_encode=coords_encode, coords_decode=coords_decode, indices_sample=indices_sample, mask=mask, emb=emb)

        x = x_conv + self.lin_skip_inner(x_res)

        x = self.mlp_layer(x) + self.lin_skip_outer(x_res)

        x = self.activation(x)
    
        return x, mask

class NOConv(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 no_layer: NoLayer,
                 p_dropout=0.,
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
                 p_dropout=0.,
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

class Serial_NOBlock(nn.Module):
  
    def __init__(self,
                 input_dim: int,
                 model_dims_out: List[int],
                 grid_layers: List[GridLayer],
                 layer_settings: List[dict],
                 input_level: int = 0,
                 output_dim: int = None,
                 rotate_coordinate_system: bool = True
                ) -> None: 
      
        super().__init__()

        self.layers = nn.ModuleList()

        current_level = input_level
        for layer_idx, layer_setting in enumerate(layer_settings):
            
            model_dim_in = input_dim if layer_idx==0 else model_dims_out[layer_idx-1]
            model_dim_out = model_dims_out[layer_idx]

            if 'NO' in layer_setting["type"]:
                global_level_decode = check_get_missing_key(layer_setting, "global_level_decode")
                global_level_no = check_get_missing_key(layer_setting, "global_level_no")
                no_layer_type = check_get_missing_key(layer_setting, "no_layer_type")
                no_layer_settings = check_get_missing_key(layer_setting, "no_layer_settings")

                global_level_in = current_level
                global_level_out = global_level_decode
                current_level = global_level_out

                no_layer = get_no_layer(no_layer_type,
                                        grid_layers[str(global_level_in)],
                                        grid_layers[str(global_level_no)],
                                        grid_layers[str(global_level_out)],
                                        precompute_encode=True,
                                        precompute_decode=True if layer_idx < len(model_dims_out)-1 else False,
                                        rotate_coordinate_system=rotate_coordinate_system,
                                        layer_settings=no_layer_settings)
                
                if "mlp" in layer_setting["type"]:
                    embedder = get_embedder_from_dict(layer_setting)
                    layer = NOBlock(
                                model_dim_in=model_dim_in,
                                model_dim_out=model_dim_out,
                                no_layer=no_layer,
                                embedder=embedder,
                                cross_no = 'cross_no' in layer_setting["type"]
                                )

            if "var_att" in layer_setting["type"]:
                embedder_att = get_embedder_from_dict(layer_setting)
                embedder_mlp = get_embedder_from_dict(layer_setting)

                layer = VariableAttention(
                    model_dim_in,
                    model_dim_out,
                    layer_setting["n_head_channels"],
                    att_dim=layer_setting.get("att_dim",None),
                    p_dropout=layer_setting.get("p_dropout",0),
                    embedder = embedder_att,
                    embedder_mlp= embedder_mlp
                )
            
            elif "spatial_att" in layer_setting["type"]:
                
                spatial_attention_configs = check_get_missing_key(layer_setting, "spatial_attention_configs")
                spatial_attention_configs["embedder_names"] = layer_setting.get("embedder_names",[[],[]])
                spatial_attention_configs["embed_confs"] = layer_setting.get("embed_confs",None)
                spatial_attention_configs["embed_mode"] = layer_setting.get("embed_mode","sum")

                layer = SpatialAttention(
                    model_dim_in,
                    model_dim_out,
                    grid_layers[str(current_level)],
                    layer_setting["n_head_channels"],
                    p_dropout=layer_setting.get("p_dropout",0),
                    spatial_attention_configs=spatial_attention_configs,
                    rotate_coord_system=rotate_coordinate_system
                )
                
            self.layers.append(layer)
            
        self.output_level = current_level
        self.output_layer = nn.Linear(model_dims_out[-1], output_dim, bias=False) if output_dim is not None else nn.Identity()


    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        for layer_idx, layer in enumerate(self.layers):
            
            if isinstance(layer, NOBlock):
                x, mask = layer(x, coords_encode=coords_in, coords_decode=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)
            elif isinstance(layer, SpatialAttention):
                x = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            else:
                x = layer(x, mask=mask, emb=emb)

        x = self.output_layer(x)

        return x, mask
    
class MGNO_Processing_Block(nn.Module):
  
    def __init__(self,
                 input_levels: List[int],
                 layer_settings_levels: List[List],
                 input_dims: List[int],
                 model_dims_out: List[List],
                 grid_layers: List[GridLayer],
                 rotate_coordinate_system: bool = True
                ) -> None: 
      
        super().__init__()

        self.output_levels = input_levels
        self.layers = nn.ModuleList()

        self.model_dims_out = []
        for level_idx, layer_settings_level in enumerate(layer_settings_levels):
            
            input_dim = input_dims[level_idx]
            current_level = input_levels[level_idx]

            level_layers_ = nn.ModuleList()

            for layer_idx, layer_setting in enumerate(layer_settings_level):
                model_dim_out = model_dims_out[level_idx][layer_idx]

                if 'NO' in layer_setting["type"]:
                    no_layer_type = check_get_missing_key(layer_setting, "no_layer_type")
                    no_layer_settings = check_get_missing_key(layer_setting, "no_layer_settings")

                    layer_setting['nh_in_encode'] = True
                    layer_setting['nh_in_decode'] = True

                    no_layer = get_no_layer(no_layer_type,
                                            grid_layers[str(current_level)],
                                            grid_layers[str(current_level)],
                                            grid_layers[str(current_level)],
                                            precompute_encode=True,
                                            precompute_decode=True,
                                            rotate_coordinate_system=rotate_coordinate_system,
                                            layer_settings=no_layer_settings)
                    
                    if "mlp" in layer_setting["type"]:
                        embedder = get_embedder_from_dict(layer_setting)
                        layer = NOBlock(
                                    model_dim_in=input_dim,
                                    model_dim_out=model_dim_out,
                                    no_layer=no_layer,
                                    embedder=embedder,
                                    cross_no = 'cross_no' in layer_setting["type"]
                                    )

                if "var_att" in layer_setting["type"]:
                    embedder_att = get_embedder_from_dict(layer_setting)
                    embedder_mlp = get_embedder_from_dict(layer_setting)

                    layer = VariableAttention(
                        input_dim,
                        model_dim_out,
                        layer_setting["n_head_channels"],
                        att_dim=layer_setting.get("att_dim",None),
                        p_dropout=layer_setting.get("p_dropout",0),
                        embedder = embedder_att,
                        embedder_mlp= embedder_mlp
                    )
                
                elif "spatial_att" in layer_setting["type"]:
                    
                    spatial_attention_configs = check_get_missing_key(layer_setting, "spatial_attention_configs")
                    spatial_attention_configs["embedder_names"] = layer_setting.get("embedder_names",[[],[]])
                    spatial_attention_configs["embed_confs"] = layer_setting.get("embed_confs",None)
                    spatial_attention_configs["embed_mode"] = layer_setting.get("embed_mode","sum")

                    layer = SpatialAttention(
                        input_dim,
                        model_dim_out,
                        grid_layers[str(current_level)],
                        layer_setting["n_head_channels"],
                        p_dropout=layer_setting.get("p_dropout",0),
                        spatial_attention_configs=spatial_attention_configs,
                        rotate_coord_system=rotate_coordinate_system
                    )
                
                input_dim = model_dim_out

                level_layers_.append(layer)

            self.layers.append(level_layers_)
            self.model_dims_out.append(model_dim_out)


    def forward(self, x_levels, coords_in=None, coords_out=None, indices_sample=None, mask_levels=None, emb=None):
        
        for level_idx, layer_levels in enumerate(self.layers):
            x = x_levels[level_idx]
            mask = mask_levels[level_idx]

            for layer_idx, layer in enumerate(layer_levels):

                if isinstance(layer, NOBlock):
                    x, mask = layer(x, coords_encode=coords_in, coords_decode=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)
                elif isinstance(layer, SpatialAttention):
                    x = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
                else:
                    x = layer(x, mask=mask, emb=emb)

            x_levels[level_idx] = x
            mask_levels[level_idx] = mask

        return x_levels, mask_levels

class LinearReductionLayer(nn.Module):
  
    def __init__(self, 
                 model_dims_in: List,
                 model_dim_out: int) -> None: 
        super().__init__()

        self.layer = nn.Linear(int(torch.tensor(model_dims_in).sum()), model_dim_out, bias=True)

    def forward(self, x_levels, mask_levels=None, emb=None):

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = torch.stack(mask_levels, dim=-1).sum(dim=-1) == len(mask_levels)
        else:
            mask_out = None
        
        x_out = self.layer(torch.concat(x_levels, dim=-1))

        return x_out, mask_out

class IdentityReductionLayer(nn.Module):
  
    def __init__(self) -> None: 
        super().__init__()

    def forward(self, x_levels, mask_levels=None, emb=None):

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = mask_levels[0]
        else:
            mask_out = None

        return x_levels[0], mask_out
    
class MGAttentionReductionLayer(nn.Module):
  
    def __init__(self, 
                 global_level_in,
                 model_dims_in: List,
                 model_dim_out: int,
                 embedder_grid: EmbedderSequential=None,
                 embedder_mlp: EmbedderSequential=None,
                 cross_var=False,
                 p_dropout=0) -> None: 
        
        super().__init__()
        
        self.register_buffer('grid_levels', torch.tensor(global_level_in))

        model_dim_total = int(torch.tensor(model_dims_in).sum())
        self.layer = nn.Linear(model_dim_total, model_dim_out, bias=True)

        self.attention_ada_lns = nn.ModuleList()

        model_dim_att_out = math.ceil(model_dim_total/len(model_dims_in))*len(model_dims_in)

        self.lin_projections = nn.ModuleList()
        for model_dim_in in model_dims_in:
            self.lin_projections.append(nn.Linear(model_dim_in, model_dim_att_out, bias=False))
            self.attention_ada_lns.append(AdaptiveLayerNorm([model_dim_att_out], model_dim_att_out, embedder=embedder_grid))


        self.attention_layer = ChannelVariableAttention(model_dim_att_out, 
                                                        1, 
                                                        model_dim_att_out//len(model_dims_in), 
                                                        model_dim_out=model_dim_att_out//len(model_dims_in),
                                                        with_res=False)

        self.lin_skip_att = nn.Linear(model_dim_total, model_dim_att_out)
        self.lin_skip_mlp = nn.Linear(model_dim_att_out, model_dim_out)

        self.attention_gamma = nn.Parameter(torch.ones(model_dim_att_out)*1e-6, requires_grad=True)
        self.attention_gamma_mlp = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)

        self.attention_ada_ln_mlp = AdaptiveLayerNorm([model_dim_att_out], model_dim_att_out, embedder=embedder_mlp)
        self.attention_mlp = ResLayer(model_dim_att_out, model_dim_out=model_dim_out, with_res=False, p_dropout=p_dropout)
        self.cross_var = cross_var

    def forward(self, x_levels, mask_levels=None, emb=None):
        
        
        x_skip = self.lin_skip_att(torch.concat(x_levels, dim=-1))

       # v = []
        for level_idx, x in enumerate(x_levels):
            emb['GridEmbedder'] = self.grid_levels[level_idx]
            x = self.lin_projections[level_idx](x)
      #      v.append(x)
            x_levels[level_idx] = self.attention_ada_lns[level_idx](x, emb=emb)

       # v = torch.stack(v, dim=-2)
        x = torch.stack(x_levels, dim=-2)
        if mask_levels is not None:
            mask = torch.stack(mask_levels, dim=-1)
        else:
            mask = None

        b, n, nv, g, c = x.shape
        if self.cross_var:
            x = einops.rearrange(x, "b n v g c -> b n (v g) c")
       #     v = einops.rearrange(v, "b n v g c -> b n (v g) c")
            mask = einops.rearrange(mask, "b n v g -> b n (v g)") if mask is not None else mask
        else:
            x = einops.rearrange(x, "b n v g c -> b (n v) g c")
          #  v = einops.rearrange(v, "b n v g c -> b (n v) g c")
            mask = einops.rearrange(mask, "b n v g -> b (n v) g") if mask is not None else mask

        x, _ = self.attention_layer(x, mask=mask)
        
        if self.cross_var:
            x = einops.rearrange(x, "b n (v g) c -> b n v g c", v=nv, g=g)
        else:
            x = einops.rearrange(x, "b (n v) g c -> b n v g c", n=n, v=nv)

        x = einops.rearrange(x, "b n v g c -> b n v (g c)")

        x = x_skip + self.attention_gamma*x

        x_skip = self.lin_skip_mlp(x)

        x = self.attention_ada_ln_mlp(x)

        x = self.attention_mlp(x)

        x = x_skip + self.attention_gamma_mlp*x

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = torch.stack(mask_levels, dim=-1).sum(dim=-1) == len(mask_levels)
        else:
            mask_out = None

        return x, mask_out

class MGNO_EncoderDecoder_Block(nn.Module):
  
    def __init__(self,
                 input_levels: List[int],
                 input_dims: List[int],
                 global_levels_decode: List[int],
                 global_levels_no: List[int],
                 model_dims_out: List[int],
                 grid_layers: List[GridLayer],
                 layer_settings: dict,
                 rotate_coordinate_system: bool = True,
                 rule = 'fc', # ">" "<"
                 mg_reduction = 'linear',
                 mg_reduction_embed_confs: Dict = None,
                 mg_reduction_embed_names: List = None,
                 mg_reduction_embed_names_mlp: List = None,
                 mg_reduction_embed_mode: str = 'sum',
                 p_dropout=0
                ) -> None: 
      
        super().__init__()

        self.output_levels = global_levels_decode
        self.model_dims_out = model_dims_out

        self.layers = nn.ModuleList()
        self.reduction_layers = nn.ModuleList()
        self.module_indices = []
        self.output_indices = []
        self.layer_indices = []

        layer_index = 0
        for output_idx, output_level in enumerate(global_levels_decode):

            input_indices = []
            layer_indices = []
            mg_input_dims = []
            mg_input_levels = []
            for input_idx, input_level in enumerate(input_levels):

                level_diff = output_level - input_level

                if rule == "<" and level_diff>0:
                   # self.layers.append(nn.Identity())
                    continue

                elif rule == ">" and level_diff<0:
                  #  self.layers.append(nn.Identity())
                    continue
                
                input_indices.append(input_idx)

                if list(layer_settings.keys())[0].isnumeric():
                    layer_setting = layer_settings[str(level_diff)]
                else:
                    layer_setting = layer_settings
                
                model_dim_in = input_dims[input_idx]
                model_dim_out = model_dims_out[output_idx]
                mg_input_dims.append(model_dim_out)
                mg_input_levels.append(input_level)

                global_level_no = global_levels_no[output_idx]

                layer_type = check_get_missing_key(layer_setting, "type")

                if 'NO' in layer_type:

                    no_layer_settings = check_get_missing_key(layer_setting, "no_layer_settings")
                    no_layer_type = check_get_missing_key(layer_setting, "no_layer_type")

                    no_layer = get_no_layer(no_layer_type,
                                            grid_layers[str(input_level)],
                                            grid_layers[str(global_level_no)],
                                            grid_layers[str(output_level)],
                                            precompute_encode=True,
                                            precompute_decode=True,
                                            rotate_coordinate_system=rotate_coordinate_system,
                                            layer_settings=no_layer_settings)
                
                    if "mlp" in layer_type:
                        embedder = get_embedder_from_dict(layer_setting)
                        layer = NOBlock(
                                    model_dim_in=model_dim_in,
                                    model_dim_out=model_dim_out,
                                    no_layer=no_layer,
                                    cross_no='cross_no' in layer_type,
                                    embedder=embedder,
                                    p_dropout=p_dropout
                                    )
                elif 'linear' in layer_type:
                    layer = nn.Linear(model_dim_in, model_dim_out) if model_dim_in!=model_dim_out else nn.Identity()
                    
                self.layers.append(layer)
                layer_indices.append(layer_index)
                layer_index += 1

            if len(mg_input_dims)>1:
                if mg_reduction == 'linear':
                    reduction_layer = LinearReductionLayer(mg_input_dims, model_dim_out)

                elif mg_reduction == 'MGAttention':
                    if mg_reduction_embed_names is not None:
                        embedder = get_embedder(embed_names=mg_reduction_embed_names,
                                                embed_confs=mg_reduction_embed_confs,
                                                embed_mode=mg_reduction_embed_mode)
                        
                    else:
                        embedder = None
                    if mg_reduction_embed_names_mlp is not None:
                        embedder_mlp = get_embedder(embed_names=mg_reduction_embed_names_mlp,
                                                embed_confs=mg_reduction_embed_confs,
                                                embed_mode=mg_reduction_embed_mode)
                    else:
                        embedder_mlp = None
                        
                    reduction_layer = MGAttentionReductionLayer(mg_input_levels,
                                                                mg_input_dims, 
                                                                model_dim_out,
                                                                embedder_grid=embedder,
                                                                embedder_mlp=embedder_mlp,
                                                                p_dropout=p_dropout)
            else:
                reduction_layer = IdentityReductionLayer()
            
            self.reduction_layers.append(reduction_layer)

            self.layer_indices.append(layer_indices)
            self.output_indices.append(input_indices)
            

    def forward(self, x_levels, coords_in=None, coords_out=None, indices_sample=None, mask_levels=None, emb=None):

        x_levels_out = []
        mask_levels_out = []

        for output_index, input_indices in enumerate(self.output_indices):

            outputs_ = []
            masks_ = []

            for layer_index, input_index in enumerate(input_indices):
                x = x_levels[input_index]
                mask = mask_levels[input_index]
                
                layer = self.layers[self.layer_indices[output_index][layer_index]]
                
                if isinstance(layer, nn.Identity) or isinstance(layer, nn.Linear):
                    x_out = layer(x)
                    mask_out = mask
                else:
                    x_out, mask_out = layer(x, coords_encode=coords_in, coords_decode=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

                if mask_out is not None:
                    mask_out = mask_out.view(x_out.shape[:3])

                outputs_.append(x_out)
                masks_.append(mask_out)

            x_out, mask_out = self.reduction_layers[output_index](outputs_, mask_levels=masks_, emb=emb)

            x_levels_out.append(x_out)
            mask_levels_out.append(mask_out)

        return x_levels_out, mask_levels_out


def get_embedder_from_dict(dict_: dict):
    if "embed_names" in dict_.keys() and "embed_confs" in dict_.keys():
        embed_mode = dict_.get("mode","sum")
        return get_embedder(dict_["embed_names"],
                            dict_["embed_confs"],
                            embed_mode)
    else:
        return None

def get_embedder(embed_names:list, 
                 embed_confs:list, 
                 embed_mode: list):
    
    emb_dict = nn.ModuleDict()
    for embed_name in embed_names:
        emb: BaseEmbedder = EmbedderManager().get_embedder(embed_name, **embed_confs[embed_name])
        emb_dict[emb.name] = emb     
        
    embedder = EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count = 1)

    return embedder
class SpatialAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out, 
                 grid_layer,
                 n_head_channels,
                 p_dropout=0.,
                 rotate_coord_system=True,
                 spatial_attention_configs = None
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
       
    def forward(self, x, indices_sample=None, mask=None, emb=None):
        nv = x.shape[2]

        # insert time dimension
        x = x.unsqueeze(dim=1)
        mask = mask.unsqueeze(dim=1) if mask is not None else mask

        x = self.grid_attention_layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
        x = x.squeeze(dim=1)
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


class VariableAttention(nn.Module):
  
    def __init__(self,
                 model_dim_in, 
                 model_dim_out,
                 n_head_channels,
                 att_dim=None,
                 p_dropout = 0,
                 embedder: EmbedderSequential=None,
                 embedder_mlp: EmbedderSequential=None
                ) -> None: 
      
        super().__init__()

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

        x_res = self.lin_skip_outer(x)

        x = self.attention_ada_ln(x, emb=emb)

        x, _ = self.attention_layer(x, mask=mask)

        x = self.attention_dropout(x)
        
        x = x_res + self.attention_gamma*x

        x_res = x

        x = self.attention_ada_ln_mlp(x, emb=emb)

        x = self.attention_res_layer(x)

        x = x_res + self.attention_gamma_mlp*x

        return x