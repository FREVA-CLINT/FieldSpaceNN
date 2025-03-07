from typing import List

import torch
import torch.nn as nn
import einops
import math

from ...modules.icon_grids.grid_layer import GridLayer

from ..transformer.attention import ChannelVariableAttention, ResLayer, AdaptiveLayerNorm, MultiHeadAttentionBlock

from ...modules.embedding.embedder import EmbedderSequential


class MGAttentionReductionLayer(nn.Module):
  
    def __init__(self, 
                 global_level_in,
                 model_dims_in: List,
                 model_dim_out: int,
                 att_dim= 128,
                 n_head_channels=16, 
                 embedder_grid: EmbedderSequential=None,
                 embedder_mlp: EmbedderSequential=None,
                 cross_var=False,
                 p_dropout=0) -> None: 
        
        super().__init__()
        
        self.register_buffer('grid_embedding_indices', torch.tensor(global_level_in))

        model_dim_total = int(torch.tensor(model_dims_in).sum())

        self.attention_ada_lns = nn.ModuleList()

        model_dim_att_out = math.ceil(model_dim_total/len(model_dims_in))*len(model_dims_in)

        self.lin_projections = nn.ModuleList()
        for model_dim_in in model_dims_in:
            self.lin_projections.append(nn.Linear(model_dim_in, model_dim_att_out, bias=False))
            self.attention_ada_lns.append(AdaptiveLayerNorm([model_dim_att_out], model_dim_att_out, embedder=embedder_grid))


        self.attention_layer = ChannelVariableAttention(model_dim_att_out, 
                                                        1,
                                                        att_dim=att_dim,
                                                        n_head_channels = n_head_channels, 
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

        v = []
        for level_idx, x in enumerate(x_levels):
            emb['GridEmbedder'] = self.grid_embedding_indices[[level_idx]]
            #add_coordinates_to_emb_dict()
            x = self.lin_projections[level_idx](x)
            v.append(x)
            x_levels[level_idx] = self.attention_ada_lns[level_idx](x, emb=emb)

        v = torch.stack(v, dim=-2)
        x = torch.stack(x_levels, dim=-2)
        if mask_levels is not None:
            mask = torch.stack(mask_levels, dim=-1)
        else:
            mask = None
            
        b, n, nv, g, c = x.shape
        if self.cross_var:
            x = einops.rearrange(x, "b n v g c -> b n (v g) c")
            v = einops.rearrange(v, "b n v g c -> b n (v g) c")
            mask = einops.rearrange(mask, "b n v g -> b n (v g)") if mask is not None else mask
        else:
            x = einops.rearrange(x, "b n v g c -> b (n v) g c")
            v = einops.rearrange(v, "b n v g c -> b (n v) g c")
            mask = einops.rearrange(mask, "b n v g -> b (n v) g") if mask is not None else mask

        x, _ = self.attention_layer(x, v=v, mask=mask)
        
        if self.cross_var:
            x = einops.rearrange(x, "b n (v g) c -> b n v g c", v=nv, g=g)
        else:
            x = einops.rearrange(x, "b (n v) g c -> b n v g c", n=n, v=nv)

        x = einops.rearrange(x, "b n v g c -> b n v (g c)")

        x = x_skip + self.attention_gamma*x

        x_skip = self.lin_skip_mlp(x)

        x = self.attention_ada_ln_mlp(x,emb=emb)

        x = self.attention_mlp(x)

        x = x_skip + self.attention_gamma_mlp*x

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = torch.stack(mask_levels, dim=-1).sum(dim=-1) == len(mask_levels)
        else:
            mask_out = None

        return x, mask_out
    
class MGDiffMAttentionReductionLayer(nn.Module):
  
    def __init__(self, 
                 global_level_diffs_in,
                 model_dims_in: List,
                 model_dim_out: int,
                 att_dim= 128,
                 n_head_channels=16,
                 embedder_grid: EmbedderSequential=None,
                 embedder_mlp: EmbedderSequential=None,
                 cross_var=False,
                 p_dropout=0) -> None: 
        
        super().__init__()
        
        grid_emb_shape = embedder_grid.embedders['GridEmbedder'].embedding_fn.weight.shape[0]
        
        assert grid_emb_shape % 2 == 1

        index_offset = (grid_emb_shape - 1) // 2
        grid_level_indices = global_level_diffs_in + index_offset

        self.register_buffer('grid_embedding_indices', torch.tensor(grid_level_indices))

        model_dim_total = int(torch.tensor(model_dims_in).sum())


        self.lin_projections = nn.ModuleList()
        for model_dim_in in model_dims_in:
            lin_proj = nn.Linear(model_dim_in, model_dim_out, bias=False) if model_dim_in != model_dim_out else nn.Identity()
            self.lin_projections.append(lin_proj)
        
        self.attention_ada_ln = AdaptiveLayerNorm([len(model_dims_in), model_dim_out], model_dim_out, embedder=embedder_grid)
        self.attention_ada_ln_mlp = AdaptiveLayerNorm([len(model_dims_in), model_dim_out], model_dim_out, embedder=embedder_grid)

        self.attention_layer = ChannelVariableAttention(model_dim_out, 
                                                        1,
                                                        att_dim=att_dim,
                                                        n_head_channels = n_head_channels, 
                                                        model_dim_out=model_dim_out,
                                                        with_res=False)

        #self.lin_skip_mlp = nn.Linear(model_dim_out*len(model_dims_in), model_dim_out)
        self.lin_skip_outer = nn.Linear(model_dim_total, model_dim_out)

        self.attention_gamma = nn.Parameter(torch.ones(len(model_dims_in), model_dim_out)*1e-6, requires_grad=True)
        self.attention_gamma_mlp = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)

        self.attention_mlp = ResLayer(model_dim_out*len(model_dims_in), model_dim_out=model_dim_out, with_res=False, p_dropout=p_dropout)
        self.cross_var = cross_var

    def forward(self, x_levels, mask_levels=None, emb=None):
        
        x_skip_outer = self.lin_skip_outer(torch.concat(x_levels, dim=-1))

        for level_idx, x in enumerate(x_levels):
            x_levels[level_idx] = self.lin_projections[level_idx](x)
        
        x_skip = x = torch.stack(x_levels, dim=-2)

        emb['GridEmbedder'] = self.grid_embedding_indices
        x = self.attention_ada_ln(x, emb=emb)

        if mask_levels is not None and mask_levels[0] is not None:
            mask = torch.stack(mask_levels, dim=-1)
        else:
            mask = None

        b, n, nv, g, c = x.shape
        if self.cross_var:
            x = einops.rearrange(x, "b n v g c -> b n (v g) c")
            mask = einops.rearrange(mask, "b n v g -> b n (v g)") if mask is not None else mask
        else:
            x = einops.rearrange(x, "b n v g c -> b (n v) g c")
            mask = einops.rearrange(mask, "b n v g -> b (n v) g") if mask is not None else mask

        x, _ = self.attention_layer(x, mask=mask)
        
        if self.cross_var:
            x = einops.rearrange(x, "b n (v g) c -> b n v g c", v=nv, g=g)
        else:
            x = einops.rearrange(x, "b (n v) g c -> b n v g c", n=n, v=nv)

        x = x_skip + self.attention_gamma*x

        #x_skip = self.lin_skip_mlp(einops.rearrange(x, "b n v g c -> b n v (g c)"))

        x = self.attention_ada_ln_mlp(x, emb=emb)

        x = einops.rearrange(x, "b n v g c -> b n v (g c)")

        x = self.attention_mlp(x)

        x = x_skip_outer + self.attention_gamma_mlp*x

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = torch.stack(mask_levels, dim=-1).sum(dim=-1) == len(mask_levels)
        else:
            mask_out = None

        return x, mask_out

class MGDiffAttentionReductionLayer(nn.Module):
  
    def __init__(self, 
                 global_level_diffs_in,
                 model_dims_in: List,
                 model_dim_out: int,
                 att_dim= 128,
                 n_head_channels=16, 
                 embedder_grid: EmbedderSequential=None,
                 embedder_mlp: EmbedderSequential=None,
                 cross_var=False,
                 p_dropout=0) -> None: 
        
        super().__init__()

        grid_emb_shape = embedder_grid.embedders['GridEmbedder'].embedding_fn.weight.shape[0]
        
        assert grid_emb_shape % 2 == 1

        index_offset = (grid_emb_shape - 1) // 2
        grid_level_indices = global_level_diffs_in + index_offset

        self.register_buffer('grid_embedding_indices', torch.tensor(grid_level_indices))

        model_dim_total = int(torch.tensor(model_dims_in).sum())

        self.attention_ada_lns = nn.ModuleList()
        #self.attention_ada_lns = nn.ModuleList()

        model_dim_att = math.ceil(model_dim_total/len(model_dims_in))*len(model_dims_in)

        self.lin_projections = nn.ModuleList()
        for model_dim_in in model_dims_in:
            self.lin_projections.append(nn.Linear(model_dim_in, model_dim_att, bias=False))
        
        self.attention_ada_ln_k = AdaptiveLayerNorm([len(model_dims_in), model_dim_att], model_dim_att, embedder=embedder_grid)
        self.attention_ada_ln_q = AdaptiveLayerNorm([len(model_dims_in), model_dim_att], model_dim_att, embedder=embedder_grid)

        self.attention_layer = MultiHeadAttentionBlock(
            att_dim, model_dim_out, att_dim //n_head_channels, input_dim=model_dim_att, qkv_proj=True, v_proj=True
            )   
        

        self.lin_skip_att = nn.Linear(model_dim_total, model_dim_out)
        self.lin_skip_mlp = nn.Linear(model_dim_out, model_dim_out)

        self.attention_gamma = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
        self.attention_gamma_mlp = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)

        self.attention_ada_ln_mlp = AdaptiveLayerNorm([model_dim_out], model_dim_out, embedder=embedder_mlp)
        self.attention_mlp = ResLayer(model_dim_out, model_dim_out=model_dim_out, with_res=False, p_dropout=p_dropout)
        self.cross_var = cross_var


    def forward(self, x_levels, mask_levels=None, emb=None):
        
        
        x_skip = self.lin_skip_att(torch.concat(x_levels, dim=-1))

        for level_idx, x in enumerate(x_levels):
            x_levels[level_idx] = self.lin_projections[level_idx](x)
        
        x = torch.stack(x_levels, dim=-2)

        emb['GridEmbedder'] = self.grid_embedding_indices
        xkv = self.attention_ada_ln_k(x, emb=emb)
        xq = self.attention_ada_ln_q(x, emb=emb).mean(dim=-2, keepdim=True)

        if mask_levels is not None:
            mask = torch.stack(mask_levels, dim=-1)
        else:
            mask = None

        b, n, nv, g, c = xq.shape
        if self.cross_var:
            xkv = einops.rearrange(xkv, "b n v g c -> (b n) (v g) c")
            xq = einops.rearrange(xq, "b n v g c -> (b n) (v g) c")
            mask = einops.rearrange(mask, "b n v g -> (b n) (v g)") if mask is not None else mask
        else:
            xkv = einops.rearrange(xkv, "b n v g c -> (b n v) g c")
            xq = einops.rearrange(xq, "b n v g c -> (b n v) g c")
            mask = einops.rearrange(mask, "b n v g -> (b n v) g") if mask is not None else mask

        x = self.attention_layer(q=xq, k=xkv, v=xkv,  mask=mask)
        
        if self.cross_var:
            x = einops.rearrange(x, "(b n) (v g) c -> b n v g c", b=b, n=n, v=nv, g=g)
        else:
            x = einops.rearrange(x, "(b n v) g c -> b n v g c", b=b, n=n, v=nv, g=g)

        x = einops.rearrange(x, "b n v g c -> b n v (g c)")

        x = x_skip + self.attention_gamma*x

        x_skip = self.lin_skip_mlp(x)

        x = self.attention_ada_ln_mlp(x,emb=emb)

        x = self.attention_mlp(x)

        x = x_skip + self.attention_gamma_mlp*x

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = torch.stack(mask_levels, dim=-1).sum(dim=-1) == len(mask_levels)
        else:
            mask_out = None

        return x, mask_out
    

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
    
    
class SumReductionLayer(nn.Module):
  
    def __init__(self) -> None: 
        super().__init__()


    def forward(self, x_levels, mask_levels=None, emb=None):

        if mask_levels is not None and mask_levels[0] is not None:
            mask_out = torch.stack(mask_levels, dim=-1).sum(dim=-1) == len(mask_levels)
        else:
            mask_out = None
        
        x_out = torch.stack(x_levels, dim=-1).sum(dim=-1)

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