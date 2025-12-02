from typing import List,Dict
from einops import rearrange
import copy

import math
import torch
import torch.nn as nn

from ..base import get_layer, IdentityLayer, LinEmbLayer, MLP_fac, EmbIdLayer, LayerNorm
from .mg_base import combine_zooms,MGFieldLayer
from ..factorization import get_cp_tensor,get_cp_tensors, get_cp_equation

from ...modules.grids.grid_layer import GridLayer
from ...modules.transformer.transformer_base import SelfAttention,safe_scaled_dot_product_attention

from ...modules.embedding.embedder import EmbedderSequential
from ..grids.grid_utils import get_matching_time_patch, insert_matching_time_patch, get_sample_configs


def get_compression_dims(dims, zoom):
    in_f = [4 for _ in dims]
    if len(in_f) == zoom +1:
        in_f[0] = 12
    out_f = [4 if d==-1 else d for d in dims]
    return in_f, out_f

class LoRA(nn.Module):

    def __init__(self, in_features, out_features, embedder=None, rank=32, layer_confs={}, layer_confs_emb={}):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.embedder = embedder

        e_dim = embedder.get_out_channels
        self.rank = rank

        self.x_proj = get_layer(in_features, out_features, layer_confs=layer_confs, bias=False) if in_features != out_features else IdentityLayer()

        self.U = get_layer(in_features, [*out_features[:-1], rank], layer_confs=layer_confs, bias=False)
        self.V = get_layer([*in_features[:-1], rank], out_features, layer_confs=layer_confs, bias=False)
        self.a = get_layer([e_dim], [rank], layer_confs=layer_confs_emb, bias=True)


    def forward(self, x, emb={}, sample_configs={}):
        
        e = self.embedder(emb, sample_configs)

        xU = self.U(x, emb=emb, sample_configs=sample_configs)
        z  = xU * self.a(e, emb=emb, sample_configs=sample_configs)
        x = self.x_proj(x, emb=emb, sample_configs=sample_configs) + self.V(z, emb=emb, sample_configs=sample_configs)

        return x

class HeadGate(nn.Module):

    def __init__(self, n_heads=4, embedder=None, scale_limit=1, layer_confs={}):
        super().__init__()
        self.embedder = embedder
   
        e_dim = embedder.get_out_channels

        self.scale_limit = scale_limit

        self.g = get_layer([e_dim], [n_heads], layer_confs=layer_confs, bias=True)

    def forward(self, x, emb={}, sample_configs={}):
        
        e = self.embedder(emb, sample_configs)

        g = (1.0 + self.scale_limit * torch.tanh(self.g(e, emb=emb, sample_configs=sample_configs))).unsqueeze(-1)
        x = (x.view(*x.shape[:-1],g.shape[-2],-1) * g).view(*x.shape[:-1],-1)

        return x

class PoolingLayer(nn.Module):

    def __init__(self, n_pool):
        super().__init__()
        self.n_pool = n_pool

    def forward(self, x: torch.tensor, emb={}, sample_configs={}, mask=None):
        
        if isinstance(self.n_pool, float):
            n_pool = int(x.shape[-2] * self.n_pool)
        else:
            n_pool = self.n_pool

        r = (x - x.mean(dim=-2, keepdim=True)).norm(p=2, dim=-1)

        keepids = r.topk(k=n_pool, dim=-1).indices

        x = torch.gather(x, dim=-2, index=keepids.unsqueeze(dim=-1).expand(*[-1]*(x.dim()-1),x.shape[-1]))

        if mask is not None:
            mask = torch.gather(mask, dim=-2, index=keepids.unsqueeze(dim=-1))

        return x, mask

class MultiZoomSelfAttention(nn.Module):
  
    def __init__(self, 
                 grid_layers: GridLayer,
                 att_zoom: int,
                 in_features: int,
                 out_features: int,
                 q_zooms: List,
                 kv_zooms: List,
                 new_zooms: List=[],
                 mult: int=1,
                 dropout: float=0.0,
                 num_heads: int=None,
                 att_dim=None,
                 n_head_channels: int=16, 
                 embedders: Dict[str, EmbedderSequential] = {},
                 compression_dims_kv: Dict[int, int]= {},
                 pooling_dims_kv: Dict[int, int]= {},
                 compression_dims_q: Dict[int, int]= {},
                 compression_zooms: Dict[int,int] = {},
                 qkv_emb_projection_settings = {},
                 cross_mode = False,
                 with_nh = True,
                 var_att = False,
                 common_affine = True,
                 lora = False,
                 film = False,
                 head_gate = False,
                 head_gate_scale_limit = .5,
                 layer_confs: Dict = {},
                 layer_confs_emb= {},
                 composed_residual=False) -> None:
        
        super().__init__()
        
        layer_confs_kv = copy.deepcopy(layer_confs)
        layer_confs_kv['bias'] = True
        
        self.grid_layers = grid_layers
        self.att_zoom = att_zoom
        grid_layer = grid_layers[str(att_zoom)]

        self.with_nh = with_nh

        self.qkv_affine_layers = nn.ModuleDict()
        self.q_affine_layers = nn.ModuleDict()
        self.kv_affine_layers = nn.ModuleDict()
        self.mlp_affine_layers = nn.ModuleDict()
        self.q_headgate_layers = nn.ModuleDict()
        self.kv_headgate_layers = nn.ModuleDict()

        self.out_attention_layers = nn.ModuleDict()

        self.q_projection_layers = nn.ModuleDict()
        self.kv_projection_layers = nn.ModuleDict()

        self.kv_compression_layers = nn.ModuleDict()

        self.mlps = nn.ModuleDict()
        self.out_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()

        att_dim = out_features if att_dim is None else att_dim
        self.num_heads = num_heads if num_heads else in_features // n_head_channels

        compression_zooms_comb = {}
        compression_zooms_comb.update(compression_zooms)
        compression_zooms_comb.update(pooling_dims_kv)

        self.CA_layers = nn.ModuleDict()
        for in_zoom, comp_dim in compression_zooms_comb.items():
            
            if in_zoom in compression_zooms.keys():
                self.CA_layers[str(in_zoom)] = MGCompressionAttention(
                                    features=att_dim,
                                    embedder_q=embedders[str(comp_dim)],
                                    n_heads=self.num_heads,
                                    head_gate=head_gate,
                                    head_gate_scale_limit=head_gate_scale_limit,
                                    qkv_emb_projection_settings=qkv_emb_projection_settings,
                                    layer_confs=layer_confs,
                                    layer_confs_emb=layer_confs_emb
                                    )
            elif in_zoom in pooling_dims_kv.keys():
                self.CA_layers[str(in_zoom)] = PoolingLayer(n_pool=comp_dim)
        
        for in_zoom in embedders.keys():
            if int(in_zoom) in q_zooms+kv_zooms: 
                #if len(qkv_emb_projection_settings)==0:
                if int(in_zoom) in new_zooms:
                    self.qkv_affine_layers[in_zoom] = EmbIdLayer([att_dim], embedder=embedders[in_zoom], layer_confs_emb=layer_confs_emb)
                    #self.qkv_affine_layers[in_zoom] = IdentityLayer()
                elif cross_mode:
                    self.qkv_affine_layers[in_zoom] = IdentityLayer()
                else:
                    if common_affine:
                        self.qkv_affine_layers[in_zoom] = LinEmbLayer(in_features, [att_dim], layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[in_zoom] if film else None, layer_norm=True, layer_confs_emb=layer_confs_emb) 
                    else:
                        self.qkv_affine_layers[in_zoom] = LinEmbLayer(in_features, [att_dim], layer_confs=layer_confs, identity_if_equal=True, layer_norm=True, layer_confs_emb=layer_confs_emb) 
 

        for q_zoom in q_zooms:
            in_f = out_f = []

            if q_zoom in compression_dims_q.keys():
                in_f, out_f = get_compression_dims(compression_dims_q[q_zoom], q_zoom)

            in_features_q =  [*in_f, att_dim]
            out_features_q = [*out_f, att_dim]

            if not common_affine:
                self.q_affine_layers[str(q_zoom)] = LinEmbLayer(att_dim, [att_dim], layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[str(q_zoom)] if film else None, layer_confs_emb=layer_confs_emb) 
            else:
                self.q_affine_layers[str(q_zoom)] = IdentityLayer() 

            if cross_mode:
                self.q_projection_layers[str(q_zoom)] = EmbIdLayer(out_features_q, embedder=embedders[str(q_zoom)], layer_confs_emb=layer_confs_emb)
            elif int(q_zoom) in new_zooms:
                self.q_projection_layers[str(q_zoom)] = IdentityLayer()
            elif lora:
                self.q_projection_layers[str(q_zoom)] = LoRA(in_features_q, out_features_q, embedder=embedders[str(q_zoom)], **qkv_emb_projection_settings, layer_confs=layer_confs, layer_confs_emb=layer_confs_emb)
            else:
                self.q_projection_layers[str(q_zoom)] = get_layer(in_features_q, out_features_q, layer_confs=layer_confs) 

            if head_gate:
                self.q_headgate_layers[str(q_zoom)] = HeadGate(self.num_heads, embedder=embedders[str(q_zoom)], scale_limit=head_gate_scale_limit)
            else:
                self.q_headgate_layers[str(q_zoom)] = IdentityLayer()

            self.gammas[str(q_zoom)] = nn.Parameter(torch.ones(in_features)*1e-6,requires_grad=True)

            self.mlp_affine_layers[str(q_zoom)] = LinEmbLayer(in_features, att_dim, layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[str(q_zoom)] if film else None, layer_norm=True, layer_confs_emb=layer_confs_emb)

            self.out_attention_layers[str(q_zoom)] = get_layer(att_dim, in_features, layer_confs=layer_confs) if att_dim!=in_features else IdentityLayer()

            self.mlps[str(q_zoom)] = MLP_fac(att_dim, in_features, mult, dropout, layer_confs=layer_confs, gamma=True) 
            
            self.out_layers[str(q_zoom)] = get_layer(in_features, out_features, layer_confs=layer_confs) if in_features!=out_features else IdentityLayer()

       #self.omit_mask_zooms = []
        for kv_zoom in kv_zooms:
            
            if kv_zoom in compression_dims_kv.keys():
                in_f, out_f = get_compression_dims(compression_dims_kv[kv_zoom], kv_zoom)
                #self.omit_mask_zooms.append(kv_zoom)

                in_features_comp_kv = [*in_f, grid_layer.adjc.shape[-1], 2*att_dim] if with_nh else [*in_f, 2*att_dim]
                out_features_comp_kv = [*out_f, grid_layer.adjc.shape[-1], 2*att_dim] if with_nh else [*out_f, 2*att_dim]

                skip_dims = [False for _ in in_features_comp_kv]
                if with_nh:
                    skip_dims[-2] = True
                
                layer_confs_ = layer_confs.copy()
                layer_confs_['rank_channel'] = 0
                layer_confs_['skip_dims'] = skip_dims

                self.kv_compression_layers[str(kv_zoom)] = get_layer(in_features_comp_kv, out_features_comp_kv, layer_confs=layer_confs_)

            else:
                self.kv_compression_layers[str(kv_zoom)] = IdentityLayer()

            if not common_affine:
                self.kv_affine_layers[str(kv_zoom)] = LinEmbLayer(att_dim, [att_dim], layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[str(kv_zoom)] if film else None, layer_confs_emb=layer_confs_emb) 
            else:
                self.kv_affine_layers[str(kv_zoom)] = IdentityLayer() 

            if lora:
                self.kv_projection_layers[str(kv_zoom)] =LoRA([att_dim], [2 * att_dim], embedder=embedders[str(kv_zoom)], **qkv_emb_projection_settings, layer_confs=layer_confs, layer_confs_emb=layer_confs_emb)
            else:
                self.kv_projection_layers[str(kv_zoom)] = get_layer(att_dim, [2 * att_dim], layer_confs=layer_confs,bias=True) 
   
            if head_gate:
                self.kv_headgate_layers[str(kv_zoom)] = HeadGate(self.num_heads, embedder=embedders[str(kv_zoom)], scale_limit=head_gate_scale_limit)
            else:
                self.kv_headgate_layers[str(kv_zoom)] = IdentityLayer()

        self.grid_layer = grid_layer
        self.max_zoom = int(max(list(embedders.keys())))

        self.composed_residual = composed_residual
        if composed_residual:
            self.composed_residual_layer = ComposedResidual(att_dim, embedders, kv_zooms, q_zooms, layer_confs)

        assert out_features==in_features,"Module does not support differen in and out features"

        if not var_att:
            self.pattern = 'b v t (s n) (NH H) -> (b v t s) NH n H'
            self.kv_pattern = 'b v t (s n) m (NH H) -> (b v t s) NH (n m) H' if with_nh else self.pattern
            self.mask_pattern = 'b v t (s n) m 1 -> (b v t s) 1 1 (n m)' if with_nh else self.pattern
            self.reverse = '(b v t s) NH n H -> b v t s n (NH H)'
        else:
            self.pattern = 'b v t (s n) (NH H) -> (b t s) NH (v n) H'
            self.kv_pattern = 'b v t (s n) m (NH H) -> (b t s) NH (v n m) H' if with_nh else self.pattern
            self.mask_pattern = 'b v t (s n) m 1 -> (b t s) 1 1 (v n m)' if with_nh else self.pattern
            self.reverse = '(b t s) NH (v n) H-> b v t s n (NH H)'
        
    def forward(self, x_zooms, mask_zooms={}, emb=None, sample_configs={}):        

        zoom_att = self.grid_layer.zoom

        q = []
        kv = []
        mask = []
        n_p = []
        x_zooms_emb = {}

        if self.composed_residual:
            x_zooms_res = self.composed_residual_layer(x_zooms, emb, sample_configs)

        for zoom, emb_layer in self.qkv_affine_layers.items():
            if int(zoom) in x_zooms.keys():
                x_zooms_emb[int(zoom)] = emb_layer(x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])
            else:
                x_zooms_emb[int(zoom)] = emb_layer(emb, sample_configs=sample_configs[int(zoom)])
        
        for zoom, q_layer in self.q_projection_layers.items():

            q_ = self.q_affine_layers[zoom](x_zooms_emb[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

            q_ = get_matching_time_patch(q_, int(zoom), self.max_zoom, sample_configs)

            b,v,t,N,f = q_.shape
            n = 4**(int(zoom)-zoom_att)

            if isinstance(q_layer, EmbIdLayer):
                q_ = q_layer(emb=emb, sample_configs=sample_configs[int(zoom)])
            else:    
                q_ = q_layer(q_, emb=emb, sample_configs=sample_configs[int(zoom)])

            q_ = self.q_headgate_layers[zoom](q_, emb=emb, sample_configs=sample_configs[int(zoom)])

            q_ = rearrange(q_, self.pattern, b=b, v=v, n=n, NH=self.num_heads)
            
            n_p.append(n)
            q.append(q_)

        for zoom, kv_layer in self.kv_projection_layers.items():

            kv_ = self.kv_affine_layers[zoom](x_zooms_emb[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

            kv_ = kv_layer(kv_, emb=emb, sample_configs=sample_configs[int(zoom)])

            kv_ = self.kv_headgate_layers[zoom](kv_, emb=emb, sample_configs=sample_configs[int(zoom)])

            kv_, mask_ = self.grid_layers[str(self.att_zoom)].get_nh(kv_, **sample_configs[int(zoom)], with_nh=self.with_nh, mask=mask_zooms[int(zoom)] if len(mask_zooms)>0 else None)
            kv_ = get_matching_time_patch(kv_, int(zoom), self.max_zoom, sample_configs)
            
            s = kv_.shape[3]

            kv_ = self.kv_compression_layers[zoom](kv_, emb=emb, sample_configs=sample_configs)
            
            kv_ = kv_.view(*kv_.shape[:4],-1, kv_.shape[-1]) 
            
            if not isinstance(self.kv_compression_layers[zoom],IdentityLayer):
                mask_ = torch.zeros_like(kv_[...,[0]], dtype=torch.bool, device=kv_.device)
            
            elif mask_ is not None:
                mask_ = get_matching_time_patch(mask_, int(zoom), self.max_zoom, sample_configs)

            if zoom in self.CA_layers.keys():
                kv_, mask_ = self.CA_layers[zoom](kv_, emb=emb, mask=mask_, sample_configs=sample_configs)

            kv_ = rearrange(kv_, self.kv_pattern, b=b, v=v, s=s, NH=self.num_heads)

            if mask_ is not None:
                mask_ = rearrange(mask_, self.mask_pattern, b=b, v=v, s=s)

            kv.append(kv_)
            mask.append(mask_)

        K, V = torch.concat(kv, dim=-2).chunk(2, dim=-1)
        Q = torch.concat(q, dim=-2)

        if mask[0] is not None:
            mask = torch.concat(mask, dim=-1)

            all_masked = mask.all(dim=-1,keepdim=True)
            mask[all_masked.expand_as(mask)] = False
        else:
            mask = None

        Q = safe_scaled_dot_product_attention(Q, K, V, mask=mask)

        Q = rearrange(Q, self.reverse, b=b, t=t, s=s, v=v)
        
        Q = Q.split(n_p, dim=-2)

        for k, zoom in enumerate(self.mlp_affine_layers.keys()):
            
            q = self.out_attention_layers[zoom](Q[k].reshape(*Q[k].shape[:3],-1,Q[k].shape[-1]), emb=emb, sample_configs=sample_configs)

            if self.composed_residual:
                x_zooms[int(zoom)] = x_zooms_res[int(zoom)] + q * self.gammas[zoom]
            else:
                if int(zoom) in x_zooms.keys():
                    x_zooms[int(zoom)] = x_zooms[int(zoom)] + self.gammas[zoom] * insert_matching_time_patch(x_zooms[int(zoom)], q, int(zoom), self.max_zoom, sample_configs)
                else:
                    x_zooms[int(zoom)] = q * self.gammas[zoom]
        
            x_mlp = self.mlp_affine_layers[zoom](x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

            x_zooms[int(zoom)] = x_zooms[int(zoom)] + self.mlps[zoom](x_mlp, emb=emb, sample_configs=sample_configs[int(zoom)])

            x_zooms[int(zoom)] = self.out_layers[zoom](x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])


        return x_zooms


class ComposedResidual(nn.Module):
    def __init__(self, features, embedders, in_zooms, out_zooms, layer_confs, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.lin_layers = nn.ModuleDict()
        self.in_zooms = in_zooms
        self.out_zooms = out_zooms

        for out_zoom in out_zooms:
            linear_layer = LinEmbLayer(features*len(in_zooms), features, layer_confs=layer_confs, embedder=embedders[str(out_zoom)])
            self.lin_layers[str(out_zoom)] = linear_layer

    def forward(self, x_zooms, emb=None, sample_configs={}):
        x_res_zooms = {}
        for out_zoom in self.out_zooms:
            res_tensor = []
            for in_zoom in x_zooms.keys():
                b, v, t, N, f = x_zooms[in_zoom].shape
                if in_zoom > out_zoom:
                    res = x_zooms[in_zoom].view(b, v, t, -1, 4**(in_zoom-out_zoom), f).mean(-2)
                elif in_zoom < out_zoom:
                    res = x_zooms[in_zoom].repeat(1, 1, 1, 4**(out_zoom - in_zoom), 1)
                else:
                    res = x_zooms[in_zoom]
                res_tensor.append(res)
            res_tensor = torch.cat(res_tensor, dim=-1)
            x_res_zooms[out_zoom] = self.lin_layers[str(out_zoom)](res_tensor, emb=emb, sample_configs=sample_configs[int(out_zoom)])
        return x_res_zooms

class MGCompressionAttention(nn.Module):
  
    def __init__(self, 
                 features: int,
                 embedder_q: EmbedderSequential=None,
                 with_nh = True,
                 n_heads = 16,
                 head_gate=False,
                 head_gate_scale_limit=1.0,
                 qkv_emb_projection_settings = {},
                 layer_confs: Dict = {},
                 layer_confs_emb= {}) -> None: 
        
        super().__init__()
        
        self.n_heads = n_heads
        assert 'MGEmbedder' in embedder_q.embedders.keys(), 'MGEmbedder is required for compression attention'
        self.emb_layerq = embedder_q.embedders['MGEmbedder']
        self.zoom = embedder_q.embedders['MGEmbedder'].zoom


        self.emb_proj =  get_layer([embedder_q.get_out_channels], [features], layer_confs=layer_confs_emb)

        if head_gate:
            self.headgate_layer = HeadGate(self.n_heads, embedder=embedder_q, scale_limit=head_gate_scale_limit)
        else:
            self.headgate_layer = IdentityLayer()

        #self.attention = SelfAttention(features, features, num_heads=features//n_head_channels,qkv_proj=False,cross=True)

        self.out_proj = get_layer(features, features*2, layer_confs=layer_confs, bias=True)

        with_nh = with_nh
        
        self.pattern = 'b v t s n (NH H) -> (b v t s) NH n H'
        self.kv_pattern = self.pattern
        self.mask_pattern = 'b v t s n 1 -> (b v t s) 1 1 n'
        self.reverse = '(b v t s) NH n H -> b v t s n (NH H)'


    def forward(self, x: torch.Tensor, mask=None, emb=None, sample_configs={}):        
        
        sample_cfg = get_sample_configs(sample_configs, self.zoom)

        q = self.emb_layerq(emb['MGEmbedder'], sample_configs=sample_cfg)
        q = self.emb_proj(q, emb=emb, sample_configs=sample_cfg)

        q = self.headgate_layer(q, emb=emb, sample_configs=sample_cfg)

        q = q.reshape(*q.shape[:3],x.shape[3],-1,q.shape[-1])

        if mask is not None:
            mask_out = mask.min(dim=-2, keepdim=True).values
        else:
            mask_out = torch.zeros_like(q[...,[0]], dtype=torch.bool, device=q.device)
            mask = None


        b, v, t, s, n, c = q.shape

        Q = rearrange(q, self.pattern, NH=self.n_heads)
        x = rearrange(x, self.pattern, NH=self.n_heads)

        K, V = x.chunk(2,dim=-1)

        if mask is not None:
            all_masked = mask.all(dim=-2,keepdim=True)
            mask[all_masked.expand_as(mask)] = False
            mask_att = rearrange(mask, self.mask_pattern)

        else:
            mask_att = None

        att_out = safe_scaled_dot_product_attention(Q, K, V, mask=mask_att)

        att_out = rearrange(att_out, self.reverse, b=b, t=t, s=s, v=v)

        #if mask is not None:
        #    att_out.masked_fill_(all_masked, 0)

        shape = att_out.shape
        att_out = self.out_proj(att_out, emb=emb, sample_configs=sample_cfg)

        if mask_out is not None:
            mask_out = mask_out.expand(*shape[:-1],1)

        return att_out.view(*shape[:-1],-1), mask_out

class MultiFieldAttention(nn.Module):
  
    def __init__(self,
                 grid_layer_field: GridLayer,
                 grid_layer_att: GridLayer,
                 q_zooms: int,
                 kv_zooms: int,
                 mult: int=1,
                 att_dim= None,
                 dropout: float=0.0,
                 n_head_channels: int=32,
                 embedder: Dict[str, EmbedderSequential] = {},
                 residual_embedder: Dict[str, EmbedderSequential] = {},
                 with_nh_post_att = False,
                 with_nh_field_mlp = False,
                 with_nh_field = True,
                 with_nh_att = False,
                 with_var_att = True,
                 with_time_att = False,
                 time_seq_len = 1,
                 with_mlp_embedder = True,
                 residual_learned = False,
                 layer_confs: Dict = {},
                 layer_confs_emb= {},
                 update='shift',
                 double_skip=False,
                 layer_norm=True) -> None: 
        
        super().__init__()

        if grid_layer_att.zoom <= 0 and with_nh_att:
            with_nh_att = False
            self.global_att = True
        else:
            self.global_att = False

        self.double_skip = double_skip
        layer_confs_kv = copy.deepcopy(layer_confs)
        layer_confs_kv['bias'] = True

        self.n_head_channels = n_head_channels
        self.grid_layer_field = grid_layer_field
        self.grid_layer_att = grid_layer_att

        self.with_nh_field = with_nh_field
        self.with_nh_att = with_nh_att
        self.with_nh_field_mlp = with_nh_field_mlp
        self.with_nh_post_att = with_nh_post_att

        self.n_groups = layer_confs.get('n_groups',1)
        self.emb_layers = nn.ModuleDict()
        self.mlp_emb_layers = nn.ModuleDict()
        self.q_layers = nn.ModuleDict()
        self.kv_layers = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        self.out_layers = nn.ModuleDict()

        self.q_zooms = q_zooms
        self.kv_zooms = kv_zooms

        out_feat_fac = 1
        out_feat_fac = 2 if update == 'scale_shift' else out_feat_fac
        
        if len(self.q_zooms) == len(self.kv_zooms):
            self.self_att = True
            self.self_att = ((torch.tensor(q_zooms) - torch.tensor(kv_zooms)) == 0).all()
        else:
            self.self_att = False

        field_zoom = grid_layer_field.zoom

        self.q_projection_layers = nn.ModuleDict()
        self.kv_projection_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()

        self.n_channels_q = {}
        for q_zoom in q_zooms:
            self.n_channels_q[q_zoom] = max([4**(q_zoom - field_zoom),1])

        self.n_channels_kv = {}
        for kv_zoom in kv_zooms:
            n_channels_zoom = max([4**(kv_zoom - field_zoom),1]) * grid_layer_field.adjc.shape[-1] if self.with_nh_field else max([4**(kv_zoom - field_zoom),1])
            self.n_channels_kv[kv_zoom] = (n_channels_zoom)

        self.in_features_q = in_features_q = sum(self.n_channels_q.values())
        self.in_features_kv = in_features_kv = sum(self.n_channels_kv.values())

        out_features_field = in_features_q
        att_dim = out_features_field if att_dim is None else att_dim

        self.emb_layer_q = LinEmbLayer(in_features_q, in_features_q, layer_confs=layer_confs, identity_if_equal=True, embedder=embedder, layer_norm=layer_norm, layer_confs_emb=layer_confs_emb)

        if not self.self_att:
            self.emb_layer_kv = LinEmbLayer(in_features_kv, in_features_kv, layer_confs=layer_confs, identity_if_equal=True, embedder=embedder, layer_norm=layer_norm, layer_confs_emb=layer_confs_emb)


        in_features_kv = [in_features_kv]
        in_features_q = [in_features_q]
        out_dim_q = [att_dim] 
        out_dim_kv = [2*att_dim]

        self.att_dim = att_dim
 
        self.q_projection_layer = get_layer(in_features_q, out_dim_q, layer_confs=layer_confs, bias=False)
        self.kv_projection_layer = get_layer(in_features_kv, out_dim_kv, layer_confs=layer_confs, bias=True)

        in_features_out = att_dim * grid_layer_field.adjc.shape[-1] if self.with_nh_post_att else att_dim
        self.out_layer_att = get_layer(in_features_out, out_features_field * out_feat_fac, layer_confs=layer_confs) if att_dim!=out_features_field else IdentityLayer() 

        self.mlp_emb_layer = LinEmbLayer(out_features_field, out_features_field, layer_confs=layer_confs, identity_if_equal=True, embedder=embedder if with_mlp_embedder else None, layer_norm=layer_norm, layer_confs_emb=layer_confs_emb)
       
        in_features_mlp = out_features_field * grid_layer_field.adjc.shape[-1] if self.with_nh_field_mlp else out_features_field

        
        self.dropout_att = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.dropout_mlp = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()

        residual_w_embed = False
        self.scale_shift = False

        if update == 'scale_shift':
            self.scale_shift = True
            self.gamma_scale = nn.Parameter(torch.ones(out_features_field)*1e-6, requires_grad=True)

        elif residual_learned and residual_embedder is None:
            self.gamma_res = nn.Parameter(torch.ones(out_features_field), requires_grad=True)
            self.gamma_res_mlp = nn.Parameter(torch.ones(out_features_field), requires_grad=True) 
        
        elif residual_embedder is not None:
            self.residual_embedder = residual_embedder
            self.embedding_layer_res = get_layer([residual_embedder.get_out_channels], [out_features_field*2], layer_confs=layer_confs_emb)
            self.embedding_layer_res_mlp = get_layer([residual_embedder.get_out_channels], [out_features_field*2], layer_confs=layer_confs_emb)
            self.activation = nn.Sigmoid()

            residual_learned = False
            residual_w_embed = True
        
            

        self.gamma = nn.Parameter(torch.ones(out_features_field)*1e-6, requires_grad=True)
        self.mlp = MLP_fac(in_features_mlp, out_features_field * out_feat_fac, mult, dropout, layer_confs=layer_confs, gamma=True) 


        self.residual_w_embed = residual_w_embed
        self.residual_learned = residual_learned

        self.pattern_channel = 'b v t N n f ->  b (f v) t N n'
        self.pattern_channel_reverse = 'b (f v) t N n ->  b v t (N n) f'

        self.with_time_att = with_time_att
        self.time_seq_len = time_seq_len

        if with_var_att:
            self.att_pattern = 'b fv t N NA (NH H)-> (b N t) NH (fv NA) H'
            self.mask_pattern = 'b v t N NA f-> (b N t) 1 1 (f v NA)'
            self.att_pattern_reverse = '(b N t) NH (fv NA) H -> b fv t (N NA) (NH H)'
        elif with_time_att:
            assert with_var_att == False
            self.att_pattern = 'b fv (tb t) N NA (NH H)-> (b fv N tb) NH (t NA) H'
            self.mask_pattern = 'b v (tb t) N NA f-> (b v N tb f) 1 1 (t NA)'
            self.att_pattern_reverse = '(b fv N tb) NH (t NA) H -> b fv (tb t) (N NA) (NH H)'
        else:
            self.att_pattern = 'b fv t N NA (NH H)-> (b fv N t) NH (NA) H'
            self.mask_pattern = 'b v t N NA f-> (b v N t f) 1 1 (NA)'
            self.att_pattern_reverse = '(b fv N t) NH (NA) H -> b fv t (N NA) (NH H)'  


    def forward(self, x_zooms, mask_zooms=[], emb=None, sample_configs={}, x_zoom_res=None):        

        zoom_field = self.grid_layer_field.zoom
        zoom_att = self.grid_layer_att.zoom

        b,nv,t,_,f = x_zooms[self.q_zooms[0]].shape

        q = combine_zooms(x_zooms, zoom_field, self.q_zooms)
        q = rearrange(q, self.pattern_channel)

        if x_zoom_res is None:
            x_res = q 
        else: 
            x_res = combine_zooms(x_zoom_res, zoom_field, self.q_zooms)
            x_res = rearrange(x_res, self.pattern_channel)

        q = self.emb_layer_q(q, emb=emb, sample_configs=sample_configs[zoom_field])
        
        if not self.self_att:
            kv = combine_zooms(x_zooms, zoom_field, self.kv_zooms)
            kv = rearrange(kv, self.pattern_channel)
            kv = self.emb_layer_kv(kv, emb=emb, sample_configs=sample_configs[zoom_field])
        else:
            kv = q
        
        q = self.q_projection_layer(q, emb=emb, sample_configs=sample_configs[zoom_field])

        q = q.reshape(*q.shape[:3], -1, self.att_dim)
        
        if self.global_att:
            q = q.reshape(*q.shape[:3], 1, -1, q.shape[-1])
        else:
            q = q.reshape(*q.shape[:3], -1, 4**(zoom_field-zoom_att), q.shape[-1])

        if self.with_nh_field:
            kv, _ = self.grid_layer_field.get_nh(kv, **sample_configs[zoom_field], with_nh=self.with_nh_field, mask=None)

        kv = self.kv_projection_layer(kv, emb=emb, sample_configs=sample_configs[zoom_field])

        kv = kv.reshape(*kv.shape[:3], -1, 2 * self.att_dim)

        mask = mask_zooms[zoom_field] if zoom_field in mask_zooms.keys() else None
    #    mask = None
        if mask is not None:
            mask = mask.expand_as(x_zooms[zoom_field])

        if self.with_nh_att:
            kv, mask = self.grid_layer_att.get_nh(kv, **sample_configs[zoom_att], with_nh=self.with_nh_att, mask=mask)
        
        elif self.global_att:
            kv = kv.reshape(*kv.shape[:3], 1, -1, kv.shape[-1])
            mask = mask.view(*mask.shape[:3], 1, -1, mask.shape[-1]) if mask is not None else None
        else:
            kv = kv.reshape(*kv.shape[:3], -1, 4**(zoom_field-zoom_att), kv.shape[-1])
            mask = mask.view(*mask.shape[:3], -1, 4**(zoom_field-zoom_att), mask.shape[-1]) if mask is not None else None


        b, fv, t, N, NA, C = q.shape
        if self.with_time_att:
            t = self.time_seq_len

        k,v = kv.chunk(2, dim=-1)
        q = rearrange(q, self.att_pattern, H=self.n_head_channels, t=t)
        k = rearrange(k, self.att_pattern, H=self.n_head_channels, t=t)
        v = rearrange(v, self.att_pattern, H=self.n_head_channels, t=t)

        mask = rearrange(mask, self.mask_pattern, t=t) if mask is not None else None

        att_out = safe_scaled_dot_product_attention(q, k, v, mask=mask)

        att_out = rearrange(att_out, self.att_pattern_reverse, b=b, fv=fv, t=t, N=N, NA=NA)

        if self.with_nh_post_att:
            att_out, _ = self.grid_layer_field.get_nh(att_out, **sample_configs[zoom_field], with_nh=True, mask=None)

        att_out = self.out_layer_att(att_out, emb=emb, sample_configs=sample_configs)

        if self.residual_learned:
            x = self.gamma_res * x_res + self.gamma * self.dropout_att(att_out)
        
        elif self.residual_w_embed:
            emb_ = self.residual_embedder(emb, sample_configs[zoom_field]).expand(*x_res.shape[:3],-1,-1)
            scale_res, scale = self.embedding_layer_res(emb_, sample_configs=sample_configs, emb=emb).chunk(2, dim=-1)
            x = self.activation(scale_res) * x_res + self.gamma * self.activation(scale) * self.dropout_att(att_out)

        elif self.scale_shift:
            scale, shift = self.dropout_att(att_out).chunk(2,dim=-1)
            x = x_res * (1 + self.gamma_scale * scale) + self.gamma * shift

        else:
            x = x_res + self.gamma * self.dropout_att(att_out)

        x_mlp = self.mlp_emb_layer(x, emb=emb, sample_configs=sample_configs[int(zoom_field)])

        if self.with_nh_field_mlp:
            x_mlp, _ = self.grid_layer_field.get_nh(x_mlp, **sample_configs[zoom_field], with_nh=True, mask=None)

        x_mlp = self.dropout_mlp(self.mlp(x_mlp, emb=emb, sample_configs=sample_configs[int(zoom_field)]))

        if self.double_skip:
            x = x_res

        if self.residual_learned:
            x = self.gamma_res_mlp * x + x_mlp
        
        elif self.residual_w_embed:
            scale_res, scale = self.embedding_layer_res_mlp(emb_, sample_configs=sample_configs, emb=emb).chunk(2, dim=-1)
            x = self.activation(scale_res) * x + self.activation(scale) * x_mlp

        elif self.scale_shift:
            scale, shift = self.dropout_mlp(x_mlp).chunk(2,dim=-1)
            x = x * (1 + scale) + shift

        else:
            x = x + x_mlp

        x = x.split(tuple(self.n_channels_q.values()), dim=-1)

        for k, (zoom, n) in enumerate(self.n_channels_q.items()):
            x_zooms[zoom] = rearrange(x[k], self.pattern_channel_reverse, n=n, f=f,v=nv)

        return x_zooms
    
