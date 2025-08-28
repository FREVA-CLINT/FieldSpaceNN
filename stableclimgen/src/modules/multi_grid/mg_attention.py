from typing import List,Dict
from einops import rearrange
import copy

import torch
import torch.nn as nn

from ..base import get_layer, IdentityLayer, LinEmbLayer, MLP_fac
from ..factorization import get_cp_tensor,get_cp_tensors, get_cp_equation

from ...modules.grids.grid_layer import GridLayer
from ...modules.transformer.transformer_base import SelfAttention,safe_scaled_dot_product_attention

from ...modules.embedding.embedder import EmbedderSequential,MGEmbedder
from ..grids.grid_utils import get_matching_time_patch, insert_matching_time_patch, get_sample_configs

    

class MultiZoomSelfAttention(nn.Module):
  
    def __init__(self, 
                 grid_layers: GridLayer,
                 att_zoom: int,
                 in_features: int,
                 out_features: int,
                 q_zooms: int,
                 kv_zooms: int,
                 mult: int=1,
                 dropout: float=0.0,
                 num_heads: int=1,
                 att_dim=None,
                 n_head_channels: int=None, 
                 embedders: Dict[str, EmbedderSequential] = {},
                 compression_dims_kv: Dict[int, int]= {},
                 compression_dims_q: Dict[int, int]= {},
                 compression_zooms: Dict[int,int] = {},
                 with_nh = True,
                 var_att = False,
                 common_kv = False,
                 common_q = False,
                 common_out = False,
                 layer_confs: Dict = {},
                 layer_confs_emb= {}) -> None: 
        
        super().__init__()
        
        layer_confs_kv = copy.deepcopy(layer_confs)
        layer_confs_kv['bias'] = True
        
        self.grid_layers = grid_layers
        self.att_zoom = att_zoom
        grid_layer = grid_layers[str(att_zoom)]

        self.with_nh = with_nh

        self.emb_layers = nn.ModuleDict()
        self.mlp_emb_layers = nn.ModuleDict()
        self.q_layers = nn.ModuleDict()
        self.kv_layers = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
       # self.res_layers_mlp = nn.ModuleDict()
        self.out_layers = nn.ModuleDict()
       # self.gammas = nn.ParameterDict()

        att_dim = out_features if att_dim is None else att_dim

        self.CA_layers = nn.ModuleDict()
        for in_zoom, out_zoom in compression_zooms.items():

            self.CA_layers[str(in_zoom)] = MGCompressionAttention(
                                   features=att_dim,
                                   embedder_q=embedders[str(out_zoom)],
                                   n_head_channels=n_head_channels,
                                   var_att=var_att,
                                   layer_confs=layer_confs,
                                   layer_confs_emb=layer_confs_emb
                                   )

        for in_zoom in embedders.keys():
            if int(in_zoom) in q_zooms+kv_zooms: 
                self.emb_layers[in_zoom] = LinEmbLayer(in_features, [att_dim], layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[in_zoom], layer_norm=True, layer_confs_emb=layer_confs_emb) 

        self.res_layers = nn.ModuleDict()
        for q_zoom in q_zooms:
            in_f = out_f = []
            if q_zoom in compression_dims_q.keys():
                in_f = [4 for _ in compression_dims_q[q_zoom]]
                if len(in_f) == q_zoom +1:
                    in_f[0] = 12
                out_f = [4 if d==-1 else d for d in compression_dims_q[q_zoom]]

            in_features_q = [*in_f, att_dim]
            out_features_q = [*out_f, att_dim]

            self.mlp_emb_layers[str(q_zoom)] = LinEmbLayer(in_features, out_features, layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[str(q_zoom)], layer_norm=True, layer_confs_emb=layer_confs_emb)
            self.mlps[str(q_zoom)] = MLP_fac(out_features, out_features, mult, dropout, layer_confs=layer_confs, gamma=True) 

            self.q_layers[str(q_zoom)] = get_layer(in_features_q, out_features_q, layer_confs=layer_confs) if not common_q else IdentityLayer()
           # self.res_layers_mlp[str(q_zoom)] = get_layer(in_features, out_features, layer_confs=layer_confs) if not common_out and in_features!=out_features else IdentityLayer()
            
            self.out_layers[str(q_zoom)] = get_layer(in_features, out_features, layer_confs=layer_confs) if not common_out and in_features!=out_features else IdentityLayer()


        self.omit_mask_zooms = []
        for kv_zoom in kv_zooms:
            in_f = out_f = []

            if kv_zoom in compression_dims_kv.keys():
                in_f = [4 for _ in compression_dims_kv[kv_zoom]]
                out_f = [4 if d==-1 else d for d in compression_dims_kv[kv_zoom]]

                self.omit_mask_zooms.append(kv_zoom)

            if with_nh:
                nh_dim = grid_layer.adjc.shape[1] if with_nh else None
                in_features_kv = [nh_dim, *in_f, att_dim]
                out_features_kv = [nh_dim, *out_f, 2*att_dim]
                layer_confs_ = layer_confs.copy()
                layer_confs_['skip_dims'] = [True, *[False]*len(out_f), False]
            else:
                layer_confs_ = layer_confs
                in_features_kv = [*in_f, att_dim]
                out_features_kv = [*out_f, 2*att_dim]

            self.kv_layers[str(kv_zoom)] = get_layer(in_features_kv, out_features_kv, layer_confs=layer_confs_,bias=True) if not common_kv else IdentityLayer()

   
        self.common_q_layer = get_layer(in_features, in_features, bias=False, layer_confs=layer_confs) if common_q else IdentityLayer()
        self.common_kv_layer = get_layer(in_features, [2*in_features], bias=True, layer_confs=layer_confs) if common_kv else IdentityLayer()

        num_heads = num_heads if not n_head_channels else in_features // n_head_channels
        self.attention = SelfAttention(att_dim, in_features, num_heads=num_heads, dropout=dropout, cross=True, qkv_proj=False)

        self.grid_layer = grid_layer
        self.max_zoom = int(max(list(embedders.keys())))

        assert out_features==in_features,"Module does not support differen in and out features"

        if not var_att:
            self.pattern = 'b v t s n c -> (b v t s) (n) c'
            self.kv_pattern = 'b v t s n kv c -> (b v t s) (n) (kv c)'
            self.nh_mask_pattern = 'b v t s n 1 -> (b v t s) 1 1 n'
            self.reverse_pattern = '(b v t s) n c -> b v t s n c'
        else:
            self.pattern = 'b v t s n c -> (b t s) (v n) c'
            self.kv_pattern = 'b v t s n kv c -> (b t s) (v n) (kv c)'
            self.nh_mask_pattern = 'b v t s n 1 -> (b t s) 1 1 (v n)'
            self.reverse_pattern = '(b t s) (v n) c -> b v t s n c'

    def forward(self, x_zooms, mask_zooms=None, emb=None, sample_configs={}):        

        zoom_att = self.grid_layer.zoom

        q = []
        kv = []
        mask = []
        n_p = []
        x_zooms_emb = {}

        for zoom, emb_layer in self.emb_layers.items():
            x_zooms_emb[int(zoom)] = emb_layer(x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

        
        for zoom, q_layer in self.q_layers.items():
            q_ = get_matching_time_patch(x_zooms_emb[int(zoom)], int(zoom), self.max_zoom, sample_configs)
            n = q_.shape[3] // 4**(int(zoom)-zoom_att)

            q_ = q_layer(q_, emb=emb, sample_configs=sample_configs[int(zoom)])
            q_ = q_.view(*q_.shape[:3], n, -1 ,q_.shape[-1])
            n_p.append(q_.shape[-2])
            q.append(q_)

        for zoom, kv_layer in self.kv_layers.items():
            #operating_zoom = self.att_zoom if zoom not in self.CA_layers.keys() else self.CA_layers[zoom].zoom
            kv_, mask_ = self.grid_layers[str(self.att_zoom)].get_nh(x_zooms_emb[int(zoom)], **sample_configs[int(zoom)], with_nh=self.with_nh, mask=mask_zooms[int(zoom)] if len(mask_zooms)>0 else None)
            kv_ = get_matching_time_patch(kv_, int(zoom), self.max_zoom, sample_configs)
            
            n = kv_.shape[3]
            kv_ = kv_layer(kv_, emb=emb, sample_configs=sample_configs[int(zoom)])
            kv_ = kv_.reshape(*kv_.shape[:3],n,-1,kv_.shape[-1])
            
            if mask_ is not None and int(zoom) not in self.omit_mask_zooms:
                mask_ = get_matching_time_patch(mask_, int(zoom), self.max_zoom, sample_configs)
            elif zoom not in self.CA_layers.keys():
                mask_ = torch.zeros_like(kv_[...,[0]], dtype=torch.bool, device=kv_.device)

            if zoom in self.CA_layers.keys():
                kv_ = self.CA_layers[zoom](kv_, emb=emb, mask=mask_, sample_configs=sample_configs)[0]
                mask_ = torch.zeros_like(kv_[...,[0]], dtype=torch.bool, device=kv_.device)

            kv.append(kv_)
            mask.append(mask_)

        kv = torch.concat(kv, dim=-2)
        q = torch.concat(q, dim=-2)

        q = self.common_q_layer(q, sample_configs=sample_configs[int(zoom)], emb=emb)
        kv_shape = kv.shape[:-1]
        kv = self.common_kv_layer(kv, sample_configs=sample_configs[int(zoom)], emb=emb).view(*kv_shape,2,-1)

        if mask[0] is not None:
            mask = torch.concat(mask, dim=-2)
            mask = rearrange(mask, self.nh_mask_pattern)

            all_masked = mask.all(dim=-1,keepdim=True)
            mask[all_masked.expand_as(mask)] = False
        else:
            mask = None

        b, v, t, s, n, c = q.shape

        q = rearrange(q, self.pattern)
        kv = rearrange(kv, self.kv_pattern)

        
        q = self.attention(q, kv, mask=mask)

        if mask is not None:
            q.masked_fill_(all_masked[...,0],0)

        q = rearrange(q, self.reverse_pattern, b=b, t=t, s=s, n=n, v=v) 

        q = q.split(n_p, dim=-2)

        for k, zoom in enumerate(self.q_layers.keys()):

            x_zooms[int(zoom)] = insert_matching_time_patch(x_zooms[int(zoom)], q[k].reshape(*q[k].shape[:3],-1,q[k].shape[-1]), int(zoom), self.max_zoom, sample_configs, add=True)
        
            x_mlp = self.mlp_emb_layers[zoom](x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

            x_zooms[int(zoom)] = x_zooms[int(zoom)] + self.mlps[zoom](x_mlp, emb=emb, sample_configs=sample_configs[int(zoom)])

            x_zooms[int(zoom)] = self.out_layers[zoom](x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])


        return x_zooms


class MGCompressionAttention(nn.Module):
  
    def __init__(self, 
                 features: int,
                 embedder_q: EmbedderSequential=None,
                 with_nh = True,
                 var_att = False,
                 n_head_channels = 16,
                 layer_confs: Dict = {},
                 layer_confs_emb= {}) -> None: 
        
        super().__init__()

        assert 'MGEmbedder' in embedder_q.embedders.keys(), 'MGEmbedder is required for compression attention'
        self.emb_layerq = embedder_q.embedders['MGEmbedder']
        self.zoom = embedder_q.embedders['MGEmbedder'].zoom
        self.emb_proj =  get_layer([embedder_q.get_out_channels], [features], layer_confs=layer_confs_emb)

        self.attention = SelfAttention(features,features, num_heads=features//n_head_channels,qkv_proj=False,cross=True)

        self.out_proj = get_layer(features, features*2, layer_confs=layer_confs,bias=True)

        with_nh = with_nh

        if not var_att:
            self.pattern = 'b v t s n c -> (b v t s) (n) c'
            self.kv_pattern = 'b v t s n c -> (b v t s) (n) (c)'
            self.mask_pattern = 'b v t s n 1 -> (b v t s) 1 1 n'
            self.reverse_pattern = '(b v t s) n c -> b v t s n c'
        else:
            self.pattern = 'b v t s n c -> (b t s) (v n) c'
            self.kv_pattern = 'b v t s n c -> (b t s) (v n) (c)'
            self.mask_pattern = 'b v t s n 1 -> (b t s) 1 1 (v n)'
            self.reverse_pattern = '(b t s) (v n) c -> b v t s n c'

    def forward(self, x, mask=None, emb=None, sample_configs={}):        
        
        sample_cfg = get_sample_configs(sample_configs, self.zoom)

        q = self.emb_layerq(emb['MGEmbedder'], sample_configs=sample_cfg)
        q = self.emb_proj(q, emb=emb, sample_configs=sample_cfg)
        q = q.reshape(*q.shape[:3],x.shape[3],-1,q.shape[-1])

        if mask is not None:
            mask_out = mask.min(dim=-2, keepdim=True).values
            mask = rearrange(mask, self.mask_pattern)
        else:
            mask_out = torch.zeros_like(q[...,[0]], dtype=torch.bool, device=q.device)
            mask = None

        b, v, t, s, n, c = q.shape

        q = rearrange(q, self.pattern)
        x = rearrange(x, self.kv_pattern)

        if mask is not None:
            all_masked = mask.all(dim=-1,keepdim=True)
            mask[all_masked.expand_as(mask)] = False

        att_out = self.attention(q, x, mask=mask)

        if mask is not None:
            all_masked = mask.all(dim=-1,keepdim=True)
            mask[all_masked.expand_as(mask)] = False
            att_out.masked_fill_(all_masked[...,0],0)

        att_out = rearrange(att_out, self.reverse_pattern, b=b, t=t, s=s, n=n, v=v, c=c) 

        shape = att_out.shape
        att_out = self.out_proj(att_out, emb=emb, sample_configs=sample_cfg)

        return att_out.view(*shape[:-1],-1), mask_out


class MultiZoomFieldAttention(nn.Module):
  
    def __init__(self, 
                 grid_layer: GridLayer,
                 in_features: int,
                 out_features: int,
                 q_zooms: int,
                 kv_zooms: int,
                 mult: int=1,
                 dropout: float=0.0,
                 num_heads: int=1,
                 rank = 16,
                 contract_zooms = True,
                 contract_channels = True,
                 att_dim=None,
                 n_head_channels: int=32, 
                 share_zoom_proj = False,
                 share_zoom_proj_qkv = True,
                 embedders: Dict[str, EmbedderSequential] = {},
                 with_nh = True,
                 var_att = False,
                 layer_confs: Dict = {},
                 layer_confs_emb= {}) -> None: 
        
        super().__init__()
        
        layer_confs_kv = copy.deepcopy(layer_confs)
        layer_confs_kv['bias'] = True

        self.with_nh = with_nh

        self.n_groups = layer_confs.get('n_groups',1)
        self.emb_layers = nn.ModuleDict()
        self.mlp_emb_layers = nn.ModuleDict()
        self.q_layers = nn.ModuleDict()
        self.kv_layers = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        self.out_layers = nn.ModuleDict()
       # self.gammas = nn.ParameterDict()

        att_dim = out_features if att_dim is None else att_dim

        att_zoom = grid_layer.zoom

        zooms = [zoom for zoom in range(att_zoom+1, max(q_zooms + kv_zooms)+1)]
        zoom_diff_att = (max(q_zooms + kv_zooms)-att_zoom)

        for in_zoom in embedders.keys():
            self.emb_layers[in_zoom] = LinEmbLayer(in_features, [att_dim], layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[in_zoom], layer_norm=True, layer_confs_emb=layer_confs_emb)


        if share_zoom_proj:

            if share_zoom_proj_qkv:
                n_dim_max = [4]*zoom_diff_att
                self.zoom_projections_q =  get_cp_tensors(n_dim_max, n_dim_max, rank=rank, n_groups=self.n_groups, keys=zooms, contract=contract_zooms)
                self.zoom_projections_kv = self.zoom_projections_q
            else:
                n_dim_max = [4]*zoom_diff_att
                self.zoom_projections_q = get_cp_tensors(n_dim_max, n_dim_max, rank=rank, n_groups=self.n_groups, keys=[zoom for zoom in range(att_zoom+1, max(q_zooms)+1)], contract=contract_zooms)
                self.zoom_projections_kv = get_cp_tensors(n_dim_max, n_dim_max, rank=rank, n_groups=self.n_groups, keys=[zoom for zoom in range(att_zoom+1, max(kv_zooms)+1)], contract=contract_zooms)

        else:
            if share_zoom_proj_qkv:
                self.zoom_projections = nn.ParameterDict({str(zoom): get_cp_tensors([4]*(int(zoom) - att_zoom), [4]*(int(zoom) - att_zoom), rank=rank, n_groups=self.n_groups, contract=contract_zooms) for zoom in sorted(self.emb_layers.keys())})
                self.zoom_projections_kv = self.zoom_projections_q = self.zoom_projections
            else:
                self.zoom_projections_q = nn.ParameterDict({str(q_zoom): get_cp_tensors([4]*(int(q_zoom) - att_zoom), [4]*(int(q_zoom) - att_zoom), rank=rank, n_groups=self.n_groups, contract=contract_zooms) for q_zoom in sorted(q_zooms)})
                self.zoom_projections_kv = nn.ParameterDict({str(kv_zoom): get_cp_tensors([4]*(int(kv_zoom) - att_zoom), [4]*(int(kv_zoom) - att_zoom), rank=rank, n_groups=self.n_groups, contract=contract_zooms) for kv_zoom in sorted(kv_zooms)})

        if share_zoom_proj:
            self.get_tensors_fun = self.get_tensors_shared

        else:
            self.get_tensors_fun = self.get_tensors 

        self.channel_projections_q = nn.ParameterDict()
        self.gammas = nn.ParameterDict()
        self.contract_fun_q = None # contract fun_q
        self.eq_q = {} #q equation
        self.shapes_q = {} 

        for q_zoom in q_zooms:
            self.shapes_q[q_zoom] = (*[4]*(q_zoom - att_zoom), in_features)
            self.channel_projections_q[str(q_zoom)] =  get_cp_tensor(in_features, out_features, rank=rank, n_groups=self.n_groups, contract=contract_channels, std=1.) 
            self.eq_q[q_zoom] = get_cp_equation(1+q_zoom-att_zoom, n_groups=self.n_groups, contract_feats=contract_zooms, contract_channel=contract_channels)  

            self.mlp_emb_layers[str(q_zoom)] = LinEmbLayer(att_dim, att_dim, layer_confs=layer_confs, identity_if_equal=True, embedder=embedders[str(q_zoom)], layer_norm=True, layer_confs_emb=layer_confs_emb)
            self.mlps[str(q_zoom)] = MLP_fac(att_dim, att_dim, mult, dropout, layer_confs=layer_confs, gamma=True) 

            self.out_layers[str(q_zoom)] = get_layer(in_features, out_features, layer_confs=layer_confs) if in_features!=out_features else IdentityLayer() 
            self.gammas[str(q_zoom)] = nn.Parameter(torch.ones(in_features)*1e-6,requires_grad=True)


        self.channel_projections_k = nn.ParameterDict()
        self.channel_projections_v = nn.ParameterDict()
        
        self.contract_fun_kv = None # contract fun_q
        self.eq_kv = {} #q equation
        self.shapes_kv = {}
        self.shapes_kv_out = {}
        self.omit_mask_zooms = []

        for kv_zoom in kv_zooms:
            self.shapes_kv[kv_zoom] = (grid_layer.adjc.shape[1] ,*[4]*(kv_zoom - att_zoom), in_features) if with_nh else (*[4]*(kv_zoom - att_zoom), in_features)
            self.shapes_kv_out[kv_zoom] = (grid_layer.adjc.shape[1] ,-1, n_head_channels) if with_nh else (-1, n_head_channels)
            self.channel_projections_k[str(kv_zoom)] =  get_cp_tensor(in_features, out_features, rank=rank, n_groups=self.n_groups, contract=contract_channels, std=1.) 
            self.channel_projections_v[str(kv_zoom)] =  get_cp_tensor(in_features, out_features, rank=rank, n_groups=self.n_groups, contract=contract_channels, std=1.) 

            self.eq_kv[kv_zoom] = get_cp_equation(1+kv_zoom-att_zoom, n_groups=self.n_groups, contract_feats=contract_zooms, contract_channel=contract_channels, nh_dim=True)  
        

        self.n_head_channels = n_head_channels
        
        self.grid_layer = grid_layer
        self.max_zoom = int(max(list(embedders.keys())))

        assert out_features==in_features,"Module does not support differen in and out features"

        if not var_att:
            self.pattern = 'b v t n NH H -> (b v t) NH n H'
            self.kv_pattern = 'b v t n m NH H -> (b v t) NH (n m) H' if with_nh else self.pattern
            self.reverse = '(b v t) NH n H -> b v t n NH H'
        else:
            self.pattern = 'b v t n NH H -> (b t) NH (v n) H'
            self.kv_pattern = 'b v t n m NH H -> (b t) NH (v n m) H' if with_nh else self.pattern
            self.reverse = '(b t) NH (v n) H -> b v t n NH H'


    def get_tensors_shared(self, zoom_projections, zoom, emb={}):
     
        tensors = [self.get_tensor(tensor, emb=emb) for zoom_p, tensor in zoom_projections.items() if int(zoom_p)<=int(zoom)]

        return tensors


    def get_tensors(self,zoom_projections, zoom, emb={}):

        tensors = [self.get_tensor(tensor, emb=emb) for tensor in zoom_projections[zoom]]

        return tensors

    def get_tensor(self, tensor, emb):
        if self.n_groups==1:
            return tensor
        else:
            return tensor[emb['GroupEmbedder']]


    def forward(self, x_zooms, mask_zooms=None, emb=None, sample_configs={}):        

        zoom_att = self.grid_layer.zoom

        q = []
        k = []
        v = []
        n_p = []
    #    mask = []
        x_zooms_emb = {}

        for zoom, emb_layer in self.emb_layers.items():
            x_zooms_emb[int(zoom)] = emb_layer(x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

        for zoom, channel_proj in self.channel_projections_q.items():
            q_ = get_matching_time_patch(x_zooms_emb[int(zoom)], int(zoom), self.max_zoom, sample_configs)

            q_factors = self.get_tensors_fun(self.zoom_projections_q, zoom, emb=emb)
            q_ = q_.view(*q_.shape[:3],-1,*self.shapes_q[int(zoom)])

            q_ = torch.einsum(self.eq_q[int(zoom)], q_, *q_factors, self.get_tensor(channel_proj,emb=emb))

            q_ = q_.reshape(*q_.shape[:4],-1, self.n_head_channels)

            n_p.append(q_.shape[-2])
            q.append(q_)

        for zoom in self.channel_projections_k.keys():
            kv_, mask_ = self.grid_layer.get_nh(x_zooms_emb[int(zoom)], **sample_configs[int(zoom)], with_nh=self.with_nh, mask=mask_zooms[int(zoom)] if len(mask_zooms)>0 else None)
            kv_ = get_matching_time_patch(kv_, int(zoom), self.max_zoom, sample_configs)
            
            kv_factors = self.get_tensors_fun(self.zoom_projections_kv, zoom, emb=emb)
            kv_ = kv_.view(*kv_.shape[:3],-1,*self.shapes_kv[int(zoom)])

            k_ = torch.einsum(self.eq_kv[int(zoom)], kv_, *kv_factors, self.get_tensor(self.channel_projections_k[zoom],emb=emb))
            v_ = torch.einsum(self.eq_kv[int(zoom)], kv_, *kv_factors, self.get_tensor(self.channel_projections_v[zoom],emb=emb))

            k_ = k_.reshape(*kv_.shape[:4],*self.shapes_kv_out[int(zoom)])
            v_ = v_.reshape(*kv_.shape[:4],*self.shapes_kv_out[int(zoom)])

            k.append(k_)
            v.append(v_)

        q = torch.concat(q, dim=-2)
        k = torch.concat(k, dim=-2)
        v = torch.concat(v, dim=-2)

        b,va,t,n,N,H = q.shape
        q = rearrange(q, self.pattern, H=self.n_head_channels)
        k = rearrange(k, self.kv_pattern, H=self.n_head_channels)
        v = rearrange(v, self.kv_pattern, H=self.n_head_channels)

        q = safe_scaled_dot_product_attention(q, k, v)

        q = rearrange(q, self.reverse, b=b, v=va, t=t, NH=N, H=H)

        q = q.split(n_p, dim=-2)

        for k, zoom in enumerate(self.mlps.keys()):
            
            x_zooms[int(zoom)] = insert_matching_time_patch(x_zooms[int(zoom)], q[k].reshape(*q[k].shape[:3],-1,self.shapes_q[int(zoom)][-1]), int(zoom), self.max_zoom, sample_configs, add=True)
        
            x_mlp = self.mlp_emb_layers[zoom](x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])

            x_zooms[int(zoom)] = x_zooms[int(zoom)] + self.mlps[zoom](x_mlp, emb=emb, sample_configs=sample_configs[int(zoom)])

            x_zooms[int(zoom)] = self.out_layers[zoom](x_zooms[int(zoom)], emb=emb, sample_configs=sample_configs[int(zoom)])


        return x_zooms