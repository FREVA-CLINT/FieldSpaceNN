from typing import List,Dict
from einops import rearrange
import copy
from omegaconf import DictConfig, ListConfig

import math
import torch
import torch.nn as nn

from ..base import get_layer, LinEmbLayer, MLP_fac
from .mg_base import combine_zooms,refine_zoom,coarsen_zoom, Tokenizer

from ..grids.grid_layer import GridLayer
from ..transformer.transformer_base import safe_scaled_dot_product_attention

from ..embedding.embedder import EmbedderSequential
from ...modules.embedding.embedder import get_embedder

from ..grids.grid_utils import get_matching_time_patch, insert_matching_time_patch, get_sample_configs


class FieldAttentionConfig:
    def __init__(self, 
                 token_zoom: int,
                 q_zooms = -1,
                 kv_zooms = -1,
                 token_len_td: List = [1, 1],
                 token_overlap_std: List = [0, 0, 0],
                 seq_zoom: int = -1,
                 seq_len_td: List = [-1,-1],
                 seq_nh_std: List = [False, False, False],
                 mlp_token_overlap_td: List = [False, False, False],
                 with_var_att= False,
                 with_time_att= False,
                 with_depth_att = False,
                 time_seq_len = 1,
                 shift= None,
                 rev_shift= True,
                 multi_shift= False,
                 update = 'shift',
                 refine_zooms = {},
                 ranks_std = [None, None, None],
                 separate_mlp_norm=False,
                 **kwargs):

        inputs = copy.deepcopy(locals())
        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)



def add_depth_overlap_from_neighbor_patches(
    x: torch.Tensor,
    overlap: int = 1,
    pad_mode: str = "zeros",  

) -> torch.Tensor:
  
    o = overlap
    if o == 0:
        return x

    b, v, T, N, D, t, n, d, f = x.shape
    assert o <= d, f"overlap={o} must be <= d={d}"

    out = x.new_empty(b, v, T, N, D, t, n, d + 2 * o, f)

    # center
    out[..., o:o + d, :] = x

    if D > 1:
        out[:, :, :, :, 1:, :, :, :o] = x[:, :, :, :, :-1, :, :, d - o : d]
        out[:, :, :, :, :-1, :, :, o + d :] = x[:, :, :, :, 1:, :, :, :o]

    # boundaries
    if pad_mode == "zeros":
        out[:, :, :, :, 0,  :, :, :o] = 0
        out[:, :, :, :, -1, :, :, o + d :] = 0

    elif pad_mode == "edge":
        left_edge  = x[:, :, :, :, 0,  :, :, :1].expand(b, v, T, N, t, n, o, f)
        right_edge = x[:, :, :, :, -1, :, :, -1:].expand(b, v, T, N, t, n, o, f)
        out[:, :, :, :, 0,  :, :, :o] = left_edge
        out[:, :, :, :, -1, :, :, o + d :] = right_edge

    else:
        raise ValueError("pad_mode must be 'zeros' or 'edge'")

    return out

def add_time_overlap_from_neighbor_patches(
    x: torch.Tensor,
    overlap: int = 1,
    pad_mode: str = "zeros",  

) -> torch.Tensor:
  
    o = overlap
    if o == 0:
        return x

    b, v, T, N, D, t, n, d, f = x.shape
    assert o <= t, f"overlap={o} must be <= t={t}"

    out = x.new_empty(b, v, T, N, D, t + 2 * o, n, d, f)

    # center
    out[..., o:o + t,:,:,:] = x

    if T > 1:
        out[:, :, 1:, :, :, :o] = x[:, :, :-1, :, :, t - o : t]
        out[:, :, :-1, :, :, o + t :] = x[:, :, 1:, :, :, :o]

    # boundaries
    if pad_mode == "zeros":
        out[:, :, 0, :, :,  :o] = 0
        out[:, :, -1, :, :, o + t :] = 0

    elif pad_mode == "edge":
        left_edge  = x[:, :, 0, :, :,  :1].expand(b, v, N, D, o, n, d, f)
        right_edge = x[:, :, -1, :, :, -1:].expand(b, v, N, D, o, n, d, f)
        out[:, :, 0, :, :,  :o] = left_edge
        out[:, :, -1, :, :, o + t :] = right_edge

    else:
        raise ValueError("pad_mode must be 'zeros' or 'edge'")

    return out



class FieldAttentionModule(nn.Module):
  
    def __init__(self, 
                grid_layers: Dict[str, GridLayer],
                in_zooms: List[int],
                out_zooms: List[int],
                q_zooms:List|int,
                kv_zooms:List|int,
                token_zoom: int,
                token_len_td: List = [1, 1],
                token_overlap_std: List = [0, 0, 0],
                seq_zoom: int = -1,
                seq_len_td: List = [-1,-1],
                seq_nh_std: List = [False, False, False],
                mlp_token_overlap_td: List = [False, False, False],
                with_var_att= False,
                use_mask = False,
                att_dim = None,
                ranks_std = [None,None,None],
                n_head_channels=16,
                refine_zooms= {},
                shift = None,
                rev_shift = True,
                multi_shift = False,
                dropout:float = 0,
                update='shift',
                separate_mlp_norm=False,
                embed_confs={},
                layer_confs={},
                layer_confs_emb={}
                 ) -> None:
        super().__init__()
        
        self.out_zooms = copy.deepcopy(out_zooms)
        in_zooms = copy.deepcopy(in_zooms)
        self.use_mask = use_mask

        if not isinstance(q_zooms, (List,ListConfig)) and (q_zooms == -1):
            q_zooms = in_zooms
        
        if not isinstance(kv_zooms,(List,ListConfig)) and (kv_zooms == -1):
            kv_zooms = in_zooms

        for k,zoom in enumerate(q_zooms):
            if zoom in refine_zooms.keys():
                q_zooms[k] = refine_zooms[zoom]
        
        for k,zoom in enumerate(kv_zooms):
            if zoom in refine_zooms.keys():
                kv_zooms[k] = refine_zooms[zoom]

        for k,zoom in enumerate(in_zooms):
            if zoom in refine_zooms.keys():
                in_zooms[k] = refine_zooms[zoom]

        for k, zoom in enumerate(kv_zooms):
            if zoom not in in_zooms:
                raise ValueError(f"Zoom level {zoom} at index {k} of kv_zooms not found in in_zooms")
        
        self.qkv_zooms = torch.tensor(q_zooms + kv_zooms).unique().tolist()

        seq_zoom = min((min(q_zooms + kv_zooms)), seq_zoom)  

        if (min(q_zooms + kv_zooms)) < token_zoom:
            raise ValueError(f"Zoom level {min(q_zooms + kv_zooms)} need to be refined. please indicate refine_zooms={refine_zooms}")


        embedder = get_embedder(**embed_confs, grid_layers=grid_layers, zoom=int(token_zoom))


        block = FieldAttentionBlock(
                    grid_layers,
                    token_zoom,
                    seq_zoom if seq_zoom > -1 else -1,
                    q_zooms,
                    kv_zooms,
                    att_dim = att_dim,
                    n_head_channels = n_head_channels,
                    token_overlap_std = token_overlap_std,
                    token_len_td = token_len_td,
                    seq_len_td = seq_len_td,
                    seq_nh_std = seq_nh_std,
                    mlp_token_overlap_td = mlp_token_overlap_td,
                    with_var_att = with_var_att,
                    ranks_std = ranks_std,
                    dropout=dropout,
                    embedder=embedder,
                    layer_confs=layer_confs,
                    layer_confs_emb=layer_confs_emb,
                    update=update,
                    separate_mlp_norm=separate_mlp_norm
                    )

        self.grid_layers = grid_layers
        self.multi_shift = multi_shift
        self.token_zoom = token_zoom

        if shift and rev_shift:
            self.fwd_fcn = self.forward_with_rev_shift
        elif shift:
            self.fwd_fcn = self.forward_with_shift
        else:
            self.fwd_fcn = self.forward_
        
        self.window_shift = shift
        self.refine_zooms = refine_zooms
        self.coarse_zooms = invert_dict(refine_zooms)

        self.block = block

    def generate_zoom(self, in_shape, zoom, device, zoom_patch_sample=-1, n_past_ts=0, n_future_ts=0, **kwargs):
        nt = n_past_ts + n_future_ts + 1

        x_zoom = self.init_missing_zooms((1,1,1,12*4**zoom,1)).to(device)

        if zoom_patch_sample > -1:
            x_zoom = x_zoom.view(1,1,1,12*4**zoom_patch_sample,-1,1)[:,:,:,0]

        return x_zoom.expand(in_shape[0],in_shape[1],nt,-1,in_shape[-1])


    def forward_with_rev_shift(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):
        for in_zoom, out_zoom in self.refine_zooms.items():
            x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
        
        if self.window_shift is not None:
            for zoom in self.qkv_zooms:
                grid_layer = self.grid_layers[str(self.token_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
                x_zooms[zoom] = grid_layer.apply_shift(x_zooms[zoom], self.window_shift, **sample_configs[zoom])[0]

        x_zooms = self.block(x_zooms, emb=emb, mask_zooms=mask_zooms if self.use_mask else {}, sample_configs=sample_configs)

        if self.window_shift is not None:
            for zoom in self.qkv_zooms:
                grid_layer = self.grid_layers[str(self.token_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
                x_zooms[zoom] = grid_layer.apply_shift(x_zooms[zoom], self.window_shift, **sample_configs[zoom], reverse=True)[0]

        for in_zoom, out_zoom in self.coarse_zooms.items():
            x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)

        return x_zooms


    def forward_with_shift(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):
        for in_zoom, out_zoom in self.refine_zooms.items():
            x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
        
        x_zooms_shift = x_zooms.copy()

        for zoom in self.qkv_zooms:
            grid_layer = self.grid_layers[str(self.token_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
            x_zooms_shift[zoom] = grid_layer.apply_shift(x_zooms_shift[zoom], self.window_shift, **sample_configs[zoom])[0]

        x_zooms = self.block(x_zooms_shift, emb=emb, mask_zooms=mask_zooms if self.use_mask else {}, sample_configs=sample_configs, x_zoom_res=x_zooms)

        for in_zoom, out_zoom in self.coarse_zooms.items():
            x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)

        return x_zooms


    def forward_(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):
        for in_zoom, out_zoom in self.refine_zooms.items():
            x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
        
        x_zooms_shift = x_zooms.copy()

        x_zooms = self.block(x_zooms_shift, emb=emb, mask_zooms=mask_zooms if self.use_mask else {}, sample_configs=sample_configs)

        for in_zoom, out_zoom in self.coarse_zooms.items():
            x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)

        return x_zooms
    

    def forward(self, x_zooms, emb=None, mask_zooms={}, sample_configs={}):
        
        x_zooms = self.fwd_fcn(x_zooms, emb=emb, mask_zooms=mask_zooms, sample_configs=sample_configs)
        
        x_zooms_out = {}
        for zoom in self.out_zooms:
            x_zooms_out[zoom] = x_zooms[zoom]

        return x_zooms_out



     


class FieldAttentionBlock(nn.Module):
  
    def __init__(self,
                 grid_layers,
                 token_zoom: int,
                 seq_zoom: int,
                 q_zooms: int,
                 kv_zooms: int,
                 att_dim= None,
                 dropout: float=0.0,
                 n_head_channels: int=32,
                 embedder: Dict[str, EmbedderSequential] = {},
                 token_len_td: List = [1, 1],
                 token_overlap_std: List = [0, 0, 0],
                 seq_len_td: List = [1, 1],
                 seq_nh_std: List = [False, False, False],
                 mlp_token_overlap_td: List = [0, 0],
                 ranks_std: List = [None, None, None],
                 with_var_att= False,
                 with_mlp_embedder = True,
                 layer_confs: Dict = {},
                 layer_confs_emb= {},
                 update='shift',
                 layer_norm=True,
                 separate_mlp_norm=False) -> None: 
               
        super().__init__()

        #add additional zoom level to indicate n_fields
        grid_layer_field = grid_layers[str(token_zoom)]
        grid_layer_att = grid_layers[str(seq_zoom)] if seq_zoom >-1 else -1

        global_update = token_zoom == 0

        if global_update:
            token_overlap_std[0] = 0

        self.with_nh_space = seq_nh_std[0] if seq_zoom>-1 else False

        self.seq_nh_std = seq_nh_std
        self.seq_overlap_td = [int(seq_nh_std[1]), int(seq_nh_std[2])]

        self.token_overlap_std = token_overlap_std
        self.mlp_token_overlap_td = mlp_token_overlap_td

        layer_confs_ = layer_confs.copy()
        layer_confs_['ranks'] = [ranks_std[1],ranks_std[0],ranks_std[2],None,None]

        layer_confs_emb_ = layer_confs_emb.copy()
        layer_confs_emb_['ranks'] = [ranks_std[1],ranks_std[0],ranks_std[2],None,None]

        self.seq_len_td = seq_len_td

        self.scale_shift = update == 'shift_scale'
        
        global_att = isinstance(grid_layer_att, int) and grid_layer_att == -1

        self.n_head_channels = n_head_channels
        self.grid_layer_field = grid_layer_field
        self.grid_layer_att = grid_layer_att

        self.n_groups = layer_confs.get('n_groups',1)
        self.emb_layers = nn.ModuleDict()
        self.mlp_emb_layers = nn.ModuleDict()
        self.q_layers = nn.ModuleDict()
        self.kv_layers = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        self.out_layers = nn.ModuleDict()

        self.dropout_att = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.dropout_mlp = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()

        self.q_zooms = q_zooms
        self.kv_zooms = kv_zooms

        out_feat_fac = 1
        out_feat_fac = 2 if update == 'shift_scale' else out_feat_fac
        self.out_feat_fac = out_feat_fac
        
        if len(self.q_zooms) == len(self.kv_zooms):
            self.self_att = True
            self.self_att = ((torch.tensor(q_zooms) - torch.tensor(kv_zooms)) == 0).all()
        else:
            self.self_att = False

        self.token_zoom = grid_layer_field.zoom

        self.q_projection_layers = nn.ModuleDict()
        self.kv_projection_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()

        self.tokenizer = Tokenizer(grid_layers, 
                                     q_zooms, 
                                     token_zoom, 
                                     overlap_thickness=token_overlap_std[0])
        if not self.self_att:
            self.kv_tokenizer = Tokenizer(grid_layers, 
                                     kv_zooms, 
                                     token_zoom, 
                                     overlap_thickness=token_overlap_std[0])
        else:
            self.kv_tokenizer = self.tokenizer

        self.n_in_features_zooms_q, self.n_out_features_zooms_q = self.tokenizer.get_features()
        self.n_in_features_zooms_kv, self.n_out_features_zooms_kv =  self.kv_tokenizer.get_features()
        
        self.token_size_in = [token_len_td[0], sum(self.n_in_features_zooms_q.values()), token_len_td[-1], 1]
        self.token_size = token_size_q = [token_len_td[0], sum(self.n_out_features_zooms_q.values()), token_len_td[-1], 1]

        self.token_size_in_kv = [token_len_td[0], sum(self.n_in_features_zooms_kv.values()), token_len_td[-1],1]
        token_size_kv = [token_len_td[0], sum(self.n_out_features_zooms_kv.values()), token_len_td[-1],1]
        
        in_features_q =  sum(self.n_in_features_zooms_q.values())
        in_features_kv = sum(self.n_in_features_zooms_kv.values())
        in_features_mlp = sum(self.n_in_features_zooms_q.values())

        in_features_t = token_len_td[0] + 2 * token_overlap_std[1]
        in_features_d = token_len_td[1] + 2 * token_overlap_std[2]
        
       
        self.in_features = in_features = [in_features_t, in_features_q, in_features_d, 1]
        self.in_features_kv = in_features_kv = [in_features_t, in_features_kv, in_features_d, 1]


        self.emb_layer_q = LinEmbLayer(self.token_size_in, 
                                       self.token_size_in, 
                                       layer_confs=layer_confs_, 
                                       identity_if_equal=True, 
                                       embedder=embedder, 
                                       layer_norm=layer_norm, 
                                       layer_confs_emb=layer_confs_emb_)

        if not self.self_att:
            self.emb_layer_kv = LinEmbLayer(self.token_size_in_kv, 
                                            self.token_size_in_kv, 
                                            layer_confs=layer_confs_,
                                            identity_if_equal=True, 
                                            embedder=embedder, 
                                            layer_norm=layer_norm, 
                                            layer_confs_emb=layer_confs_emb_)
        
        self.separate_mlp_norm = separate_mlp_norm
        if separate_mlp_norm:
            self.emb_layer_mlp = LinEmbLayer(self.token_size_in, 
                                            self.token_size_in, 
                                            layer_confs=layer_confs_, 
                                            identity_if_equal=True, 
                                            embedder=embedder if with_mlp_embedder else None, 
                                            layer_norm=layer_norm, 
                                            layer_confs_emb=layer_confs_emb_)

        out_dim_q = [1, 1 , 1, att_dim] 
        out_dim_kv = [1, 1, 1, 2 * att_dim]

        self.att_dim = att_dim

        
        self.q_projection_layer = get_layer(in_features, out_dim_q, layer_confs=layer_confs_, bias=False)
        self.kv_projection_layer = get_layer(in_features_kv, out_dim_kv, layer_confs=layer_confs_, bias=True)

        update_dim = copy.deepcopy(self.token_size_in)
        update_dim[-1] = update_dim[-1] * out_feat_fac

        self.out_layer_att = get_layer(out_dim_q, update_dim, layer_confs=layer_confs_)

        update_dim_mlp = copy.deepcopy(self.token_size)
        update_dim_mlp[-1] = update_dim_mlp[-1] * out_feat_fac

        self.mlp = MLP_fac(self.token_size_in, update_dim_mlp, hidden_dim=out_dim_q, dropout=dropout, layer_confs=layer_confs_, gamma=False) 

        self.gamma_res = nn.Parameter(torch.ones(self.token_size_in)*1e-7, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(self.token_size_in)*1e-7, requires_grad=True)
                
        self.gamma_res_mlp = nn.Parameter(torch.ones(len(q_zooms))*1e-7, requires_grad=True) 
        self.gamma_mlp = nn.Parameter(torch.ones(len(q_zooms))*1e-7, requires_grad=True) 
        

        self.pattern_tokens = 'b v (T t) N n (D d) f ->  b v T N D t n d f'
        self.pattern_tokens_reverse = 'b v T N D t n d f ->  b v (T t) (N n) (D d) f'
       # self.pattern_tokens_unfold = 'b v T N D (t n d) f ->  b v T N D t n d f'
        self.pattern_tokens_fold = 'b v T N D t n d f ->  b v T N D (t n d f)'

        self.pattern_tokens_nh_space = 'b v T N NH D (t n d f) -> b v T N D t (n NH) d f'


        self.att_pattern_chunks = 'b v (T t) (N n) (D d) 1 1 1 f ->  b v T N D t n d f'
        self.att_pattern_chunks_w_nh = 'b v (T t) N n (D d) 1 1 1 f ->  b v T N D t n d f'
        self.rearrange_dict = {}
        if global_att:
            self.rearrange_dict.update({'N': 1})
            self.seq_nh_std[0] = False
        else:
            self.rearrange_dict.update({'n': 4**(grid_layer_field.zoom-grid_layer_att.zoom)})
        
        if seq_len_td[0]==-1:
            self.rearrange_dict.update({'T': 1})
            self.seq_nh_std[1] = False
        else:
            self.rearrange_dict.update({'t': seq_len_td[0]})

        if seq_len_td[1]==-1:
            self.rearrange_dict.update({'D': 1})
            self.seq_nh_std[2] = False
        else:
            self.rearrange_dict.update({'d': seq_len_td[1]})

        self.rearrange_dict_nh = self.rearrange_dict.copy()
        if seq_zoom > -1:
            self.rearrange_dict_nh['n'] = self.grid_layer_att.adjc.shape[-1] * 4**(self.token_zoom - seq_zoom)
        
        if with_var_att:
            self.att_pattern = 'b v T N D t n d (NH H) -> (b T N D) NH (v t n d) H'
            self.mask_pattern = 'b v T N D t n d 1 -> (b T N D) 1 1 (v t n d)'
            self.att_pattern_reverse = '(b T N D) NH (v t n d) H -> b v (T t) (N n) (D d) 1 1 1 (NH H)'

        else:
            self.att_pattern = 'b v T N D t n d (NH H) -> (b v T N D) NH (t n d) H'
            self.mask_pattern = 'b v T N D t n d 1 -> (b v T N D) 1 1 (v t n d)'
            self.att_pattern_reverse = '(b v T N D) NH (t n d) H -> b v (T t) (N n) (D d) 1 1 1 (NH H)'

    def get_ms_features(self, zooms):
        features = {}
        for zoom in zooms:
            if self.token_zoom == 0:
                features[zoom] = max([12*4**(zoom - self.token_zoom),1])
            else: 
                features[zoom] = max([4**(zoom - self.token_zoom),1])
        return features
    

    def get_time_depth_overlaps(self, x: torch.Tensor,  overlap_td=None, mask=None):

        if overlap_td[0]>0:
            x = add_time_overlap_from_neighbor_patches(x, overlap=overlap_td[0], pad_mode= "edge")
        
        if overlap_td[1]>0:
            x = add_depth_overlap_from_neighbor_patches(x, overlap=overlap_td[1], pad_mode= "edge")

        return x
    
    
    def tokenize(self, x_zooms, zooms):

        x = combine_zooms(x_zooms, self.grid_layer_field.zoom, zooms)
        x = rearrange(x, self.pattern_tokens, t=self.token_size[0], n=self.token_size[1], d=self.token_size[2])

        return x
    
    def tokenize_emb(self, emb: Dict, sample_configs=None):
        if sample_configs is None:
            sample_configs = {}

        emb_cpy = dict(emb)  # shallow copy of top level
        emb_cpy['TimeEmbedder'] = {max(self.q_zooms): emb_cpy['TimeEmbedder'][max(self.q_zooms)]}
        
        if self.token_size[0] > 1 and 'TimeEmbedder' in emb_cpy:
            time_zooms = emb_cpy['TimeEmbedder']
            max_zoom_time = time_zooms[max(self.q_zooms)]
            time_zooms[max(self.q_zooms)] = max_zoom_time.view(max_zoom_time.shape[0], -1, self.token_size[0])[..., 0]

        if self.token_size[1] > 1 and 'DepthEmbedder' in emb_cpy:
            depth = emb_cpy['DepthEmbedder']
            emb_cpy['DepthEmbedder'] = depth.view(depth.shape[0], -1, self.token_size[2])[..., 0]

        return emb_cpy

    def forward(self, x_zooms, mask_zooms=[], emb=None, sample_configs={}):        

        zoom_field = self.grid_layer_field.zoom
    
        b,nv,t,s,d,f = x_zooms[self.q_zooms[0]].shape

        emb_tokenized = self.tokenize_emb(emb, sample_configs=sample_configs)

        x = self.tokenizer(x_zooms, sample_configs)
        x = rearrange(x, self.pattern_tokens, t=self.token_size_in[0], n=self.token_size_in[1], d=self.token_size_in[2])

        q = self.emb_layer_q(x, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])

        if not self.separate_mlp_norm:
            x = q

        q = self.get_time_depth_overlaps(q, overlap_td=self.token_overlap_std[1:])
        
        if not self.self_att:
            kv = self.kv_tokenizer(x_zooms, sample_configs)
            kv = rearrange(kv, self.pattern_tokens, t=self.token_size_in_kv[0], n=self.token_size_in_kv[1], d=self.token_size_in_kv[2])
            kv = self.emb_layer_kv(kv, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])
            kv = self.get_time_depth_overlaps(kv, overlap_td=self.token_overlap_std[1:])
        else:
            kv = q
        
        q = self.q_projection_layer(q, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])

        kv = self.kv_projection_layer(kv, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])

        q = rearrange(q, self.att_pattern_chunks, **self.rearrange_dict)

        mask = mask_zooms[zoom_field] if zoom_field in mask_zooms.keys() else None
        if self.with_nh_space:
            kv, mask = self.grid_layer_att.get_nh(kv, input_zoom=zoom_field, sample_configs=sample_configs[zoom_field], mask=mask)
            kv = rearrange(kv, self.att_pattern_chunks_w_nh, **self.rearrange_dict_nh)
        else:
            kv = rearrange(kv, self.att_pattern_chunks, **self.rearrange_dict)

        kv = self.get_time_depth_overlaps(kv, overlap_td=self.seq_overlap_td)
        
        if mask is not None:
            mask = self.get_time_depth_overlaps(mask, sample_configs=sample_configs[zoom_field])

        K,V = kv.chunk(2, dim=-1)
        kv=None

        b, v, T, N, D, t, n, d, f = q.shape
        q = rearrange(q, self.att_pattern, H=self.n_head_channels)
        K = rearrange(K, self.att_pattern, H=self.n_head_channels)
        V = rearrange(V, self.att_pattern, H=self.n_head_channels)

        mask = rearrange(mask, self.mask_pattern) if mask is not None else None

        att_out = safe_scaled_dot_product_attention(q, K, V, mask=mask)

        att_out = rearrange(att_out, self.att_pattern_reverse, b=b, v = v, T=T, N=N, D=D, t=t, n=n, d=d)

        att_out = self.out_layer_att(att_out, emb=emb_tokenized, sample_configs=sample_configs)

        if self.scale_shift:
            scale, shift = self.dropout_att(att_out).chunk(2,dim=-1)
            x = x * (1 + self.gamma_res * self.dropout_att(scale)) + self.gamma * self.dropout_att(shift)
        else:
            x = self.gamma_res * x + self.gamma * self.dropout_att(att_out)

        if self.separate_mlp_norm:
            x = self.emb_layer_mlp(x, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])      

        x = self.dropout_mlp(self.mlp(x, emb=emb_tokenized, sample_configs=sample_configs[int(zoom_field)]))

        x = x.split(tuple(self.n_out_features_zooms_q.values()), dim=-3)

        for k, (zoom, n) in enumerate(self.n_out_features_zooms_q.items()):
            
            x_ = rearrange(x[k], self.pattern_tokens_reverse, n=n)

            if self.scale_shift:
                scale, shift = x_.chunk(2,dim=-1)
                shift = insert_matching_time_patch(x_zooms[zoom], shift, zoom, max(self.q_zooms), sample_configs)
                x_zooms[zoom] = x_zooms[zoom] * (1 + self.gamma_res_mlp[k] * scale) + self.gamma_mlp[k] * shift
            else:
                x_ = insert_matching_time_patch(x_zooms[zoom], x_, zoom, max(self.q_zooms), sample_configs)
                x_zooms[zoom] = (1 + self.gamma_res_mlp[k]) * x_zooms[zoom] +  self.gamma_mlp[k] * x_

        return x_zooms
    
def invert_dict(d):
    inverted_d = {}
    for key, value in d.items():
        inverted_d[value] = key
    return inverted_d