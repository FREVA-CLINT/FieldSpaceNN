from typing import List,Dict
from einops import rearrange
import copy
from omegaconf import ListConfig

import torch
import torch.nn as nn

from ..base import get_layer, MLP_fac
from .mg_base import refine_zoom,coarsen_zoom, Tokenizer, LinEmbLayer, add_time_overlap_from_neighbor_patches, add_depth_overlap_from_neighbor_patches

from ..grids.grid_layer import GridLayer
from ..transformer.transformer_base import safe_scaled_dot_product_attention

from ..embedding.embedder import EmbedderSequential
from ...modules.embedding.embedder import get_embedder

from ..grids.grid_utils import insert_matching_time_patch

from ...utils.helpers import check_value


class FieldAttentionConfig:
    def __init__(self, 
                 token_zoom: int,
                 q_zooms = -1,
                 kv_zooms = -1,
                 att_dim= 64,
                 target_zooms: List = None,
                 token_len_depth: List|int = [1],
                 token_len_time: List|int = 1,
                 token_overlap_space: List|int = False,
                 token_overlap_time: List|int = False,
                 token_overlap_depth: List|int = False,
                 token_overlap_mlp_time: List|bool = False,
                 token_overlap_mlp_depth: List|bool = False,
                 rank_space: List|int = None,
                 rank_time: List|int = None,
                 rank_depth: List|int = None,
                 rank_features: List|int = None,
                 seq_len_zoom: int = -1,
                 seq_len_time: List|int = -1,
                 seq_len_depth: List|int = -1,
                 seq_overlap_space: bool = False,
                 seq_overlap_time: bool = False,
                 seq_overlap_depth: bool = False,
                 with_var_att= False,
                 shift= None,
                 multi_shift= False,
                 update = 'shift',
                 refine_zooms = {},
                 separate_mlp_norm=True,
                 **kwargs):

        inputs = copy.deepcopy(locals())

        for input, value in inputs.items():
            if input == 'kwargs':
                for input_kw, value_kw in value.items():
                    setattr(self, input_kw, value_kw)
            else:
                setattr(self, input, value)


class FieldAttentionModule(nn.Module):
  
    def __init__(self, 
                grid_layers: Dict[str, GridLayer],
                in_zooms: List[int],
                out_zooms: List[int],
                q_zooms:List|int,
                kv_zooms:List|int,
                token_zoom: int,
                target_zooms: List = None,
                in_features: int = 1,
                n_groups_variables: List = [1],
                token_len_depth: List|int = 1,
                token_len_time: List|int = 1,
                token_overlap_space: List|int = False,
                token_overlap_time: List|int = False,
                token_overlap_depth: List|int = False,
                token_overlap_mlp_time: List|bool = False,
                token_overlap_mlp_depth: List|bool = False,
                rank_space : List|int =None,
                rank_time : List|int =None,
                rank_depth : List|int =None,
                rank_features: List|int =None,
                seq_len_zoom: int = -1,
                seq_len_time: List = -1,
                seq_len_depth: List = -1,
                seq_overlap_space: bool = False,
                seq_overlap_time: bool = False,
                seq_overlap_depth: bool = False,
                with_var_att= False,
                use_mask = False,
                att_dim = None,
                n_head_channels=16,
                refine_zooms= {},
                shift = False,
                multi_shift = False,
                dropout:float = 0,
                update='shift',
                separate_mlp_norm=True,
                embed_confs={},
                layer_confs:List|Dict ={},
                layer_confs_emb:List|Dict ={}
                 ) -> None:
        super().__init__()
        
        n_groups = len(n_groups_variables)
        token_len_depth = check_value(token_len_depth, n_groups)
        token_len_time = check_value(token_len_time, n_groups)

        token_overlap_space = check_value(token_overlap_space, n_groups)
        token_overlap_time = check_value(token_overlap_time, n_groups)
        token_overlap_depth = check_value(token_overlap_depth, n_groups)
        token_overlap_mlp_time = check_value(token_overlap_mlp_time, n_groups)
        token_overlap_mlp_depth = check_value(token_overlap_mlp_depth, n_groups)

        layer_confs = check_value(layer_confs, n_groups)

        layer_confs_emb = check_value(layer_confs_emb, n_groups)

        rank_space = check_value(rank_space, n_groups)
        rank_time = check_value(rank_time, n_groups)
        rank_depth = check_value(rank_depth, n_groups)
        rank_features = check_value(rank_features, n_groups)
        

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

        seq_zoom = min((min(q_zooms + kv_zooms)), seq_len_zoom)  

        if (min(q_zooms + kv_zooms)) < token_zoom:
            raise ValueError(f"Zoom level {min(q_zooms + kv_zooms)} need to be refined. please indicate refine_zooms={refine_zooms}")

        self.blocks = nn.ModuleList()

        for k in range(n_groups):
            
            layer_confs[k]['n_variables'] = n_groups_variables[k]
            block = FieldAttentionBlock(
                        grid_layers,
                        token_zoom,
                        seq_zoom if seq_zoom > -1 else -1,
                        q_zooms,
                        kv_zooms,
                        att_dim,
                        target_zooms = target_zooms,
                        in_features = in_features,
                        token_len_depth= token_len_depth[k],
                        token_len_time= token_len_time[k],
                        token_overlap_space= token_overlap_space[k],
                        token_overlap_time= token_overlap_time[k],
                        token_overlap_depth= token_overlap_depth[k],
                        token_overlap_mlp_time= token_overlap_mlp_time[k],
                        token_overlap_mlp_depth= token_overlap_mlp_depth[k],
                        rank_space = rank_space[k],
                        rank_time = rank_time[k],
                        rank_depth = rank_depth[k],
                        rank_features = rank_features[k],
                        seq_len_time= seq_len_time,
                        seq_len_depth= seq_len_depth,
                        seq_overlap_space = seq_overlap_space,
                        seq_overlap_time = seq_overlap_time,
                        seq_overlap_depth = seq_overlap_depth,
                        with_var_att = with_var_att,
                        n_head_channels = n_head_channels,
                        dropout=dropout,
                        embed_confs=embed_confs,
                        layer_confs=layer_confs[k],
                        layer_confs_emb=layer_confs_emb[k],
                        update=update,
                        separate_mlp_norm=separate_mlp_norm
                        )
            self.blocks.append(block)

        self.grid_layers = grid_layers
        self.multi_shift = multi_shift
        self.token_zoom = token_zoom
        self.shift = shift
        self.direction = 'east'

        self.refine_zooms = refine_zooms
        self.coarse_zooms = invert_dict(refine_zooms)

        self.block = block


    def refine_groups(self, x_zooms_groups: List):
        for k, x_zooms in enumerate(x_zooms_groups):
            for in_zoom, out_zoom in self.refine_zooms.items():
                x_zooms[out_zoom] = refine_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
            x_zooms_groups[k] = x_zooms

        return x_zooms_groups
        
    def coarse_groups(self, x_zooms_groups: List):
        for k, x_zooms in enumerate(x_zooms_groups):
            for in_zoom, out_zoom in self.coarse_zooms.items():
                x_zooms[out_zoom] = coarsen_zoom(x_zooms[in_zoom], in_zoom, out_zoom)
            x_zooms_groups[k] = x_zooms
        
        return x_zooms_groups

    def shift_groups(self, x_zooms_groups, sample_configs={}, reverse=False):
        for k, x_zooms in enumerate(x_zooms_groups):
            for zoom in self.qkv_zooms:
                grid_layer = self.grid_layers[str(self.token_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
                x_zooms[zoom] = grid_layer.apply_shift(x_zooms[zoom], self.direction, **sample_configs[zoom])[0]
        x_zooms_groups[k] = x_zooms

        return x_zooms_groups

    def shift_groups_reverse(self, x_zooms_groups, sample_configs={}, reverse=False):
        for k, x_zooms in enumerate(x_zooms_groups):
            for zoom in self.qkv_zooms:
                grid_layer = self.grid_layers[str(self.token_zoom + 1)] if not self.multi_shift else self.grid_layers[str(zoom)]
                x_zooms[zoom] = grid_layer.apply_shift(x_zooms[zoom], self.direction, **sample_configs[zoom], reverse=True)[0]
        x_zooms_groups[k] = x_zooms

        return x_zooms_groups

    def forward(self, x_zooms_groups: List, emb_groups: list, mask_groups={}, sample_configs={}):
        
        x_zooms_groups = self.refine_groups(x_zooms_groups)

        if self.shift:
            x_zooms_groups = self.shift_groups(x_zooms_groups, sample_configs=sample_configs)
        
        x_ress, qs, Ks, Vs, masks, shapes, seq_lens = [], [], [], [], [], [], []
        for k, block in enumerate(self.blocks):
            x_res, q, K, V, mask, shape = block.create_QKV(x_zooms_groups[k], emb=emb_groups[k], mask_zooms=mask_groups[k] if self.use_mask else {}, sample_configs=sample_configs)
            x_ress.append(x_res)
            qs.append(q)
            Ks.append(K)
            Vs.append(V)
            masks.append(mask)
            shapes.append(shape)
            seq_lens.append(q.shape[-2])
        
        q = torch.concat(qs, dim=-2)
        K = torch.concat(Ks, dim=-2)
        V = torch.concat(Vs, dim=-2)
        mask = torch.concat(masks, dim=-2) if self.use_mask else None

        att_out = safe_scaled_dot_product_attention(q, K, V, mask=mask)

        att_outs = att_out.split(seq_lens, dim=-2)

        for k, att_out in enumerate(att_outs):
            x_zooms_groups[k] = self.blocks[k].forward_mlp(x_zooms_groups[k], x_ress[k], att_outs[k], shapes[k], emb=emb_groups[k], sample_configs=sample_configs)

        if self.shift:
            x_zooms_groups = self.shift_groups_reverse(x_zooms_groups, sample_configs=sample_configs)
        
        for k, x_zooms in enumerate(x_zooms_groups):
            x_zooms_out = {}

            for zoom in self.out_zooms:
                x_zooms_out[zoom] = x_zooms[zoom]

            x_zooms_groups[k] = x_zooms_out

        return x_zooms_groups



     


class FieldAttentionBlock(nn.Module):
  
    def __init__(self,
                 grid_layers,
                 token_zoom: int,
                 seq_zoom: int,
                 q_zooms: List,
                 kv_zooms: List,
                 att_dim: int,
                 target_zooms: List = None,
                 in_features: int = 1,
                 token_len_depth: int = 1,
                 token_len_time: int = 1,
                 token_overlap_space: bool = False,
                 token_overlap_time: bool = False,
                 token_overlap_depth: bool = False,
                 token_overlap_mlp_time: bool = False,
                 token_overlap_mlp_depth: bool = False,
                 rank_space : int =None,
                 rank_time : int =None,
                 rank_depth : int =None,
                 rank_features: int = None,
                 dropout: float=0.0,
                 n_head_channels: int=32,
                 embed_confs: Dict = {},
                 seq_len_time: int = -1,
                 seq_len_depth: int = -1,
                 seq_overlap_space: bool = False,
                 seq_overlap_time: bool = False,
                 seq_overlap_depth: bool = False,
                 with_var_att= False,
                 layer_confs: Dict = {},
                 layer_confs_emb= {},
                 update='shift',
                 layer_norm=True,
                 separate_mlp_norm=False) -> None: 
               
        super().__init__()

        target_zooms = q_zooms if target_zooms is None else target_zooms
        self.seq_overlap_time = seq_overlap_time
        self.seq_overlap_depth = seq_overlap_depth
        self.seq_overlap_space = seq_overlap_space if seq_zoom > -1 else False
        self.token_len_time = token_len_time
        self.token_len_depth = token_len_depth
        self.token_overlap_depth = token_overlap_depth
        self.token_overlap_time = token_overlap_time

        grid_layer_field = grid_layers[str(token_zoom)] if token_zoom >-1 else grid_layers[str(0)]
        grid_layer_att = grid_layers[str(seq_zoom)] if seq_zoom >-1 else -1

        global_update = token_zoom == -1

        if global_update:
            token_overlap_space = 0

        self.att_dim = att_dim

        layer_confs_ = layer_confs.copy()
        layer_confs_['ranks'] = [rank_time,rank_space,rank_depth,rank_features,rank_features]

        layer_confs_emb_ = layer_confs_emb.copy()
        layer_confs_emb_['ranks'] = [rank_time,rank_space,rank_depth,rank_features,rank_features]

        self.scale_shift = update == 'shift_scale'
        
        global_att = isinstance(grid_layer_att, int) and grid_layer_att == -1

        self.n_head_channels = n_head_channels
        self.grid_layer_field = grid_layer_field
        self.grid_layer_att = grid_layer_att

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

        update_dim = in_features
        update_dim = 2 * in_features if update == 'shift_scale' else update_dim
        
        if len(self.q_zooms) == len(self.kv_zooms):
            self.self_att = True
            self.self_att = ((torch.tensor(q_zooms) - torch.tensor(kv_zooms)) == 0).all()
        else:
            self.self_att = False

        self.token_zoom = grid_layer_field.zoom

        self.q_projection_layers = nn.ModuleDict()
        self.kv_projection_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()

        tokenizer_update = Tokenizer(target_zooms, 
                                    token_zoom,
                                    grid_layers=grid_layers,
                                    overlap_thickness=int(token_overlap_space),
                                    token_len_time=token_len_time,
                                    token_len_depth=token_len_depth)
        
        self.tokenizer = Tokenizer(q_zooms, 
                                    token_zoom,
                                    grid_layers=grid_layers,
                                    overlap_thickness=int(token_overlap_space),
                                    token_len_time=token_len_time,
                                    token_len_depth=token_len_depth)
        if not self.self_att:
            self.kv_tokenizer = Tokenizer(kv_zooms, 
                                     token_zoom,
                                     grid_layers=grid_layers, 
                                     overlap_thickness=int(token_overlap_space),
                                     token_len_time=token_len_time,
                                     token_len_depth=token_len_depth)
        else:
            self.kv_tokenizer = self.tokenizer

        _, self.n_out_features_update = tokenizer_update.get_features()
        self.n_in_features_zooms_q, self.n_out_features_zooms_q = self.tokenizer.get_features()
        self.n_in_features_zooms_kv, self.n_out_features_zooms_kv =  self.kv_tokenizer.get_features()
        
        self.token_size_space = [token_len_time, sum(self.n_in_features_zooms_q.values()), token_len_depth, in_features]
        self.token_size_space_kv = [token_len_time, sum(self.n_in_features_zooms_kv.values()), token_len_depth, in_features]
        self.token_size_update = [token_len_time, sum(self.n_out_features_update.values()), token_len_depth, in_features]

        token_size_in_overlap = [token_len_time + 2 * token_overlap_time, sum(self.n_in_features_zooms_q.values()), token_len_depth + 2 * token_overlap_depth, in_features]
        token_size_in_mlp_overlap = [token_len_time + 2 * token_overlap_mlp_time, sum(self.n_in_features_zooms_q.values()), token_len_depth + 2 * token_overlap_mlp_depth, in_features]
        token_size_in_kv_overlap = [token_len_time + 2 * token_overlap_time, sum(self.n_in_features_zooms_kv.values()), token_len_depth + 2 * token_overlap_depth, in_features]

        self.separate_mlp_norm = separate_mlp_norm

        input_zoom_field = embed_confs.get("input_zoom", min(q_zooms))
        embedder: EmbedderSequential = get_embedder(**embed_confs, grid_layers=grid_layers, zoom=input_zoom_field)

        emb_tokenizer = Tokenizer(
            input_zooms=[input_zoom_field] if embedder and embedder.has_space() else [],
            token_zoom=token_zoom,
            token_len_time=token_len_time if embedder and embedder.has_time() else 1,
            token_len_depth=token_len_depth if embedder and embedder.has_depth() else 1,
            overlap_thickness=int(embed_confs.get("token_overlap_space", False)),
            grid_layers=grid_layers
        ) 

        layer_confs_emb_field = layer_confs_emb_.copy()
        layer_confs_emb_field['ranks'] = embed_confs.get("ranks", [*layer_confs_['ranks'], None]) 

        emb_tokenizer_out_features = copy.deepcopy(self.token_size_space)
        emb_tokenizer_out_features[1] = self.token_size_space[1] if embedder and embedder.has_space() else 1

        self.emb_layer_q_field = LinEmbLayer(
            emb_tokenizer_out_features,
            emb_tokenizer_out_features,
            layer_confs=layer_confs_,
            identity_if_equal=True,
            embedder=embedder,
            field_tokenizer= emb_tokenizer,
            output_zoom=max(self.q_zooms),
            layer_norm=True,
            layer_confs_emb=layer_confs_emb_field
        )
            
        
        if separate_mlp_norm:
            self.emb_layer_mlp = LinEmbLayer(
                self.token_size_space,
                self.token_size_space,
                layer_confs=layer_confs_,
                identity_if_equal=True,
                embedder=embedder,
                field_tokenizer= emb_tokenizer,
                output_zoom=max(self.q_zooms),
                layer_norm=layer_norm,
                layer_confs_emb=layer_confs_emb_,
            )
        else:
            self.emb_layer_mlp = None

        if not self.self_att:
            self.emb_layer_kv = LinEmbLayer(
                self.token_size_space_kv,
                self.token_size_space_kv,
                layer_confs=layer_confs_,
                identity_if_equal=True,
                embedder=embedder,
                field_tokenizer= emb_tokenizer,
                output_zoom=max(self.q_zooms),
                layer_norm=layer_norm,
                layer_confs_emb=layer_confs_emb_,
            )
        else:
            self.emb_layer_kv = None

        out_dim_q = [1, 1 , 1, att_dim] 
        out_dim_kv = [1, 1, 1, 2 * att_dim]

        update_dims = [*self.token_size_space[:-1], update_dim]
        update_dims_mlp = [*self.token_size_update[:-1], update_dim]

        self.q_projection_layer = get_layer(token_size_in_overlap, out_dim_q, layer_confs=layer_confs_, bias=False)
        self.kv_projection_layer = get_layer(token_size_in_kv_overlap, out_dim_kv, layer_confs=layer_confs_, bias=True)
        self.out_layer_att = get_layer(out_dim_q, update_dims, layer_confs=layer_confs_)

        self.gamma_res = nn.Parameter(torch.ones(self.token_size_space) * 1e-7, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(self.token_size_space) * 1e-7, requires_grad=True)

        self.mlp = MLP_fac(token_size_in_mlp_overlap, update_dims_mlp, hidden_dim=out_dim_q, dropout=dropout, layer_confs=layer_confs_, gamma=False)
        self.gamma_res_mlp = nn.Parameter(torch.ones(len(target_zooms)) * 1e-7, requires_grad=True)
        self.gamma_mlp = nn.Parameter(torch.ones(len(target_zooms)) * 1e-7, requires_grad=True)
        

        self.pattern_tokens = 'b v (T t) N n (D d) f ->  b v T N D t n d f'
        self.pattern_tokens_reverse = 'b v T N D t n d f ->  b v (T t) (N n) (D d) f'
        self.pattern_tokens_fold = 'b v T N D t n d f ->  b v T N D (t n d f)'

        self.pattern_tokens_nh_space = 'b v T N NH D (t n d f) -> b v T N D t (n NH) d f'

        self.att_pattern_chunks = 'b v (T t) (N n) (D d) 1 1 1 f ->  b v T N D t n d f'
        self.att_pattern_chunks_w_nh = 'b v (T t) N n (D d) 1 1 1 f ->  b v T N D t n d f'
        self.rearrange_dict = {}
        if global_att:
            self.rearrange_dict.update({'N': 1})
            self.seq_overlap_space = False
        else:
            self.rearrange_dict.update({'n': 4**(grid_layer_field.zoom-grid_layer_att.zoom)})
        
        if seq_len_time ==-1:
            self.rearrange_dict.update({'T': 1})
            self.seq_overlap_time = False
        else:
            self.rearrange_dict.update({'t': seq_len_time})

        if seq_len_depth==-1:
            self.rearrange_dict.update({'D': 1})
            self.seq_overlap_depth = False
        else:
            self.rearrange_dict.update({'d': seq_len_depth})

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
    

    def get_time_depth_overlaps(self, x: torch.Tensor,  overlap_time=False, overlap_depth=False, mask=None):

        if overlap_time:
            x = add_time_overlap_from_neighbor_patches(x, overlap=1, pad_mode= "edge")
        
        if overlap_depth:
            x = add_depth_overlap_from_neighbor_patches(x, overlap=1, pad_mode= "edge")

        return x
    
    
    def select_emb(self, emb: Dict, sample_configs=None):
        if sample_configs is None:
            sample_configs = {}

        if emb is None:
            return None

        emb_cpy = dict(emb)  # shallow copy of top level
        emb_cpy['TimeEmbedder'] = {max(self.q_zooms): emb_cpy['TimeEmbedder'][max(self.q_zooms)]}
        


        return emb_cpy
    
    def create_QKV(self, x_zooms, emb=None, sample_configs={}, mask_zooms={}):
        zoom_field = self.grid_layer_field.zoom

        x = self.tokenizer(x_zooms, sample_configs)

        emb_tokenized = emb#self.select_emb(emb)

        if self.emb_layer_q_field is not None:
            q = self.emb_layer_q_field(x, emb=emb_tokenized, sample_configs=sample_configs)

        x_base = q if not self.separate_mlp_norm else x

        q = self.get_time_depth_overlaps(q, overlap_time=self.token_overlap_time, overlap_depth=self.token_overlap_depth)

        if not self.self_att:
            kv = self.kv_tokenizer(x_zooms, sample_configs)
            kv = self.emb_layer_kv(kv, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])
            kv = self.get_time_depth_overlaps(kv, overlap_time=self.token_overlap_time, overlap_depth=self.token_overlap_depth)
        else:
            kv = q

        q = self.q_projection_layer(q, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])
        kv = self.kv_projection_layer(kv, emb=emb_tokenized, sample_configs=sample_configs[zoom_field])

        zoom_field = self.grid_layer_field.zoom

        q = rearrange(q, self.att_pattern_chunks, **self.rearrange_dict)

        mask = mask_zooms[zoom_field] if zoom_field in mask_zooms.keys() else None
        if self.seq_overlap_space:
            kv, mask = self.grid_layer_att.get_nh(kv, input_zoom=zoom_field, sample_configs=sample_configs[zoom_field], mask=mask)
            kv = rearrange(kv, self.att_pattern_chunks_w_nh, **self.rearrange_dict_nh)
        else:
            kv = rearrange(kv, self.att_pattern_chunks, **self.rearrange_dict)

        kv = self.get_time_depth_overlaps(kv, overlap_time=self.seq_overlap_time, overlap_depth=self.seq_overlap_depth)
        
        if mask is not None:
            mask = self.get_time_depth_overlaps(mask, overlap_time=self.seq_overlap_time, overlap_depth=self.seq_overlap_depth)

        K, V = kv.chunk(2, dim=-1)

        b, v, T, N, D, t, n, d, f = q.shape
        q = rearrange(q, self.att_pattern, H=self.n_head_channels)
        K = rearrange(K, self.att_pattern, H=self.n_head_channels)
        V = rearrange(V, self.att_pattern, H=self.n_head_channels)

        mask = rearrange(mask, self.mask_pattern) if mask is not None else None

        shape = {'b': b, 'v': v, 'T': T, 'N': N, 'D': D, 't': t, 'n': n, 'd': d}

        return x_base, q, K, V, mask, shape
    

    def attend(self, q, kv, mask=None, sample_configs={}):
        zoom_field = self.grid_layer_field.zoom

        q = rearrange(q, self.att_pattern_chunks, **self.rearrange_dict)

        if self.seq_overlap_space:
            kv, mask = self.grid_layer_att.get_nh(kv, input_zoom=zoom_field, sample_configs=sample_configs[zoom_field], mask=mask)
            kv = rearrange(kv, self.att_pattern_chunks_w_nh, **self.rearrange_dict_nh)
        else:
            kv = rearrange(kv, self.att_pattern_chunks, **self.rearrange_dict)

        kv = self.get_time_depth_overlaps(kv, overlap_time=self.seq_overlap_time, overlap_depth=self.seq_overlap_depth)
        
        if mask is not None:
            mask = self.get_time_depth_overlaps(mask, overlap_time=self.seq_overlap_time, overlap_depth=self.seq_overlap_depth)

        K, V = kv.chunk(2, dim=-1)

        b, v, T, N, D, t, n, d, f = q.shape
        q = rearrange(q, self.att_pattern, H=self.n_head_channels)
        K = rearrange(K, self.att_pattern, H=self.n_head_channels)
        V = rearrange(V, self.att_pattern, H=self.n_head_channels)

        mask = rearrange(mask, self.mask_pattern) if mask is not None else None

        att_out = safe_scaled_dot_product_attention(q, K, V, mask=mask)

        att_out = rearrange(att_out, self.att_pattern_reverse, b=b, v=v, T=T, N=N, D=D, t=t, n=n, d=d)

        shape = {'b': b, 'v': v, 'T': T, 'N': N, 'D': D, 't': t, 'n': n, 'd': d}
        return att_out, shape

    def forward_mlp(self, x_zooms, x_base, att_out, shape, emb=None, sample_configs={}):

        emb_tokenized = emb#self.select_emb(emb)

        att_out = rearrange(att_out, self.att_pattern_reverse, **shape)

        zoom_field = self.grid_layer_field.zoom

        att_out = self.out_layer_att(att_out, emb=emb_tokenized, sample_configs=sample_configs)
        if self.scale_shift:
            scale, shift = self.dropout_att(att_out).chunk(2, dim=-1)
            x = x_base * (1 + self.gamma_res * self.dropout_att(scale)) + self.gamma * self.dropout_att(shift)
        else:
            x = self.gamma_res * x_base + self.gamma * self.dropout_att(att_out)

        if self.separate_mlp_norm and self.emb_layer_mlp is not None:
            x = self.emb_layer_mlp(x, emb=emb_tokenized, sample_configs=sample_configs)

        x = self.mlp(x, emb=emb_tokenized, sample_configs=sample_configs[int(zoom_field)])

        x = x.split(tuple(self.n_out_features_update.values()), dim=-3)

        for k, (zoom, n) in enumerate(self.n_out_features_update.items()):
            if x_zooms and x is not None:
                x_out = rearrange(x[k], self.pattern_tokens_reverse, n=n)

                if self.scale_shift:
                    scale, shift = x_out.chunk(2, dim=-1)
                    shift = insert_matching_time_patch(x_zooms[zoom], shift, zoom, max(self.q_zooms), sample_configs)
                    scale = insert_matching_time_patch(x_zooms[zoom], scale, zoom, max(self.q_zooms), sample_configs)
                    x_zooms[zoom] = x_zooms[zoom] * (1 + self.gamma_res_mlp[k] * scale) + self.gamma_mlp[k] * shift
                else:
                    x_out = insert_matching_time_patch(x_zooms[zoom], x_out, zoom, max(self.q_zooms), sample_configs)
                    x_zooms[zoom] = (1 + self.gamma_res_mlp[k]) * x_zooms[zoom] + self.gamma_mlp[k] * x_out

        return x_zooms

    def forward(self, x_zooms={}, mask_zooms={}, emb=None, sample_configs={}):
        x_base, q, kv = self.create_QKV(x_zooms, emb=emb, sample_configs=sample_configs)
        att_out = self.attend(q, kv, mask=mask_zooms[self.grid_layer_field.zoom], sample_configs=sample_configs)
        return self.forward_mlp(x_zooms, x_base, att_out, emb=emb, sample_configs=sample_configs)

    
def invert_dict(d):
    inverted_d = {}
    for key, value in d.items():
        inverted_d[value] = key
    return inverted_d
