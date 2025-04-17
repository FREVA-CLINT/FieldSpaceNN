from typing import List,Dict
import torch.nn as nn
import torch

from ...modules.icon_grids.grid_layer import GridLayer, MultiStepRelativeCoordinateManager, MultiRelativeCoordinateManager
from ...utils.helpers import check_get_missing_key
from ...modules.neural_operator.no_helpers import add_coordinates_to_emb_dict

from ...modules.neural_operator.no_helpers import get_no_layer,get_embedder_from_dict,get_embedder
from ...modules.neural_operator.no_blocks import PreActivation_NOBlock, NOBlock, Stacked_NOConv, Stacked_NOBlock, Stacked_PreActivationNOBlock, Stacked_PreActivationAttNOBlock
from ...modules.neural_operator import mg_layers as mg

class MGNO_EncoderDecoder_Block(nn.Module):
  
    def __init__(self,
                 rcm: MultiRelativeCoordinateManager,
                 input_levels: List[int],
                 input_dims: List[int],
                 global_levels_decode: List[int],
                 global_levels_no: List[int],
                 model_dims_out: List[int],
                 no_layer_settings: dict,
                 rule = 'fc', # ">" "<"
                 block_type = 'post_layer_norm',
                 mg_reduction = 'linear',
                 mg_reduction_embed_confs: Dict = None,
                 mg_reduction_embed_names: List = None,
                 mg_reduction_embed_names_mlp: List = None,
                 mg_reduction_embed_mode: str = 'sum',
                 embed_confs: Dict = None,
                 embed_names: List = None,
                 embed_mode: str = 'sum',
                 with_gamma: bool = True,
                 omit_backtransform: bool = False,
                 mg_att_dim = 128,
                 mg_n_head_channels = 16,
                 p_dropout=0,
                 mask_as_embedding = False,
                 level_diff_zero_linear = False,
                 layer_type='Dense',
                 rank=4,
                 n_vars_total=1,
                 rank_vars=4,
                 factorize_vars=False
                ) -> None: 
      
        super().__init__()
        
        vars_settings = {'n_vars_total': n_vars_total,
                         'rank_vars': rank_vars,
                         'factorize_vars': factorize_vars}
        self.mask_as_embedding = mask_as_embedding
        self.output_levels = global_levels_decode
        self.model_dims_out = model_dims_out
        self.rcm=rcm
        self.layers = nn.ModuleList()
        self.reduction_layers = nn.ModuleList()
        self.module_indices = []
        self.output_indices = []
        self.layer_indices = []
        self.output_levels = []
        self.add_coordinate_embedding_dict = {}
        layer_index = 0
        for output_idx, output_level in enumerate(global_levels_decode):

            input_indices = []
            layer_indices = []
            mg_input_dims = []
            mg_input_levels = []
            for input_idx, input_level in enumerate(input_levels):

                level_diff = output_level - input_level

                if  "<" in rule and level_diff>0:
                    continue

                elif ">" in rule and level_diff<0:
                    continue

                elif "=" in rule and level_diff!=0:
                    continue

                if rule == ">max" and (input_level!=max(input_levels) and (level_diff!=0)):
                    continue
                
                if rule == "<max" and (input_level!=max(input_levels) and (level_diff!=0)):
                    continue

                input_indices.append(input_idx)

                model_dim_in = input_dims[input_idx]
                model_dim_out = model_dims_out[output_idx]
                mg_input_dims.append(model_dim_out)
                mg_input_levels.append(input_level)

                global_level_no = global_levels_no[output_idx]

                if level_diff_zero_linear and level_diff==0:
                    layer = nn.Linear(model_dim_in, model_dim_out) if model_dim_in!=model_dim_out else nn.Identity()

                else:
                    no_layer_type = check_get_missing_key(no_layer_settings, "no_layer_type")

                    no_layer = get_no_layer(rcm,
                                            no_layer_type,
                                            input_level,
                                            global_level_no,
                                            output_level,
                                            precompute_encode=True,
                                            precompute_decode=True,
                                            layer_settings=no_layer_settings,
                                            normalize_to_mask= (mask_as_embedding==False))
                    
                    embedder = get_embedder(embed_names, embed_confs, embed_mode) if embed_names is not None and len(embed_names)>0 else None
                    
                    if 'post_layer_norm' in block_type:
                        layer = NOBlock(
                                        model_dim_in=model_dim_in,
                                        model_dim_out=model_dim_out,
                                        no_layer=no_layer,
                                        embedder=embedder,
                                        layer_type=layer_type,
                                        rank=rank,
                                        with_gamma = with_gamma,
                                        mask_as_embedding=mask_as_embedding,
                                        OW_zero=omit_backtransform,
                                        n_vars_total=n_vars_total,
                                        rank_vars=rank_vars,
                                        factorize_vars=factorize_vars
                                        )
                    elif 'pre_layer_norm' in block_type:
                        layer = PreActivation_NOBlock(
                                    model_dim_in=model_dim_in,
                                    model_dim_out=model_dim_out,
                                    no_layer=no_layer,
                                    embedder=embedder,
                                    layer_type=layer_type,
                                    rank=rank,
                                    with_gamma = with_gamma,
                                    mask_as_embedding=mask_as_embedding,
                                    n_vars_total=n_vars_total,
                                    rank_vars=rank_vars,
                                    factorize_vars=factorize_vars
                                    )
                
                self.layers.append(layer)
                layer_indices.append(layer_index)
                layer_index += 1

            if len(mg_input_dims)>1:
                if mg_reduction == 'linear':
                    reduction_layer = mg.LinearReductionLayer(mg_input_dims, model_dim_out,**vars_settings)
                                    
                elif mg_reduction == 'sum':
                    reduction_layer = mg.SumReductionLayer()

                elif mg_reduction == 'MGAttention' or 'MGDiffAttention'or 'MGDiffMAttention':
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
                    
                    if mg_reduction == 'MGAttention':
                        reduction_layer = mg.MGAttentionReductionLayer(mg_input_levels,
                                                                    mg_input_dims, 
                                                                    model_dim_out,
                                                                    att_dim=mg_att_dim,
                                                                    n_head_channels=mg_n_head_channels,
                                                                    embedder_grid=embedder,
                                                                    embedder_mlp=embedder_mlp,
                                                                    p_dropout=p_dropout,
                                                                    **vars_settings)
                    elif mg_reduction == 'MGDiffAttention':
                        reduction_layer = mg.MGDiffAttentionReductionLayer(torch.tensor(mg_input_levels)-output_level,
                                                                    mg_input_dims, 
                                                                    model_dim_out,
                                                                    att_dim=mg_att_dim,
                                                                    n_head_channels=mg_n_head_channels,
                                                                    embedder_grid=embedder,
                                                                    embedder_mlp=embedder_mlp,
                                                                    p_dropout=p_dropout,
                                                                    **vars_settings)
                    elif mg_reduction == 'MGDiffMAttention':
                        reduction_layer = mg.MGDiffMAttentionReductionLayer(torch.tensor(mg_input_levels)-output_level,
                                                                    mg_input_dims, 
                                                                    model_dim_out,
                                                                    att_dim=mg_att_dim,
                                                                    n_head_channels=mg_n_head_channels,
                                                                    embedder_grid=embedder,
                                                                    embedder_mlp=embedder_mlp,
                                                                    p_dropout=p_dropout,
                                                                    **vars_settings)
            else:
                reduction_layer = mg.IdentityReductionLayer()
            
            self.output_levels.append(output_level)
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

                masks_.append(mask_out)
                outputs_.append(x_out)

            emb = add_coordinates_to_emb_dict(self.rcm.grid_layers[str(self.output_levels[output_index])], 
                                              indices_sample["indices_layers"] if indices_sample else None, 
                                              emb=emb)
            
            x_out, mask_out = self.reduction_layers[output_index](outputs_, mask_levels=masks_, emb=emb)

            x_levels_out.append(x_out)
            mask_levels_out.append(mask_out)

        return x_levels_out, mask_levels_out

class MGNO_StackedEncoderDecoder_Block(nn.Module):
  
    def __init__(self,
                 rcm: MultiRelativeCoordinateManager,
                 input_levels: List[int],
                 input_dims: List[int],
                 global_levels_decode: List[int],
                 global_level_no: int,
                 model_dims_out: List[int],
                 no_layer_settings: dict,
                 no_level_step: int = 1,
                 concat_model_dim = 1,
                 block_type = 'post_layer_norm',
                 layer_type = 'Tucker',
                 concat_layer_type='Tucker',
                 reduction_layer_type = 'CrossTucker', 
                 rank=4,
                 rank_cross=2,
                 no_rank_decay=0,
                 embed_names=None,
                 embed_confs=None,
                 embed_mode='sum',
                 mask_as_embedding = False,
                 with_gamma = False,
                 p_dropout=0,
                 n_head_channels=16,
                 seq_level=2,
                 n_vars_total=1,
                 rank_vars=4,
                 factorize_vars=False
                ) -> None: 
      
        super().__init__()

        self.output_levels = global_levels_decode
        self.model_dims_out = model_dims_out
        self.rcm=rcm

        no_layer_type = check_get_missing_key(no_layer_settings, "no_layer_type")
        no_layers = nn.ModuleList()

        for input_level in range(int(torch.tensor(input_levels).min()), global_level_no + 1 - no_level_step, no_level_step):
            
            global_level_no_k = input_level + no_level_step

            no_layer = get_no_layer(rcm,
                                    no_layer_type,
                                    input_level,
                                    global_level_no_k,
                                    input_level,
                                    precompute_encode=True,
                                    precompute_decode=True,
                                    layer_settings=no_layer_settings,
                                    normalize_to_mask= (mask_as_embedding==False))

            no_layers.append(no_layer)

        if embed_names is not None:
            embedder = get_embedder(embed_names=embed_names,
                                        embed_confs=embed_confs,
                                        embed_mode=embed_mode)
        else:
            embedder = None
        
        no_conv_layer = Stacked_NOConv(
            input_levels,
            input_dims, 
            model_dims_out,
            no_layers,
            global_levels_decode,
            rank=rank,
            rank_cross=rank_cross,
            no_rank_decay=no_rank_decay,
            layer_type=layer_type,
            concat_model_dim=concat_model_dim,
            concat_layer_type=concat_layer_type,
            output_reduction_layer_type=reduction_layer_type,
            n_vars_total=n_vars_total,
            rank_vars=rank_vars,
            factorize_vars=factorize_vars
            )

        if block_type == 'post_layer_norm':
            self.layer = Stacked_NOBlock(no_conv_layer,
                                        layer_type=layer_type,
                                        rank=rank,
                                        embedder=embedder,
                                        with_gamma=with_gamma,
                                        grid_layers=rcm.grid_layers,
                                        p_dropout=p_dropout,
                                        n_vars_total=n_vars_total,
                                        rank_vars=rank_vars,
                                        factorize_vars=factorize_vars)
        
        elif block_type == 'pre_layer_norm':
            self.layer = Stacked_PreActivationNOBlock(no_conv_layer,
                                        layer_type=layer_type,
                                        rank=rank,
                                        embedder=embedder,
                                        with_gamma=with_gamma,
                                        grid_layers=rcm.grid_layers,
                                        p_dropout=p_dropout,
                                        n_vars_total=n_vars_total,
                                        rank_vars=rank_vars,
                                        factorize_vars=factorize_vars)
        
        elif block_type == 'pre_att_layer_norm':
            self.layer = Stacked_PreActivationAttNOBlock(no_conv_layer,
                                        layer_type=layer_type,
                                        rank=rank,
                                        embedder=embedder,
                                        with_gamma=with_gamma,
                                        grid_layers=rcm.grid_layers,
                                        n_head_channels=n_head_channels,
                                        p_dropout=p_dropout,
                                        seq_level=seq_level,
                                        n_vars_total=n_vars_total,
                                        rank_vars=rank_vars,
                                        factorize_vars=factorize_vars)

    def forward(self, x_levels, coords_in=None, coords_out=None, indices_sample=None, mask_levels=None, emb=None):
        
        x_levels, mask_levels = self.layer(x_levels, indices_sample=indices_sample, mask_levels=mask_levels, emb=emb)

        return x_levels, mask_levels