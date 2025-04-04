import torch
import torch.nn as nn

from typing import List

from .mgno_encoderdecoder_block import MGNO_EncoderDecoder_Block, MGNO_StackedEncoderDecoder_Block
from .mgno_processing_block import MGNO_Processing_Block

from .mgno_block_confs import MGProcessingConfig, MGEncoderDecoderConfig, MGStackedEncoderDecoderConfig
from .mgno_base_model import MGNO_base_model

from ...modules.neural_operator.no_helpers import get_embedder
from ...modules.neural_operator.no_helpers import add_coordinates_to_emb_dict,add_mask_to_emb_dict

class MGNO_Transformer_MG(MGNO_base_model):
    def __init__(self, 
                 mgrids,
                 block_configs: List,
                 input_dim: int=1,
                 lifting_dim: int=1,
                 output_dim: int=1,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True,
                 p_dropout=0.,
                 mask_as_embedding = False,
                 input_embed_names = None,
                 input_embed_confs = None,
                 input_embed_mode = 'sum',
                 concat_interp = False,
                 **kwargs
                 ) -> None: 
        
        self.input_dim = input_dim + concat_interp
        
        global_levels = torch.tensor(0).view(-1)
        for block_conf in block_configs:
            if hasattr(block_conf,'global_levels_output'):
                global_levels = torch.concat((global_levels, torch.tensor(block_conf.global_levels_output).view(-1))) 
            if hasattr(block_conf,'global_levels_no'):
                #if not isinstance(block_conf.global_levels_no, list):
                global_levels = torch.concat((global_levels, torch.tensor(block_conf.global_levels_no).view(-1)))

        global_levels_max = torch.concat((torch.tensor(global_levels).view(-1)
                                     ,torch.tensor(0).view(-1))).max()
        
        global_levels = torch.arange(global_levels_max+1)
        
        super().__init__(mgrids, 
                         global_levels,
                         rotate_coord_system=rotate_coord_system,
                         interpolate_input=kwargs.get("interpolate_input", False),
                         density_embedder=kwargs.get("density_embedder", False),
                         interpolator_settings=kwargs.get("interpolator_settings", None),
                         concat_interp=concat_interp)
        
       
        self.Blocks = nn.ModuleList()

        input_levels = [0]
        input_dims = [lifting_dim]

        for block_idx, block_conf in enumerate(block_configs):
            
            if isinstance(block_conf, MGEncoderDecoderConfig):
                block = MGNO_EncoderDecoder_Block(
                                            self.rcm,
                                            input_levels,
                                            input_dims,
                                            block_conf.global_levels_output,
                                            block_conf.global_levels_no,
                                            block_conf.model_dims_out,
                                            rule=block_conf.rule,
                                            no_layer_settings=check_get(block_conf,kwargs,'no_layer_settings'),
                                            block_type=check_get(block_conf,kwargs,'block_type'),
                                            mg_reduction=check_get(block_conf, kwargs, "mg_reduction"),
                                            mg_reduction_embed_confs=check_get(block_conf, kwargs, "mg_reduction_embed_confs"),
                                            mg_reduction_embed_names=check_get(block_conf, kwargs, "mg_reduction_embed_names"),
                                            mg_reduction_embed_names_mlp=check_get(block_conf, kwargs, "mg_reduction_embed_names_mlp"),
                                            mg_reduction_embed_mode=check_get(block_conf, kwargs, "mg_reduction_embed_mode"),
                                            embed_confs=check_get(block_conf, kwargs, "embed_confs"),
                                            embed_names=check_get(block_conf, kwargs, "embed_names"),
                                            embed_mode=check_get(block_conf, kwargs, "embed_mode"),
                                            with_gamma=check_get(block_conf, kwargs, "with_gamma"),
                                            omit_backtransform=check_get(block_conf, kwargs, "omit_backtransform"),
                                            mg_att_dim=check_get(block_conf, kwargs, "mg_att_dim"),
                                            mg_n_head_channels=check_get(block_conf, kwargs, "mg_n_head_channels"),
                                            level_diff_zero_linear=check_get(block_conf, kwargs, "level_diff_zero_linear"),
                                            mask_as_embedding=mask_as_embedding,
                                            layer_type=check_get(block_conf, kwargs, "layer_type"),
                                            rank=check_get(block_conf, kwargs, "rank"))  
                
            elif isinstance(block_conf, MGStackedEncoderDecoderConfig):
                block = MGNO_StackedEncoderDecoder_Block(
                    self.rcm,
                    input_levels,
                    input_dims,
                    block_conf.global_levels_output,
                    block_conf.global_levels_no,
                    block_conf.model_dims_out,
                    no_layer_settings=check_get(block_conf,kwargs,'no_layer_settings'),
                    block_type=check_get(block_conf,kwargs,'block_type'),
                    mask_as_embedding= mask_as_embedding,
                    layer_type=block_conf.layer_type if "layer_type" not in kwargs.keys() else kwargs['layer_type'],
                    no_level_step = check_get(block_conf, kwargs, "no_level_step"),
                    concat_model_dim= check_get(block_conf, kwargs, "concat_model_dim"),
                    reduction_layer_type=check_get(block_conf, kwargs, "reduction_layer_type"),
                    concat_layer_type=check_get(block_conf, kwargs, "concat_layer_type"),
                    rank=check_get(block_conf, kwargs, "rank"),
                    rank_cross=check_get(block_conf, kwargs, "rank_cross"),
                    no_rank_decay=check_get(block_conf, kwargs, "no_rank_decay"),
                    with_gamma=check_get(block_conf, kwargs, "with_gamma"),
                    embed_confs=check_get(block_conf, kwargs, "embed_confs"),
                    embed_names=check_get(block_conf, kwargs, "embed_names"),
                    embed_mode=check_get(block_conf, kwargs, "embed_mode"),
                )
                
            elif isinstance(block_conf, MGProcessingConfig):
                block = MGNO_Processing_Block(
                            input_levels,
                            block_conf.layer_settings_levels,
                            input_dims,
                            block_conf.model_dims_out,
                            self.grid_layers,
                            mask_as_embedding=mask_as_embedding)
                        
                
            self.Blocks.append(block)     

            input_dims = block.model_dims_out
            input_levels = block.output_levels

        self.out_layer = nn.Linear(input_dims[0], output_dim, bias=False)


        self.input_layer = InputLayer(self.input_dim,
                                      lifting_dim, 
                                      self.grid_layer_0, 
                                      embed_names=input_embed_names,
                                      embed_confs=input_embed_confs,
                                      embed_mode=input_embed_mode)

    def forward_(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):

        """
        Forward pass for the ICON_Transformer model.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param coords_input: Input coordinates for x
        :param coords_output: Output coordinates for position embedding.
        :param sampled_indices_batch_dict: Dictionary of sampled indices for regional models.
        :param mask: Mask for dropping cells in the input tensor.
        :return: Output tensor of shape (batch_size, num_cells, output_dim).
        """

        b,n,nh,nv,nc = x.shape[:5]
        x = x.view(b,n,-1,self.input_dim)
        b,n,nv,nc = x.shape[:4]

        x = self.input_layer(x, indices_sample=indices_sample, mask=mask, emb=emb)

        x_levels = [x]
        mask_levels = [mask]
        for k, block in enumerate(self.Blocks):
            
            coords_in = coords_input if k==0 else None
            coords_out = coords_output if k==len(self.Blocks)-1  else None
            
            # Process input through the block
            x_levels, mask_levels = block(x_levels, coords_in=coords_in, coords_out=coords_out, indices_sample=indices_sample, mask_levels=mask_levels, emb=emb)

        
        x = self.out_layer(x_levels[0])
        x = x.view(b,n,-1)

        return x
    
def check_get(block_conf, arg_dict, key):
    if key in arg_dict:
        return arg_dict[key]
    else:
        return getattr(block_conf,key)
    
class InputLayer(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 grid_layer_0,
                 embed_names=None,
                 embed_confs=None,
                 embed_mode='sum'
                ) -> None: 
      
        super().__init__()


        if embed_names is not None:
            if 'CoordinateEmbedder' in embed_names:
                self.grid_layer_0 = grid_layer_0

            self.embedder = get_embedder(embed_names, embed_confs, embed_mode=embed_mode)

            emb_dim = self.embedder.get_out_channels if self.embedder is not None else None

            self.embedding_layer = nn.Linear(emb_dim, model_dim_out*2)


        self.linear = nn.Linear(model_dim_in, model_dim_out, bias=False)

    def forward(self, x, mask=None, emb=None, indices_sample=None):
        
        if hasattr(self, 'grid_layer_0') and hasattr(self,"embedding_layer"):
            emb = add_coordinates_to_emb_dict(self.grid_layer_0, indices_layers=indices_sample["indices_layers"] if indices_sample else None, emb=emb)

        if mask is not None and hasattr(self,"embedding_layer"):
            emb = add_mask_to_emb_dict(emb, mask)

        x = self.linear(x)
        x_shape = x.shape
        if hasattr(self,"embedding_layer"):
            emb_ = self.embedder(emb).squeeze(dim=1)
            scale, shift = self.embedding_layer(emb_).chunk(2, dim=-1)
            n = scale.shape[1]
            scale, shift = scale.view(scale.shape[0],scale.shape[1],-1,*x_shape[3:]), shift.view(scale.shape[0],scale.shape[1],-1,*x_shape[3:])
            x = x * (scale + 1) + shift    

        return x