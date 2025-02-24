import torch
import torch.nn as nn

from typing import List

from ...modules.neural_operator.no_blocks import MGNO_EncoderDecoder_Block, MGNO_Processing_Block
from .mgno_block_confs import MGProcessingConfig, MGEncoderDecoderConfig
from .mgno_base_model import MGNO_base_model


class MGNO_Transformer_MG(MGNO_base_model):
    def __init__(self, 
                 mgrids,
                 block_configs: List,
                 input_dim: int=1,
                 lifting_dim: int=1,
                 output_dim: int=1,
                 n_head_channels:int=16,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True,
                 p_dropout=0.,
                 mask_as_embedding = False
                 ) -> None: 
        
        self.input_dim = input_dim
        
        global_levels = []
        for block_conf in block_configs:
            if hasattr(block_conf,'global_levels_output'):
                global_levels += block_conf.global_levels_output
            if hasattr(block_conf,'global_levels_no'):
                global_levels += block_conf.global_levels_no
        
        global_levels = torch.concat((torch.tensor(global_levels).view(-1)
                                     ,torch.tensor(0).view(-1))).unique()
        
        super().__init__(mgrids, 
                         global_levels,
                         mask_as_embedding=mask_as_embedding)
        
       
        self.Blocks = nn.ModuleList()

        input_levels = [0]
        input_dims = [lifting_dim]

        for block_idx, block_conf in enumerate(block_configs):
            
            if isinstance(block_conf, MGEncoderDecoderConfig):
                block = MGNO_EncoderDecoder_Block(
                                            input_levels,
                                            input_dims,
                                            block_conf.global_levels_output,
                                            block_conf.global_levels_no,
                                            block_conf.model_dims_out,
                                            self.grid_layers,
                                            block_conf.layer_settings,
                                            rule=block_conf.rule,
                                            mg_reduction=block_conf.reduction,
                                            mg_reduction_embed_confs=block_conf.mg_reduction_embed_confs,
                                            mg_reduction_embed_names=block_conf.mg_reduction_embed_names,
                                            mg_reduction_embed_names_mlp=block_conf.mg_reduction_embed_names_mlp,
                                            mg_reduction_embed_mode=block_conf.mg_reduction_embed_mode,
                                            mg_att_dim=block_conf.mg_att_dim,
                                            mg_n_head_channels=block_conf.mg_n_head_channels,
                                            rotate_coordinate_system=rotate_coord_system)  
                
            elif isinstance(block_conf, MGProcessingConfig):
                block = MGNO_Processing_Block(
                            input_levels,
                            block_conf.layer_settings_levels,
                            input_dims,
                            block_conf.model_dims_out,
                            self.grid_layers,
                            rotate_coordinate_system=rotate_coord_system)
                        
                
            self.Blocks.append(block)     

            input_dims = block.model_dims_out
            input_levels = block.output_levels

        self.out_layer = nn.Linear(input_dims[0], output_dim, bias=False)

        self.lifting_layer = nn.Linear(input_dim, lifting_dim, bias=False) if lifting_dim>1 else nn.Identity()

        

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

        x = self.lifting_layer(x)

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
