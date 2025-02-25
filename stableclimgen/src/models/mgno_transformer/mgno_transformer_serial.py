import torch
import torch.nn as nn

from typing import List

from .mgno_block_confs import NOBlockConfig
from .mgno_base_model import MGNO_base_model
from .mgno_serial_block import Serial_NOBlock

class MGNO_Transformer_serial(MGNO_base_model):
    def __init__(self, 
                 mgrids,
                 block_configs: List[NOBlockConfig],
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

        global_levels_out = [[layer_setting.get("global_level_decode", 0) 
                              for layer_setting in block_conf.layer_settings]
                              for block_conf in block_configs]
        
        global_levels_no = [[layer_setting.get("global_level_no", 0) 
                              for layer_setting in block_conf.layer_settings]
                              for block_conf in block_configs]
        
        global_levels = torch.concat((torch.tensor(global_levels_out).view(-1) 
                                     ,torch.tensor(global_levels_no).view(-1) 
                                     ,torch.tensor(0).view(-1))).unique()
        
        super().__init__(mgrids, 
                         global_levels)
        
       
        self.Blocks = nn.ModuleList()

        for block_idx, block_conf in enumerate(block_configs):
            layer_settings = block_conf.layer_settings
            model_dims_out = block_conf.model_dims_out

            if block_conf.block_type == 'Serial':
                block = Serial_NOBlock(
                    lifting_dim,
                    model_dims_out,
                    self.grid_layers,
                    layer_settings,
                    rotate_coordinate_system=rotate_coord_system,
                    mask_as_embedding=mask_as_embedding)
                
            self.Blocks.append(block)     
        
        self.out_layer = nn.Linear(block_conf.model_dims_out[-1], output_dim, bias=False)

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

        for k, block in enumerate(self.Blocks):
            
            coords_in = coords_input if k==0 else None
            coords_out = coords_output if k==len(self.Blocks)-1  else None
            
            # Process input through the block
            x, mask = block(x, coords_in=coords_in, coords_out=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)

            if mask is not None:
                mask = mask.view(x.shape[:3])
        
        x = self.out_layer(x)
        x = x.view(b,n,-1)

        return x