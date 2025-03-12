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
                 n_head_channels:int=16,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True,
                 p_dropout=0.,
                 mask_as_embedding = False
                 ) -> None: 
        
        self.input_dim = input_dim
        
        global_levels = torch.tensor(0).view(-1)
        for block_conf in block_configs:
            if hasattr(block_conf,'global_levels_output'):
                global_levels = torch.concat((global_levels, torch.tensor(block_conf.global_levels_output).view(-1))) 
            if hasattr(block_conf,'global_levels_no'):
                #if not isinstance(block_conf.global_levels_no, list):
                global_levels = torch.concat((global_levels, torch.tensor(block_conf.global_levels_output).view(-1)))

        global_levels_max = torch.concat((torch.tensor(global_levels).view(-1)
                                     ,torch.tensor(0).view(-1))).max()
        
        global_levels = torch.arange(global_levels_max+1)
        
        super().__init__(mgrids, 
                         global_levels)
        
       
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
                                            block_conf.layer_settings,
                                            rule=block_conf.rule,
                                            mg_reduction=block_conf.reduction,
                                            mg_reduction_embed_confs=block_conf.mg_reduction_embed_confs,
                                            mg_reduction_embed_names=block_conf.mg_reduction_embed_names,
                                            mg_reduction_embed_names_mlp=block_conf.mg_reduction_embed_names_mlp,
                                            mg_reduction_embed_mode=block_conf.mg_reduction_embed_mode,
                                            mg_att_dim=block_conf.mg_att_dim,
                                            mg_n_head_channels=block_conf.mg_n_head_channels,
                                            level_diff_zero_linear=block_conf.level_diff_zero_linear,
                                            mask_as_embedding=mask_as_embedding)  
                
            elif isinstance(block_conf, MGStackedEncoderDecoderConfig):
                block = MGNO_StackedEncoderDecoder_Block(
                    self.rcm,
                    input_levels,
                    input_dims,
                    block_conf.global_levels_output,
                    block_conf.global_levels_no,
                    block_conf.model_dims_out,
                    block_conf.layer_settings,
                    mask_as_embedding= mask_as_embedding
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
    
class EmbedLayer(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 grid_layer_0,
                 embed_names=None,
                 embed_confs=None,
                ) -> None: 
      
        super().__init__()
        self.grid_layer_0 = grid_layer_0
        if embed_names is not None:
            embedder = get_embedder(embed_names, embed_confs, embed_mode="sum")
            emb_dim = embedder.get_out_channels if embedder is not None else None
            self.embedding_layer = torch.nn.Linear(emb_dim, model_dim_out*2)

        self.linear = nn.Linear(model_dim_in, model_dim_out, bias=True)

    def forward(self, x, mask=None, emb=None):
        
        emb = add_coordinates_to_emb_dict()
        x_shape = x.shape
        if hasattr(self,"embedding_layer"):
            emb_ = self.embedder(emb).squeeze(dim=1)
            scale, shift = self.embedding_layer(emb_).chunk(2, dim=-1)
            n = scale.shape[1]
            scale, shift = scale.view(scale.shape[0],scale.shape[1],-1,*x_shape[3:]), shift.view(scale.shape[0],scale.shape[1],-1,*x_shape[3:])
            x = self.linear(x) * (scale + 1) + shift    
        else:
            x = self.linear(x)