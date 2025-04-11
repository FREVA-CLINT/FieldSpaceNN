import torch
import torch.nn as nn

from typing import List

from .mgno_encoderdecoder_block import MGNO_EncoderDecoder_Block, MGNO_StackedEncoderDecoder_Block
from .mgno_processing_block import MGNO_Processing_Block

from .mgno_block_confs import MGProcessingConfig, MGEncoderDecoderConfig, MGStackedEncoderDecoderConfig, defaults
from .mgno_base_model import MGNO_base_model

from ...modules.neural_operator.no_blocks import get_lin_layer, DenseLayer

from ...modules.neural_operator.no_helpers import get_embedder
from ...modules.neural_operator.no_helpers import add_coordinates_to_emb_dict,add_mask_to_emb_dict

class MGNO_Transformer_MG(MGNO_base_model):
    def __init__(self, 
                 mgrids,
                 block_configs: List,
                 input_dim: int=1,
                 lifting_dim: int=1,
                 output_dim: int=1,
                 **kwargs
                 ) -> None: 
        
        self.input_dim = input_dim + kwargs.get("concat_interp", False)
        
        predict_var = kwargs.get("predict_var", defaults['predict_var'])
        
        super().__init__(mgrids,
                         rotate_coord_system=kwargs.get("rotate_coord_system", False),
                         interpolate_input=kwargs.get("interpolate_input", False),
                         density_embedder=kwargs.get("density_embedder", False),
                         interpolator_settings=kwargs.get("interpolator_settings", None),
                         concat_interp=kwargs.get("concat_interp", False))
        
        mask_as_embedding = kwargs.get("mask_as_embedding", False)

        if predict_var:
            output_dim = output_dim * 2
            self.activation_var = nn.Softplus()

        self.output_dim = output_dim
        self.predict_var = predict_var

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
                                            no_layer_settings=check_get(block_conf,kwargs,defaults,'no_layer_settings'),
                                            block_type=check_get(block_conf,kwargs,defaults,'block_type'),
                                            mg_reduction=check_get(block_conf, kwargs,defaults, "mg_reduction"),
                                            mg_reduction_embed_confs=check_get(block_conf, kwargs,defaults, "mg_reduction_embed_confs"),
                                            mg_reduction_embed_names=check_get(block_conf, kwargs,defaults, "mg_reduction_embed_names"),
                                            mg_reduction_embed_names_mlp=check_get(block_conf, kwargs,defaults, "mg_reduction_embed_names_mlp"),
                                            mg_reduction_embed_mode=check_get(block_conf, kwargs,defaults, "mg_reduction_embed_mode"),
                                            embed_confs=check_get(block_conf, kwargs,defaults, "embed_confs"),
                                            embed_names=check_get(block_conf, kwargs,defaults, "embed_names"),
                                            embed_mode=check_get(block_conf, kwargs,defaults, "embed_mode"),
                                            with_gamma=check_get(block_conf, kwargs,defaults, "with_gamma"),
                                            omit_backtransform=check_get(block_conf, kwargs,defaults, "omit_backtransform"),
                                            mg_att_dim=check_get(block_conf, kwargs,defaults, "mg_att_dim"),
                                            mg_n_head_channels=check_get(block_conf, kwargs,defaults, "mg_n_head_channels"),
                                            level_diff_zero_linear=check_get(block_conf, kwargs,defaults, "level_diff_zero_linear"),
                                            mask_as_embedding=mask_as_embedding,
                                            layer_type=check_get(block_conf, kwargs,defaults, "layer_type"),
                                            rank=check_get(block_conf, kwargs,defaults, "rank"),
                                            n_vars_total=check_get(block_conf, kwargs,defaults, "n_vars_total"),
                                            rank_vars=check_get(block_conf, kwargs,defaults, "rank_vars"),
                                            factorize_vars=check_get(block_conf, kwargs,defaults, "factorize_vars"))  
                
            elif isinstance(block_conf, MGStackedEncoderDecoderConfig):
                block = MGNO_StackedEncoderDecoder_Block(
                    self.rcm,
                    input_levels,
                    input_dims,
                    block_conf.global_levels_output,
                    block_conf.global_levels_no,
                    block_conf.model_dims_out,
                    no_layer_settings=check_get(block_conf,kwargs,defaults,'no_layer_settings'),
                    block_type=check_get(block_conf,kwargs,defaults,'block_type'),
                    mask_as_embedding= mask_as_embedding,
                    layer_type=block_conf.layer_type if "layer_type" not in kwargs.keys() else kwargs['layer_type'],
                    no_level_step = check_get(block_conf, kwargs,defaults, "no_level_step"),
                    concat_model_dim= check_get(block_conf, kwargs,defaults, "concat_model_dim"),
                    reduction_layer_type=check_get(block_conf, kwargs,defaults, "reduction_layer_type"),
                    concat_layer_type=check_get(block_conf, kwargs,defaults, "concat_layer_type"),
                    rank=check_get(block_conf, kwargs,defaults, "rank"),
                    rank_cross=check_get(block_conf, kwargs,defaults, "rank_cross"),
                    no_rank_decay=check_get(block_conf, kwargs,defaults, "no_rank_decay"),
                    with_gamma=check_get(block_conf, kwargs,defaults, "with_gamma"),
                    embed_confs=check_get(block_conf, kwargs,defaults, "embed_confs"),
                    embed_names=check_get(block_conf, kwargs,defaults, "embed_names"),
                    embed_mode=check_get(block_conf, kwargs,defaults, "embed_mode"),
                    n_head_channels=check_get(block_conf, kwargs,defaults, "n_head_channels"),
                    p_dropout=check_get(block_conf, kwargs,defaults, "p_dropout"),
                    seq_level=check_get(block_conf, kwargs,defaults, "seq_level"),
                    n_vars_total=check_get(block_conf, kwargs,defaults, "n_vars_total"),
                    rank_vars=check_get(block_conf, kwargs,defaults, "rank_vars"),
                    factorize_vars=check_get(block_conf, kwargs,defaults, "factorize_vars")
                )
                
            elif isinstance(block_conf, MGProcessingConfig):
                block = MGNO_Processing_Block(
                            input_levels,
                            block_conf.layer_settings_levels,
                            input_dims,
                            block_conf.model_dims_out,
                            self.grid_layers,
                            mask_as_embedding=mask_as_embedding,
                            n_vars_total=check_get(block_conf, kwargs,defaults, "n_vars_total"),
                            rank_vars=check_get(block_conf, kwargs,defaults, "rank_vars"),
                            factorize_vars=check_get(block_conf, kwargs,defaults, "factorize_vars"))
                        
                
            self.Blocks.append(block)     

            input_dims = block.model_dims_out
            input_levels = block.output_levels

       # self.out_layer = nn.Linear(input_dims[0], output_dim, bias=False)

        self.out_layer = get_lin_layer(input_dims[0], 
                                       output_dim, 
                                       n_vars_total=kwargs.get('n_vars_total',1), 
                                       rank_vars=kwargs.get('rank_vars',4), 
                                       factorize_vars=kwargs.get('factorize_vars',False), 
                                       bias=False)

        self.input_layer = InputLayer(self.input_dim,
                                      lifting_dim, 
                                      self.grid_layer_0, 
                                      embed_names=kwargs.get('input_embed_names',defaults['input_embed_names']),
                                      embed_confs=kwargs.get('input_embed_confs',defaults['input_embed_confs']),
                                      embed_mode=kwargs.get('input_embed_mode',defaults['input_embed_mode']),
                                      n_vars_total = kwargs.get('n_vars_total',1),
                                      rank_vars = kwargs.get('rank_vars',4),
                                      factorize_vars= kwargs.get('factorize_vars',False),
                                      with_gamma=kwargs.get('with_input_gamma',False))

        self.learn_residual = kwargs.get("learn_residual", False)

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

        if self.learn_residual:
            x_res = x
            # assume gauss
           # if self.predict_var:
           #     x_res_var = x_res.abs() * (1-(emb['DensityEmbedder']))

        x = self.input_layer(x, indices_sample=indices_sample, mask=mask, emb=emb)

        x_levels = [x]
        mask_levels = [mask]
        for k, block in enumerate(self.Blocks):
            
            coords_in = coords_input if k==0 else None
            coords_out = coords_output if k==len(self.Blocks)-1  else None
            
            # Process input through the block
            x_levels, mask_levels = block(x_levels, coords_in=coords_in, coords_out=coords_out, indices_sample=indices_sample, mask_levels=mask_levels, emb=emb)

        if isinstance(self.out_layer, DenseLayer):
            x = self.out_layer(x_levels[0], emb=emb)
        else: 
            x = self.out_layer(x_levels[0])

        x = x.view(b,n,nv,-1)

        if self.learn_residual and not self.predict_var:
            x = x_res.view(x.shape) + x

        elif self.predict_var and self.learn_residual:
            x, x_var = x.chunk(2,dim=-1) 
            x = x_res.view(x.shape) + x
            x = torch.concat((x, self.activation_var(x_var)),dim=-1)

        elif self.predict_var:
            x, x_var = x.chunk(2,dim=-1) 
            x = torch.concat((x, self.activation_var(x_var)),dim=-1)

        return x
    
def check_get(block_conf, arg_dict, defaults, key):
    if hasattr(block_conf,key):
        return getattr(block_conf, key)
    elif key in arg_dict:
        return arg_dict[key]
    elif key in defaults:
        return defaults[key]
    else:
        raise KeyError(f"Key '{key}' not found block_conf, model arguments and defaults")
    
class InputLayer(nn.Module):
  
    def __init__(self,
                 model_dim_in,
                 model_dim_out,
                 grid_layer_0,
                 embed_names=None,
                 embed_confs=None,
                 embed_mode='sum',
                 n_vars_total=1,
                 rank_vars=4,
                 factorize_vars=False,
                 with_gamma=False
                ) -> None: 
      
        super().__init__()


        if embed_names is not None:
            if 'CoordinateEmbedder' in embed_names:
                self.grid_layer_0 = grid_layer_0

            self.embedder = get_embedder(embed_names, embed_confs, embed_mode=embed_mode)

            emb_dim = self.embedder.get_out_channels if self.embedder is not None else None

            self.embedding_layer = nn.Linear(emb_dim, model_dim_out*2)

            if with_gamma:
                self.gamma1 = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
                self.gamma2 = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)

        self.linear = nn.Linear(model_dim_in, model_dim_out, bias=False)

        self.linear = get_lin_layer(model_dim_in, model_dim_out, n_vars_total=n_vars_total, rank_vars=rank_vars, factorize_vars=factorize_vars, bias=False)

        
    def forward(self, x, mask=None, emb=None, indices_sample=None):
        
        if hasattr(self, 'grid_layer_0') and hasattr(self,"embedding_layer"):
            emb = add_coordinates_to_emb_dict(self.grid_layer_0, indices_layers=indices_sample["indices_layers"] if indices_sample else None, emb=emb)

        if mask is not None and hasattr(self,"embedding_layer"):
            emb = add_mask_to_emb_dict(emb, mask)

        if isinstance(self.linear, DenseLayer):
            x = self.linear(x, emb=emb)
        else: 
            x = self.linear(x)

        x_shape = x.shape
        if hasattr(self,"embedding_layer"):
            emb_ = self.embedder(emb).squeeze(dim=1)
            scale, shift = self.embedding_layer(emb_).chunk(2, dim=-1)
            n = scale.shape[1]
            scale, shift = scale.view(scale.shape[0],scale.shape[1],-1,*x_shape[3:]), shift.view(scale.shape[0],scale.shape[1],-1,*x_shape[3:])

            if hasattr(self, "gamma1"):
                x = x * (self.gamma1 * scale + 1) + self.gamma2 * shift
            else:
                x = x * (scale + 1) + shift

        return x