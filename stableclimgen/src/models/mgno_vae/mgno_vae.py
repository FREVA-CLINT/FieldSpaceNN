import torch
import torch.nn as nn
from typing import List, Dict

from stableclimgen.src.modules.vae.quantization import Quantization
from ..mgno_transformer.mgno_encoderdecoder_block import MGNO_StackedEncoderDecoder_Block, MGNO_EncoderDecoder_Block
from ..mgno_transformer.mgno_processing_block import MGNO_Processing_Block
from ..mgno_transformer.mgno_base_model import InputLayer, check_get, MGNO_base_model

from ..mgno_transformer.mgno_block_confs import MGProcessingConfig, MGStackedEncoderDecoderConfig, \
    MGEncoderDecoderConfig, defaults
from ...modules.icon_grids.grid_layer import Interpolator


class QuantConfig:
    """
    Configuration class for defining quantization parameters in the VAE model.

    :param z_ch: Number of latent channels in the quantized representation.
    :param latent_ch: Number of latent channels in the bottleneck.
    :param block_type: Block type used in quantization (e.g., 'conv' or 'resnet').
    """

    def __init__(self, latent_ch: List[int], n_head_channels: List[int], block_type: str, sub_confs: dict):
        self.latent_ch = latent_ch
        self.block_type = block_type
        self.sub_confs = sub_confs
        self.n_head_channels = n_head_channels


def create_encoder_decoder_block(rcm, input_levels, input_dims, mask_as_embedding, block_conf, grid_layers, **kwargs):
    if isinstance(block_conf, MGEncoderDecoderConfig):
        block = MGNO_EncoderDecoder_Block(
            rcm,
            input_levels,
            input_dims,
            block_conf.global_levels_output,
            block_conf.global_levels_no,
            block_conf.model_dims_out,
            rule=block_conf.rule,
            no_layer_settings=check_get(block_conf, kwargs, defaults, 'no_layer_settings'),
            block_type=check_get(block_conf, kwargs, defaults, 'block_type'),
            mg_reduction=check_get(block_conf, kwargs, defaults, "mg_reduction"),
            mg_reduction_embed_confs=check_get(block_conf, kwargs, defaults, "mg_reduction_embed_confs"),
            mg_reduction_embed_names=check_get(block_conf, kwargs, defaults, "mg_reduction_embed_names"),
            mg_reduction_embed_names_mlp=check_get(block_conf, kwargs, defaults, "mg_reduction_embed_names_mlp"),
            mg_reduction_embed_mode=check_get(block_conf, kwargs, defaults, "mg_reduction_embed_mode"),
            embed_confs=check_get(block_conf, kwargs, defaults, "embed_confs"),
            embed_names=check_get(block_conf, kwargs, defaults, "embed_names"),
            embed_mode=check_get(block_conf, kwargs, defaults, "embed_mode"),
            with_gamma=check_get(block_conf, kwargs, defaults, "with_gamma"),
            omit_backtransform=check_get(block_conf, kwargs, defaults, "omit_backtransform"),
            mg_att_dim=check_get(block_conf, kwargs, defaults, "mg_att_dim"),
            mg_n_head_channels=check_get(block_conf, kwargs, defaults, "mg_n_head_channels"),
            level_diff_zero_linear=check_get(block_conf, kwargs, defaults, "level_diff_zero_linear"),
            mask_as_embedding=mask_as_embedding,
            layer_type=check_get(block_conf, kwargs, defaults, "layer_type"),
            rank=check_get(block_conf, kwargs, defaults, "rank"),
            n_vars_total=check_get(block_conf, kwargs, defaults, "n_vars_total"),
            rank_vars=check_get(block_conf, kwargs, defaults, "rank_vars"),
            factorize_vars=check_get(block_conf, kwargs, defaults, "factorize_vars"))

    elif isinstance(block_conf, MGStackedEncoderDecoderConfig):
        block = MGNO_StackedEncoderDecoder_Block(
            rcm,
            input_levels,
            input_dims,
            block_conf.global_levels_output,
            block_conf.global_levels_no,
            block_conf.model_dims_out,
            no_layer_settings=check_get(block_conf, kwargs, defaults, 'no_layer_settings'),
            block_type=check_get(block_conf, kwargs, defaults, 'block_type'),
            mask_as_embedding=mask_as_embedding,
            layer_type=block_conf.layer_type if "layer_type" not in kwargs.keys() else kwargs['layer_type'],
            no_level_step=check_get(block_conf, kwargs, defaults, "no_level_step"),
            concat_model_dim=check_get(block_conf, kwargs, defaults, "concat_model_dim"),
            reduction_layer_type=check_get(block_conf, kwargs, defaults, "reduction_layer_type"),
            concat_layer_type=check_get(block_conf, kwargs, defaults, "concat_layer_type"),
            rank=check_get(block_conf, kwargs, defaults, "rank"),
            rank_cross=check_get(block_conf, kwargs, defaults, "rank_cross"),
            no_rank_decay=check_get(block_conf, kwargs, defaults, "no_rank_decay"),
            with_gamma=check_get(block_conf, kwargs, defaults, "with_gamma"),
            embed_confs=check_get(block_conf, kwargs, defaults, "embed_confs"),
            embed_names=check_get(block_conf, kwargs, defaults, "embed_names"),
            embed_mode=check_get(block_conf, kwargs, defaults, "embed_mode"),
            n_head_channels=check_get(block_conf, kwargs, defaults, "n_head_channels"),
            p_dropout=check_get(block_conf, kwargs, defaults, "p_dropout"),
            seq_level=check_get(block_conf, kwargs, defaults, "seq_level"),
            n_vars_total=check_get(block_conf, kwargs, defaults, "n_vars_total"),
            rank_vars=check_get(block_conf, kwargs, defaults, "rank_vars"),
            factorize_vars=check_get(block_conf, kwargs, defaults, "factorize_vars")
        )

    else:
        assert isinstance(block_conf, MGProcessingConfig)
        block = MGNO_Processing_Block(
            input_levels,
            block_conf.layer_settings_levels,
            input_dims,
            block_conf.model_dims_out,
            grid_layers,
            mask_as_embedding=mask_as_embedding)

    return block


class MGNO_VAE(MGNO_base_model):
    def __init__(self, 
                 mgrids,
                 encoder_block_configs: List,
                 decoder_block_configs: List,
                 quant_config: QuantConfig,
                 input_dim: int=1,
                 lifting_dim: int=1,
                 output_dim: int=1,
                 n_vars_total:int=1,
                 rotate_coord_system: bool=True,
                 mask_as_embedding=False,
                 input_embed_names=None,
                 input_embed_confs=None,
                 input_embed_mode='sum',
                 **kwargs
                 ) -> None:

        super().__init__(mgrids,rotate_coord_system=rotate_coord_system)

        self.input_dim = input_dim

        # Construct blocks based on configurations
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        self.input_layer = InputLayer(self.input_dim,
                                      lifting_dim,
                                      self.grid_layer_0,
                                      embed_names=input_embed_names,
                                      embed_confs=input_embed_confs,
                                      embed_mode=input_embed_mode)

        input_levels = [0]
        input_dims = [lifting_dim]

        for block_idx, block_conf in enumerate(encoder_block_configs):
            block = create_encoder_decoder_block(
                self.rcm,
                input_levels,
                input_dims,
                mask_as_embedding,
                block_conf,
                self.grid_layers,
                **kwargs
            )
            input_dims = block.model_dims_out
            input_levels = block.output_levels
            
            self.encoder_blocks.append(block)

        bottleneck_level = input_levels[-1]

        self.quantization = Quantization(input_dims[-1], quant_config.latent_ch, quant_config.block_type, 1,
                                        **quant_config.sub_confs,
                                        grid_layer=self.grid_layers[str(bottleneck_level)],
                                        rotate_coord_system=rotate_coord_system,
                                        n_head_channels=quant_config.n_head_channels)

        for block_idx, block_conf in enumerate(decoder_block_configs):
            block = create_encoder_decoder_block(
                self.rcm,
                input_levels,
                input_dims,
                mask_as_embedding,
                block_conf,
                self.grid_layers,
                **kwargs
            )
            input_dims = block.model_dims_out
            input_levels = block.output_levels

            self.decoder_blocks.append(block)

        self.out_layer = nn.Linear(input_dims[0], output_dim, bias=False)


    def encode(self, x, coords_input=None, indices_sample=None, mask=None, emb=None):
        b,n,nh,nv,nc = x.shape[:5]
        x = x.view(b,n,-1,self.input_dim)

        x = self.input_layer(x, indices_sample=indices_sample, mask=mask, emb=emb)

        x_levels = [x]
        mask_levels = [mask]
        for k, block in enumerate(self.encoder_blocks):
            coords_input = coords_input if k==0 else None
            x_levels, mask_levels = block(x_levels, coords_in=coords_input, indices_sample=indices_sample, mask_levels=mask_levels, emb=emb)

        x = x_levels[0]

        x = self.quantization.quantize(x, indices_sample=indices_sample, emb=emb)
        posterior = self.quantization.get_distribution(x)
        return posterior, mask_levels[0]

    def decode(self, x, coords_output=None, indices_sample=None, mask=None, emb=None):
        x = self.quantization.post_quantize(x, indices_sample=indices_sample, emb=emb)

        x_levels = [x]
        mask_levels = [None]
        for k, block in enumerate(self.decoder_blocks):
            coords_out = coords_output if k==len(self.decoder_blocks)-1  else None
            x_levels, _ = block(x_levels, coords_out=coords_out, indices_sample=indices_sample, mask_levels=mask_levels, emb=emb)

        x = self.out_layer(x_levels[0])
        return x


    def forward(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None, residual=None):
        b,n,nh,nv,nc = x.shape[:5]

        posterior, mask = self.encode(x, coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        if hasattr(self, "quantization"):
            z = posterior.sample()
        else:
            z = posterior
        dec = self.decode(z, coords_output, indices_sample=indices_sample, mask=mask, emb=emb) + (residual if torch.is_tensor(residual) else 0)
        dec = dec.view(b,n,-1)
        return dec, posterior