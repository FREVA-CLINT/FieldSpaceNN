import torch
import torch.nn as nn
from typing import List, Dict

from stableclimgen.src.models.mgno_transformer.mgno_base_model import MGNO_base_model

from stableclimgen.src.modules.vae.quantization import Quantization
from ..mgno_transformer.mgno_encoderdecoder_block import MGNO_StackedEncoderDecoder_Block, MGNO_EncoderDecoder_Block
from ..mgno_transformer.mgno_processing_block import MGNO_Processing_Block
from ..mgno_transformer.mgno_transformer_mg import InputLayer, check_get

from ..mgno_transformer.mgno_block_confs import MGProcessingConfig, MGStackedEncoderDecoderConfig, \
    MGEncoderDecoderConfig
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
            no_layer_settings=check_get(block_conf, kwargs, 'no_layer_settings'),
            block_type=check_get(block_conf, kwargs, 'block_type'),
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
            rcm,
            input_levels,
            input_dims,
            block_conf.global_levels_output,
            block_conf.global_levels_no,
            block_conf.model_dims_out,
            no_layer_settings=check_get(block_conf, kwargs, 'no_layer_settings'),
            block_type=check_get(block_conf, kwargs, 'block_type'),
            mask_as_embedding=mask_as_embedding,
            layer_type=block_conf.layer_type if "layer_type" not in kwargs.keys() else kwargs['layer_type'],
            no_level_step=check_get(block_conf, kwargs, "no_level_step"),
            concat_model_dim=check_get(block_conf, kwargs, "concat_model_dim"),
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
                 interpolate_input_residual=False,
                 residual_interpolator_settings: Dict = None,
                 **kwargs
                 ) -> None:
        global_levels = torch.tensor(0).view(-1)
        for block_conf in encoder_block_configs:
            if hasattr(block_conf, 'global_levels_output'):
                global_levels = torch.concat((global_levels, torch.tensor(block_conf.global_levels_output).view(-1)))
            if hasattr(block_conf, 'global_levels_no'):
                # if not isinstance(block_conf.global_levels_no, list):
                global_levels = torch.concat((global_levels, torch.tensor(block_conf.global_levels_no).view(-1)))

        global_levels_max = torch.concat((torch.tensor(global_levels).view(-1)
                                          , torch.tensor(0).view(-1))).max()

        global_levels = torch.arange(global_levels_max + 1)

        super().__init__(mgrids,
                         global_levels,
                         rotate_coord_system=rotate_coord_system,
                         interpolate_input=kwargs.get("interpolate_input",False),
                         interpolator_settings=kwargs.get("interpolator_settings",None))

        self.input_dim = input_dim

        # Construct blocks based on configurations
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        self.input_layer = InputLayer(input_dim,
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

        if interpolate_input_residual:
            self.residual_interpolator_down = Interpolator(self.grid_layers,
                                             residual_interpolator_settings.get("search_level", 3),
                                             0,
                                                           bottleneck_level,
                                             residual_interpolator_settings.get("precompute", True),
                                             residual_interpolator_settings.get("nh_inter", 3),
                                             residual_interpolator_settings.get("power", 1)
                                             )
            self.residual_interpolator_up = Interpolator(self.grid_layers,
                                                           residual_interpolator_settings.get("search_level", 3),
                                                           bottleneck_level,
                                                           0,
                                                           residual_interpolator_settings.get("precompute", True),
                                                           residual_interpolator_settings.get("nh_inter", 3),
                                                           residual_interpolator_settings.get("power", 1)
                                                           )

        self.interpolate_input_residual = interpolate_input_residual

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

        x = self.quantization.quantize(x.unsqueeze(1), indices_sample=indices_sample, emb=emb)
        x = x.squeeze(dim=1)
        posterior = self.quantization.get_distribution(x)
        return posterior, mask_levels[0]

    def decode(self, x, coords_output=None, indices_sample=None, mask=None, emb=None):
        x = self.quantization.post_quantize(x.unsqueeze(dim=1), indices_sample=indices_sample, emb=emb)
        x = x.squeeze(dim=1)

        x_levels = [x]
        mask_levels = [None]
        for k, block in enumerate(self.decoder_blocks):
            coords_out = coords_output if k==len(self.decoder_blocks)-1  else None
            x_levels, _ = block(x_levels, coords_out=coords_out, indices_sample=indices_sample, mask_levels=mask_levels, emb=emb)

        x = self.out_layer(x_levels[0])
        return x


    def forward_(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):
        b,n,nh,nv,nc = x.shape[:5]
        interp_x = 0
        if self.interpolate_input_residual:
            interp_x, _ = self.residual_interpolator_down(x,
                                                      calc_density=False,
                                                      indices_sample=indices_sample)
            interp_x, _ = self.residual_interpolator_up(interp_x.unsqueeze(dim=-2),
                                                      calc_density=False,
                                                      indices_sample=indices_sample)

        posterior, mask = self.encode(x, coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        if hasattr(self, "quantization"):
            z = posterior.sample()
        else:
            z = posterior
        dec = self.decode(z, coords_output, indices_sample=indices_sample, mask=mask, emb=emb) + interp_x
        dec = dec.view(b,n,-1)
        return dec, posterior