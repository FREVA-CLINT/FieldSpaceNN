import torch
import torch.nn as nn
from typing import List

from stableclimgen.src.models.mgno_transformer.mgno_base_model import MGNO_base_model

from stableclimgen.src.modules.vae.quantization import Quantization
from ..mgno_transformer.mgno_encoderdecoder_block import MGNO_StackedEncoderDecoder_Block, MGNO_EncoderDecoder_Block
from ..mgno_transformer.mgno_processing_block import MGNO_Processing_Block
from ..mgno_transformer.mgno_serial_block import Serial_NOBlock
from ..mgno_transformer.mgno_transformer_mg import InputLayer, check_get

from ..mgno_transformer.mgno_block_confs import MGProcessingConfig, MGStackedEncoderDecoderConfig, \
    MGEncoderDecoderConfig


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


class MGNO_VAE_serial(MGNO_base_model):
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

        global_levels_out_enc = [[layer_setting.get("global_level_decode", 0)
                                  for layer_setting in block_conf.layer_settings]
                                 for block_conf in encoder_block_configs]

        global_levels_out_dec = [[layer_setting.get("global_level_decode", 0)
                                  for layer_setting in block_conf.layer_settings]
                                 for block_conf in decoder_block_configs]

        global_levels_no_enc = [[layer_setting.get("global_level_no", 0)
                                 for layer_setting in block_conf.layer_settings]
                                for block_conf in encoder_block_configs]

        global_levels_no_dec = [[layer_setting.get("global_level_no", 0)
                                 for layer_setting in block_conf.layer_settings]
                                for block_conf in decoder_block_configs]

        global_levels = torch.concat((torch.tensor(global_levels_out_enc).view(-1),
                                      torch.tensor(global_levels_out_dec).view(-1),
                                      torch.tensor(global_levels_no_enc).view(-1),
                                      torch.tensor(global_levels_no_dec).view(-1),
                                      torch.tensor(0).view(-1))).unique()

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
            layer_settings = block_conf.layer_settings
            model_dims_out = block_conf.model_dims_out

            if block_conf.block_type == 'Serial':
                block = Serial_NOBlock(
                    self.rcm,
                    lifting_dim,
                    model_dims_out,
                    self.rcm.grid_layers,
                    layer_settings,
                    rotate_coordinate_system=rotate_coord_system,
                    mask_as_embedding=mask_as_embedding)
            
            self.encoder_blocks.append(block)
        self.quantization = Quantization(model_dims_out[-1], quant_config.latent_ch, quant_config.block_type, 1,
                                        **quant_config.sub_confs,
                                        grid_layer=self.grid_layers[str(block.output_level)],
                                        rotate_coord_system=rotate_coord_system,
                                        n_head_channels=quant_config.n_head_channels)

        for block_idx, block_conf in enumerate(decoder_block_configs):
            if block_conf.block_type == 'Serial':
                block = Serial_NOBlock(
                    self.rcm,
                    model_dims_out[-1],
                    block_conf.model_dims_out,
                    self.rcm.grid_layers,
                    block_conf.layer_settings,
                    input_level=block.output_level,
                    rotate_coordinate_system=rotate_coord_system,
                    mask_as_embedding=mask_as_embedding)

            self.decoder_blocks.append(block)

        self.out_layer = nn.Linear(input_dims[0], output_dim, bias=False)


    def encode(self, x, coords_input=None, indices_sample=None, mask=None, emb=None):
        b,n,nh,nv,nc = x.shape[:5]
        x = x.view(b,n,-1,self.input_dim)

        x = self.input_layer(x, indices_sample=indices_sample, mask=mask, emb=emb)

        for k, block in enumerate(self.encoder_blocks):
            coords_input = coords_input if k==0 else None
            x, mask = block(x, coords_in=coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

            if mask is not None:
                mask = mask.view(x.shape[:3])

        x = self.quantization.quantize(x.unsqueeze(1), indices_sample=indices_sample, emb=emb)
        x = x.squeeze(dim=1)
        posterior = self.quantization.get_distribution(x)
        return posterior

    def decode(self, x, coords_output=None, indices_sample=None, mask=None, emb=None):
        x = self.quantization.post_quantize(x.unsqueeze(dim=1), indices_sample=indices_sample, emb=emb)
        x = x.squeeze(dim=1)

        for k, block in enumerate(self.decoder_blocks):
            coords_out = coords_output if k==len(self.decoder_blocks)-1  else None
            x, _ = block(x, coords_out=coords_out, indices_sample=indices_sample, emb=emb)

        x = self.out_layer(x)
        return x


    def forward_(self, x, coords_input, coords_output, indices_sample=None, mask=None, emb=None):
        b,n,nh,nv,nc = x.shape[:5]
        posterior = self.encode(x, coords_input, indices_sample=indices_sample, mask=mask, emb=emb)

        if hasattr(self, "quantization"):
            z = posterior.sample()
        else:
            z = posterior
        dec = self.decode(z, coords_output, indices_sample=indices_sample, mask=mask, emb=emb)
        dec = dec.view(b,n,-1)

        return dec, posterior