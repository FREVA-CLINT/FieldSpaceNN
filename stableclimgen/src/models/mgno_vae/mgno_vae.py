from typing import List

import torch.nn as nn
from omegaconf import ListConfig

from stableclimgen.src.modules.vae.quantization import Quantization
from .confs import MGNOQuantConfig
from ..mgno_transformer.confs import MGNOEncoderDecoderConfig, MGNOStackedEncoderDecoderConfig, defaults
from ..mgno_transformer.mgno_base_model import MGNO_base_model
from ...modules.base import LinEmbLayer
from ...modules.embedding.embedder import get_embedder
from ...modules.multi_grid.confs import MGProcessingConfig
from ...modules.multi_grid.processing import MG_Block
from ...modules.neural_operator import mgno_encoder_decoder as enc_dec
from ...utils.helpers import check_get


class MGNO_VAE(MGNO_base_model):
    def __init__(self,
                 mgrids,
                 encoder_block_configs: List,
                 decoder_block_configs: List,
                 quant_config: MGNOQuantConfig,
                 in_features: int=1,
                 lift_features: int=1,
                 out_features: int=1,
                 **kwargs
                 ) -> None:

        self.max_zoom = kwargs.get("max_zoom", mgrids[-1]['zoom'])
        self.in_features = in_features
        self.out_features = out_features

        super().__init__(mgrids,
                         rotate_coord_system=kwargs.get("rotate_coord_system", False))

        # Construct blocks based on configurations
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        embedder_input = get_embedder(**check_get([kwargs, defaults], "input_embed_confs"), zoom=self.max_zoom)
        self.in_layer = LinEmbLayer(in_features,
                                    lift_features,
                                    layer_confs=check_get([kwargs, defaults], "input_layer_confs"),
                                    embedder=embedder_input)

        in_zooms = [self.max_zoom]
        in_features = [lift_features]

        for block_idx, block_conf in enumerate(encoder_block_configs):
            block = self.create_encoder_decoder_block(block_conf, in_zooms, in_features, defaults, **kwargs)
            in_features = block.out_features
            in_zooms = block.out_zooms
            
            self.encoder_blocks.append(block)

        self.bottleneck_zoom = in_zooms[-1]

        quant_embedders = get_embedder(**quant_config.embed_confs, zoom=self.bottleneck_zoom)
        self.quantization = Quantization(in_features[-1],
                                         quant_config.latent_ch,
                                         quant_config.block_type,
                                         1,
                                         **quant_config.layer_settings,
                                         embedders=quant_embedders,
                                         layer_confs=check_get([quant_config,kwargs,defaults], "layer_confs"))

        for block_idx, block_conf in enumerate(decoder_block_configs):
            block = self.create_encoder_decoder_block(block_conf, in_zooms, in_features, defaults, **kwargs)
            in_features = block.out_features
            in_zooms = block.out_zooms

            self.decoder_blocks.append(block)

        embedder_output = get_embedder(**check_get([kwargs, defaults], "output_embed_confs"), zoom=self.max_zoom)

        self.out_layer = LinEmbLayer(
            in_features[0] if isinstance(in_features, list) or isinstance(in_features, ListConfig) else in_features,
            out_features,
            layer_confs=check_get([kwargs, defaults], "input_layer_confs"),
            embedder=embedder_output)


    def encode(self, x, coords_input, sample_dict={}, mask=None, emb=None):
        b, nt, n, nv, nc = x.shape[:5]
        x = x.view(b, nt, n, -1, self.in_features)

        x = self.in_layer(x, sample_dict=sample_dict, emb=emb)

        x_zooms = {int(sample_dict['zoom'][0]): x} if 'zoom' in sample_dict.keys() else {self.max_zoom: x}
        mask_zooms = {int(sample_dict['zoom'][0]): mask} if 'zoom' in sample_dict.keys() else {self.max_zoom: mask}

        for k, block in enumerate(self.encoder_blocks):
            x_zooms = block(x_zooms, sample_dict=sample_dict, mask_zooms=mask_zooms, emb=emb)

        x = x_zooms[self.bottleneck_zoom]

        x = self.quantization.quantize(x, sample_dict=sample_dict, emb=emb)
        posterior = self.quantization.get_distribution(x)
        return posterior

    def decode(self, x, coords_output, sample_dict={}, mask=None, emb=None):
        x = self.quantization.post_quantize(x, sample_dict=sample_dict, emb=emb)

        x_zooms = {self.bottleneck_zoom: x}
        mask_zooms = {self.bottleneck_zoom: mask}

        for k, block in enumerate(self.decoder_blocks):
            x_zooms = block(x_zooms, sample_dict=sample_dict, mask_zooms=mask_zooms, emb=emb)

        x = x_zooms[int(sample_dict['zoom'][0]) if sample_dict else self.max_zoom]

        x = self.out_layer(x, emb=emb, sample_dict=sample_dict)
        return x


    def forward(self, x, coords_input, coords_output, sample_dict={}, mask=None, emb=None, residual=0):
        b, nt, n, nv, nc = x.shape[:5]

        assert nc == self.in_features, f" the input has {nc} features, which doesnt match the numnber of specified input_features {self.in_features}"
        assert nc == self.out_features, f" the input has {nc} features, which doesnt match the numnber of specified out_features {self.out_features}"

        posterior = self.encode(x, coords_input, sample_dict=sample_dict, mask=mask, emb=emb)

        z = posterior.sample()

        dec = self.decode(z, coords_output, sample_dict=sample_dict, mask=mask, emb=emb)
        dec = dec + residual

        dec = dec.view(b,nt,n,nv,-1)
        return dec, posterior

    def create_encoder_decoder_block(self, block_conf, in_zooms, in_features, defaults, **kwargs):
        if isinstance(block_conf, MGNOEncoderDecoderConfig):

            block = enc_dec.MGNO_EncoderDecoder_Block(
                self.rcm,
                in_zooms,
                in_features,
                block_conf.out_zooms,
                block_conf.no_zooms,
                block_conf.out_features,
                rule=block_conf.rule,
                no_layer_settings=check_get([block_conf, kwargs, defaults], 'no_layer_settings'),
                block_type=check_get([block_conf, kwargs, defaults], 'block_type'),
                with_gamma=check_get([block_conf, kwargs, defaults], "with_gamma"),
                embed_confs=check_get([block_conf, kwargs, defaults], "embed_confs"),
                omit_backtransform=check_get([block_conf, kwargs, defaults], "omit_backtransform"),
                layer_confs=check_get([block_conf, kwargs, defaults], "layer_confs"),
                concat_prev=check_get([block_conf, kwargs, defaults], "concat_prev"))


        elif isinstance(block_conf, MGNOStackedEncoderDecoderConfig):
            block = enc_dec.MGNO_StackedEncoderDecoder_Block(
                self.rcm,
                in_zooms,
                in_features,
                block_conf.out_zooms,
                block_conf.no_zooms,
                block_conf.out_features,
                no_zoom_step=check_get([block_conf, kwargs, defaults], 'no_zoom_step'),
                no_layer_settings=check_get([block_conf, kwargs, defaults], 'no_layer_settings'),
                block_type=check_get([block_conf, kwargs, defaults], 'block_type'),
                with_gamma=check_get([block_conf, kwargs, defaults], "with_gamma"),
                embed_confs=check_get([block_conf, kwargs, defaults], "embed_confs"),
                layer_confs=check_get([block_conf, kwargs, defaults], "layer_confs"),
                concat_prev=check_get([block_conf, kwargs, defaults], "concat_prev"))

        elif isinstance(block_conf, MGProcessingConfig):
            layer_settings = block_conf.layer_settings
            layer_settings['layer_confs'] = check_get([block_conf, kwargs, defaults], "layer_confs")

            block = MG_Block(
                self.rcm.grid_layers,
                in_zooms,
                layer_settings,
                in_features,
                block_conf.out_features)

        return block