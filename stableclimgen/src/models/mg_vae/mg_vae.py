from typing import List,Dict

import torch
import torch.nn as nn
from stableclimgen.src.modules.distributions.distributions import DiagonalGaussianDistribution, DiracDistribution

from ..mg_transformer.confs import defaults
from ..mg_transformer.mg_base_model import MG_base_model
from .confs import MGQuantConfig
from ..mg_transformer.mg_transformer import DiffDecoder
from ...modules.embedding.embedder import get_embedder
from ...modules.multi_grid.confs import MGProcessingConfig, MGSelfProcessingConfig, MGConservativeConfig, \
    MGCoordinateEmbeddingConfig, MGFieldAttentionConfig
from ...modules.multi_grid.mg_base import ConservativeLayer, MGEmbedding, get_mg_embeddings, MFieldLayer
from ...modules.multi_grid.processing import MG_SingleBlock, MG_MultiBlock
from ...modules.vae.quantization import Quantization
from ...utils.helpers import check_get


class MG_VAE(MG_base_model):
    def __init__(self, 
                 mgrids,
                 in_zooms: List,
                 encoder_block_configs: List,
                 decoder_block_configs: List,
                 quant_config: MGQuantConfig,
                 in_features: int=1,
                 out_features: int=1,
                 mg_emb_confs: dict={},
                 distribution: str = "gaussian",
                 decoder_settings = {},
                 sample_gamma = False,
                 **kwargs
                 ) -> None: 
        
        
        super().__init__(mgrids)
        self.max_zoom = max(in_zooms)
        self.in_zooms = in_zooms

        self.in_features = in_features 
        predict_var = kwargs.get("predict_var", defaults['predict_var'])

        if predict_var:
            out_features = out_features * 2
            self.activation_var = nn.Softplus()

        self.out_features = out_features
        self.predict_var = predict_var
        self.distribution = distribution

        if len(mg_emb_confs)>0:
            self.mg_emeddings = get_mg_embeddings(mg_emb_confs, self.grid_layers)
        else:
            self.mg_emeddings = None

        # Construct blocks based on configurations
        self.encoder_blocks = nn.ModuleDict()
        self.decoder_blocks = nn.ModuleDict()

        in_features = [in_features]*len(in_zooms)

        for block_key, block_conf in encoder_block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = self.create_encoder_decoder_block(block_conf, in_zooms, in_features, **kwargs)
                
            self.encoder_blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        self.bottleneck_zooms = in_zooms

        if sample_gamma:
            self.gammas = nn.ParameterDict()
            for i, zoom in enumerate(self.bottleneck_zooms):
                self.gammas[str(zoom)] = nn.Parameter(torch.ones(quant_config.out_features[i]) * 1e-6, requires_grad=True)
        else:
            self.gammas = None

        quant_out_feat = [(1+(self.distribution=="gaussian")) * feat for feat in quant_config.out_features]
        self.quantize = self.create_encoder_decoder_block(quant_config, self.bottleneck_zooms, in_features,
                                                          quant_out_feat, **kwargs)
        self.post_quantize = self.create_encoder_decoder_block(quant_config, self.bottleneck_zooms, quant_config.out_features,
                                                               in_features, **kwargs)

        for block_key, block_conf in decoder_block_configs.items():
            block = self.create_encoder_decoder_block(block_conf, in_zooms, in_features, **kwargs)
            self.decoder_blocks[block_key] = block

            in_features = block.out_features
            in_zooms = block.out_zooms

        block.out_features = [in_features[0]]

        if len(decoder_settings) == 0:
            self.decoder = DiffDecoder()

        else:
            self.decoder = MFieldLayer(
                in_features,
                decoder_settings['out_features'],
                in_zooms,
                self.grid_layers,
                with_nh=decoder_settings.get('with_nh', True),
                embed_confs=decoder_settings.get('embed_confs', {}),
                N=decoder_settings.get('N', 2),
                kmin=decoder_settings.get('kmin', 0),
                kmax=decoder_settings.get('kmin', 0.5),
                layer_confs=decoder_settings.get('layer_confs', {}))

    def create_encoder_decoder_block(self, block_conf, in_zooms, in_features, out_features=None, **kwargs):
        layer_confs = check_get([block_conf, kwargs, defaults], "layer_confs")

        if isinstance(block_conf, MGProcessingConfig):
            layer_settings = block_conf.layer_settings

            block = MG_SingleBlock(
                self.grid_layers,
                in_zooms,
                layer_settings,
                in_features,
                out_features if out_features is not None else block_conf.out_features,
                layer_confs=layer_confs,
                zooms=check_get([block_conf, kwargs, {"zooms": None}], "zooms"),
                layer_confs_emb=check_get([block_conf,kwargs,{"layer_confs_emb": {}}], "layer_confs_emb"),
                use_mask=check_get([block_conf, kwargs,{"use_mask": False}], "use_mask"))

        elif isinstance(block_conf, MGConservativeConfig):
            block = ConservativeLayer(in_zooms,
                                      first_feature_only=self.predict_var)
            block.out_features = in_features

        elif isinstance(block_conf, MGCoordinateEmbeddingConfig):
            block = MGEmbedding(self.grid_layers[str(block_conf.emb_zoom)],
                                block_conf.features,
                                n_groups=check_get([block_conf, kwargs, {'n_groups': 1}], "n_groups"),
                                zooms=in_zooms,
                                init_mode=check_get([block_conf, kwargs, {'init_mode': "fourier_sphere"}], "init_mode"),
                                layer_confs=layer_confs
                                )
            block.out_zooms = in_zooms

        elif isinstance(block_conf, MGSelfProcessingConfig):
            layer_settings = block_conf.layer_settings
            layer_settings['layer_confs'] = check_get([block_conf, kwargs, defaults], "layer_confs")

            block = MG_MultiBlock(
                self.grid_layers,
                in_zooms,
                check_get([block_conf, {'out_zooms': in_zooms}], "out_zooms"),
                layer_settings,
                in_features,
                out_features if out_features is not None else block_conf.out_features,
                q_zooms=check_get([block_conf, kwargs, {"q_zooms": -1}], "q_zooms"),
                kv_zooms=check_get([block_conf, kwargs, {"kv_zooms": -1}], "kv_zooms"),
                layer_confs=layer_confs,
                layer_confs_emb=check_get([block_conf,kwargs,{"layer_confs_emb": {}}], "layer_confs_emb"),
                use_mask=check_get([block_conf, kwargs,{"use_mask": False}], "use_mask"),
                init_missing_zooms=check_get([block_conf, kwargs, {"init_missing_zooms": "zeros"}], "init_missing_zooms"),
                residual=check_get([block_conf, {'residual': False}], "residual"))

        elif isinstance(block_conf, MGFieldAttentionConfig):
            layer_settings = block_conf.layer_settings
            layer_settings['layer_confs'] = check_get([block_conf, kwargs, defaults], "layer_confs")

            block = MG_MultiBlock(
                self.grid_layers,
                in_zooms,
                check_get([block_conf, {'out_zooms': in_zooms}], "out_zooms"),
                layer_settings,
                in_features,
                block_conf.out_features,
                q_zooms=check_get([block_conf, kwargs, {"q_zooms": -1}], "q_zooms"),
                kv_zooms=check_get([block_conf, kwargs, {"kv_zooms": -1}], "kv_zooms"),
                layer_confs=layer_confs,
                layer_confs_emb=check_get([block_conf, kwargs, {"layer_confs_emb": {}}], "layer_confs_emb"),
                use_mask=check_get([block_conf, kwargs, {"use_mask": False}], "use_mask"),
                type='field_att',
                init_missing_zooms=check_get([block_conf, kwargs, {"init_missing_zooms": "zeros"}], "init_missing_zooms"))
        return block

    def vae_encode(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        emb['MGEmbedder'] = (self.mg_emeddings, emb['GroupEmbedder'])

        for k, block in enumerate(self.encoder_blocks.values()):
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        x_zooms = self.quantize(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)
        posterior_zooms = {int(zoom): self.get_distribution(x) for zoom, x in x_zooms.items()}
        return posterior_zooms

    def vae_decode(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        x_zooms = self.post_quantize(x_zooms, sample_configs=sample_configs, emb=emb)

        for k, block in enumerate(self.decoder_blocks.values()):
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        return x_zooms

    def decode(self, x_zooms:Dict[str, torch.Tensor], sample_configs: Dict, out_zoom: int=None, emb={}):
        return self.decoder(x_zooms, emb=emb, sample_configs=sample_configs, out_zoom=out_zoom)


    def forward(self, x_zooms: Dict[int, torch.Tensor], sample_configs={}, mask_zooms: Dict[int, torch.Tensor]= None, emb=None, out_zoom=None):

        """
        Forward pass for the ICON_Transformer model.

        :param x: Input tensor of shape (batch_size, num_cells, input_dim).
        :param coords_input: Input coordinates for x
        :param coords_output: Output coordinates for position embedding.
        :param sampled_indices_batch_dict: Dictionary of sampled indices for regional models.
        :param mask: Mask for dropping cells in the input tensor.
        :return: Output tensor of shape (batch_size, num_cells, output_dim).
        """

        b, nv, nt, n, nc = x_zooms[list(x_zooms.keys())[0]].shape

        assert nc == self.in_features, f" the input has {nc} features, which doesnt match the numnber of specified input_features {self.in_features}"
        assert nc == (self.out_features // (1+ self.predict_var)), f" the input has {nc} features, which doesnt match the numnber of specified out_features {self.out_features}"

        posterior_zooms = self.vae_encode(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        z_zooms = {int(zoom): x.sample(gamma=self.gammas[str(zoom)] if self.gammas else None) for zoom, x in posterior_zooms.items()}

        dec = self.vae_decode(z_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)
        dec = self.decoder(dec, sample_configs=sample_configs, emb=emb, out_zoom=out_zoom)

        return dec, posterior_zooms

    def get_distribution(self, x):
        """
        Encodes the input tensor x into a quantized latent space.

        :param x: Input tensor.
        :return: Distribution for tensor.
        """
        assert self.distribution == "gaussian" or self.distribution == "dirac"
        if self.distribution == "gaussian":
            return DiagonalGaussianDistribution(x)
        elif self.distribution == "dirac":
            return DiracDistribution(x)
        else:
            return None