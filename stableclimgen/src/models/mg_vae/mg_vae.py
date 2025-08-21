from typing import List,Dict

import torch
import torch.nn as nn
from stableclimgen.src.modules.distributions.distributions import DiagonalGaussianDistribution, DiracDistribution

from ..mg_transformer.confs import defaults
from ..mg_transformer.mg_base_model import MG_base_model
from .confs import MGQuantConfig
from ...modules.embedding.embedder import get_embedder
from ...modules.multi_grid.confs import MGProcessingConfig, MGSelfProcessingConfig, MGConservativeConfig, \
    MGCoordinateEmbeddingConfig, MGFieldAttentionConfig
from ...modules.multi_grid.mg_base import ConservativeLayer, MGEmbedding, get_mg_embedding
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
            self.mg_emeddings = nn.ParameterDict()

            for zoom, features, n_groups, init_method in zip(mg_emb_confs['zooms'],mg_emb_confs['features'],mg_emb_confs["n_groups"],mg_emb_confs['init_methods']):
            
                self.mg_emeddings[str(zoom)] = get_mg_embedding(
                    self.grid_layers[str(zoom)],
                    features,
                    n_groups,
                    init_mode=init_method)
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

        quant_out_feat = [2 * feat for feat in quant_config.out_features]
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
                init_missing_zooms=check_get([block_conf, kwargs, {"init_missing_zooms": "zeros"}], "init_missing_zooms"))

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

    def encode(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        emb['MGEmbedder'] = (self.mg_emeddings, emb['GroupEmbedder'])

        for k, block in enumerate(self.encoder_blocks.values()):
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        x_zooms = self.quantize(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)
        posterior_zooms = {int(zoom): self.get_distribution(x) for zoom, x in x_zooms.items()}
        return posterior_zooms

    def decode(self, x_zooms, sample_configs={}, mask_zooms=None, emb=None):
        x_zooms = self.post_quantize(x_zooms, sample_configs=sample_configs, emb=emb)

        for k, block in enumerate(self.decoder_blocks.values()):
            x_zooms = block(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        return x_zooms


    def forward(self, x_zooms: Dict[int, torch.Tensor], sample_configs={}, mask_zooms: Dict[int, torch.Tensor]= None, emb=None):

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

        posterior_zooms = self.encode(x_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

        z_zooms = {int(zoom): x.sample() for zoom, x in posterior_zooms.items()}

        dec = self.decode(z_zooms, sample_configs=sample_configs, mask_zooms=mask_zooms, emb=emb)

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