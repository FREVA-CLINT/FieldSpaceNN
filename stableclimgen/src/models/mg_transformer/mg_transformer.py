import torch
import torch.nn as nn
from typing import List,Dict
from ...utils.helpers import check_get
from ...modules.multi_grid.mg_base import DiffDecoder
from ...modules.grids.grid_utils import decode_zooms
from .confs import defaults
from .mg_base_model import MG_base_model, create_encoder_decoder_block

class MG_Transformer(MG_base_model):
    def __init__(self, 
                 mgrids,
                 block_configs: List,
                 in_zooms: List,
                 in_features: int=1,
                 out_features: int=1,
                 n_groups_variables: List = [1],

                 **kwargs
                 ) -> None: 
        
        
        super().__init__(mgrids)

        self.in_zooms = in_zooms
        self.in_features = in_features 
        predict_var = kwargs.get("predict_var", defaults['predict_var'])

        if predict_var:
            out_features = out_features * 2
            self.activation_var = nn.Softplus()

        self.out_features = out_features
        self.predict_var = predict_var

        self.Blocks = nn.ModuleDict()

        in_features = [in_features]*len(in_zooms)

        for block_key, block_conf in block_configs.items():
            assert isinstance(block_key, str), "block keys should be strings"
            block = create_encoder_decoder_block(block_conf, in_zooms, in_features, n_groups_variables, self.predict_var, self.grid_layers)

            self.Blocks[block_key] = block     

            in_features = block.out_features
            in_zooms = block.out_zooms        

        block.out_features = [in_features[0]]
        
        self.masked_residual = check_get([kwargs, defaults], "masked_residual")
        self.learn_residual = check_get([kwargs, defaults], "learn_residual") if not self.masked_residual else True

        self.decoder = DiffDecoder()

        
    def decode(self, x_zooms:Dict[str, torch.Tensor], sample_configs: Dict, out_zoom: int=None, emb={}):
        return self.decoder(x_zooms, emb=emb, sample_configs=sample_configs, out_zoom=out_zoom)


    def forward(self, 
                x_zooms_groups: List[Dict[int, torch.Tensor]] = None,
                mask_zooms_groups: List[Dict[int, torch.Tensor]] = None,
                emb_groups: List[Dict] = None,
                sample_configs={},
                out_zoom=None):

        if x_zooms_groups is None:
            x_zooms_groups = []
        if mask_zooms_groups is None:
            mask_zooms_groups = [None] * len(x_zooms_groups)
        if emb_groups is None:
            emb_groups = [{} for _ in range(len(x_zooms_groups))]

        x_zooms_groups_res = None
        #if self.learn_residual:
        x_zooms_groups_res = [x.copy() for x in x_zooms_groups]

        mask_zooms_groups_res = None
        if self.masked_residual:
            mask_zooms_groups_res = [
                mask.copy() if mask is not None else None for mask in mask_zooms_groups
            ]

        for block in self.Blocks.values():
            x_zooms_groups = block(
                x_zooms_groups,
                sample_configs=sample_configs,
                mask_groups=mask_zooms_groups,
                emb_groups=emb_groups,
            )
            pass

        for i, x_zooms in enumerate(x_zooms_groups):

            x_res = x_zooms_groups_res[i] if self.learn_residual else None
            x_zooms_groups[i] = self.apply_residuals(x_zooms, x_res, mask_zooms_groups, mask_zooms_groups_res, sample_configs)

        if out_zoom is not None:
            for i, x_zooms in enumerate(x_zooms_groups):
                x_zooms_groups[i] = (
                    self.decoder(x_zooms, sample_configs=sample_configs, out_zoom=out_zoom)
                    if x_zooms
                    else {}
                )

        return x_zooms_groups


    def apply_residuals(self, x_zooms, x_zooms_res, mask_zooms, mask_res, sample_configs):
        if not x_zooms:
            return x_zooms

        if mask_res is None:
            mask_res = mask_zooms

        if self.learn_residual:
            for i, x_zooms_res in enumerate(x_zooms_res):
                if len(x_zooms_res) == 1 :
                    x_zooms_res = decode_zooms(
                        x_zooms_res,
                        sample_configs=sample_configs,
                        out_zoom=list(x_zooms_res.keys())[0],
                    )

        if self.predict_var and self.learn_residual:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2, dim=-1)

                if not self.masked_residual or mask_zooms is None:
                    x = x_zooms_res[zoom] + x
                elif mask_zooms[zoom].dtype == torch.bool:
                    x = (1 - mask_zooms[zoom]) * x_zooms_res[zoom] + (mask_zooms[zoom]) * x
                else:
                    x = mask_zooms[zoom] * x_zooms_res[zoom] + (1 - mask_zooms[zoom]) * x

                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)), dim=-1)

        elif self.predict_var:
            for zoom, x in x_zooms.items():
                x, x_var = x.chunk(2, dim=-1)
                x_zooms[zoom] = torch.concat((x, self.activation_var(x_var)), dim=-1)

        elif self.learn_residual:
            for zoom in x_zooms.keys():
                if not self.masked_residual or mask_zooms is None:
                    x_zooms[zoom] = x_zooms_res[zoom] + x_zooms[zoom]
                elif mask_zooms[zoom].dtype == torch.bool:
                    x_zooms[zoom] = (1 - 1. * mask_zooms[zoom]) * x_zooms_res[zoom] + (mask_zooms[zoom]) * x_zooms[zoom]
                else:
                    x_zooms[zoom] = mask_zooms[zoom] * x_zooms_res[zoom] + (1 - mask_res[zoom]) * x_zooms[zoom]

        return x_zooms
