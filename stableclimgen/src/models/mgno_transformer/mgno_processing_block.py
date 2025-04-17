from typing import List
import torch.nn as nn

from ...modules.icon_grids.grid_layer import GridLayer
from ...utils.helpers import check_get_missing_key
from ...modules.neural_operator.no_helpers import get_no_layer,get_embedder_from_dict
from ...modules.neural_operator.no_blocks import PreActivation_NOBlock, NOBlock, VariableAttention, SpatialAttention


class MGNO_Processing_Block(nn.Module):
  
    def __init__(self,
                 input_levels: List[int],
                 layer_settings_levels: List[List],
                 input_dims: List[int],
                 model_dims_out: List[List],
                 grid_layers: List[GridLayer],
                 rotate_coordinate_system: bool = True,
                 mask_as_embedding = False,
                 n_vars_total = 1,
                 rank_vars = 4,
                 factorize_vars = False
                ) -> None: 
      
        super().__init__()

        self.mask_as_embedding = mask_as_embedding
        self.output_levels = input_levels
        self.layers = nn.ModuleList()

        self.model_dims_out = model_dims_out[-1]
        for level_idx, layer_settings_level in enumerate(layer_settings_levels):
            
            input_dim = input_dims[level_idx]
            current_level = input_levels[level_idx]

            level_layers_ = nn.ModuleList()

            for layer_idx, layer_setting in enumerate(layer_settings_level):
                model_dim_out = model_dims_out[layer_idx][level_idx]

                min_lvl = layer_setting.get("min_lvl",0)
                
                if current_level < min_lvl:
                    continue

                if 'NO' in layer_setting["type"]:
                    no_layer_type = check_get_missing_key(layer_setting, "no_layer_type")
                    no_layer_settings = check_get_missing_key(layer_setting, "no_layer_settings")

                    layer_setting['nh_in_encode'] = True
                    layer_setting['nh_in_decode'] = True

                    no_layer = get_no_layer(no_layer_type,
                                            grid_layers[str(current_level)],
                                            grid_layers[str(current_level)],
                                            grid_layers[str(current_level)],
                                            precompute_encode=True,
                                            precompute_decode=True,
                                            rotate_coordinate_system=rotate_coordinate_system,
                                            layer_settings=no_layer_settings,
                                            normalize_to_mask=(mask_as_embedding==False))
                    
                    if "mlp" in layer_setting["type"]:
                        embedder = get_embedder_from_dict(layer_setting)
                        if 'pre' in layer_setting["type"]:
                            layer = PreActivation_NOBlock(
                                        model_dim_in=input_dim,
                                        model_dim_out=model_dim_out,
                                        no_layer=no_layer,
                                        embedder=embedder,
                                        cross_no = 'cross_no' in layer_setting["type"],
                                        with_gamma = 'gamma' in layer_setting["type"],
                                        n_vars_total=n_vars_total,
                                        rank_vars=rank_vars,
                                        factorize_vars=factorize_vars
                                        )
                        else:
                            layer = NOBlock(
                                        model_dim_in=input_dim,
                                        model_dim_out=model_dim_out,
                                        no_layer=no_layer,
                                        embedder=embedder,
                                        cross_no = 'cross_no' in layer_setting["type"],
                                        with_gamma = 'gamma' in layer_setting["type"],
                                        n_vars_total=n_vars_total,
                                        rank_vars=rank_vars,
                                        factorize_vars=factorize_vars
                                        )

                elif "var_att" in layer_setting["type"]:
                    embedder_att = get_embedder_from_dict(layer_setting)
                    embedder_mlp = get_embedder_from_dict(layer_setting)

                    layer = VariableAttention(
                        input_dim,
                        model_dim_out,
                        grid_layers[str(current_level)],
                        layer_setting["n_head_channels"],
                        att_dim=layer_setting.get("att_dim",None),
                        p_dropout=layer_setting.get("p_dropout",0),
                        embedder = embedder_att,
                        embedder_mlp= embedder_mlp,
                        mask_as_embedding=mask_as_embedding,
                        n_vars_total=n_vars_total,
                        rank_vars=rank_vars,
                        factorize_vars=factorize_vars
                    )
                
                elif "spatial_att" in layer_setting["type"]:
                    
                    spatial_attention_configs = check_get_missing_key(layer_setting, "spatial_attention_configs")
                    spatial_attention_configs["embedder_names"] = layer_setting.get("embedder_names",[[],[]])
                    spatial_attention_configs["embed_confs"] = layer_setting.get("embed_confs",None)
                    spatial_attention_configs["embed_mode"] = layer_setting.get("embed_mode","sum")

                    layer = SpatialAttention(
                        input_dim,
                        model_dim_out,
                        grid_layers[str(current_level)],
                        layer_setting["n_head_channels"],
                        p_dropout=layer_setting.get("p_dropout",0),
                        spatial_attention_configs=spatial_attention_configs,
                        rotate_coord_system=rotate_coordinate_system,
                        mask_as_embedding=mask_as_embedding
                    )
                
                input_dim = model_dim_out

                level_layers_.append(layer)

            self.layers.append(level_layers_)
        


    def forward(self, x_levels, coords_in=None, coords_out=None, indices_sample=None, mask_levels=None, emb=None):
        
        for level_idx, layer_levels in enumerate(self.layers):
            x = x_levels[level_idx]
            mask = mask_levels[level_idx]

            for layer_idx, layer in enumerate(layer_levels):

                if isinstance(layer, NOBlock) or isinstance(layer, PreActivation_NOBlock):
                    x, mask = layer(x, coords_encode=coords_in, coords_decode=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)
                elif isinstance(layer, SpatialAttention):
                    x = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
                else:
                    x = layer(x, mask=mask, emb=emb)

            x_levels[level_idx] = x
            mask_levels[level_idx] = mask

        return x_levels, mask_levels