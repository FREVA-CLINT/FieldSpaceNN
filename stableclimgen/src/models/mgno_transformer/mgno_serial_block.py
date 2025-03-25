from typing import List
import torch.nn as nn

from ...modules.icon_grids.grid_layer import GridLayer
from ...utils.helpers import check_get_missing_key
from ...modules.neural_operator.no_helpers import get_no_layer,get_embedder_from_dict
from ...modules.neural_operator.no_blocks import PreActivation_NOBlock, NOBlock, VariableAttention, SpatialAttention


class Serial_NOBlock(nn.Module):
  
    def __init__(self,
                 rcm,
                 input_dim: int,
                 model_dims_out: List[int],
                 grid_layers: List[GridLayer],
                 layer_settings: List[dict],
                 input_level: int = 0,
                 output_dim: int = None,
                 rotate_coordinate_system: bool = True,
                 mask_as_embedding = False
                ) -> None: 
      
        super().__init__()

        self.layers = nn.ModuleList()

        current_level = input_level
        for layer_idx, layer_setting in enumerate(layer_settings):
            
            model_dim_in = input_dim if layer_idx==0 else model_dims_out[layer_idx-1]
            model_dim_out = model_dims_out[layer_idx]

            if 'NO' in layer_setting["type"]:
                global_level_decode = check_get_missing_key(layer_setting, "global_level_decode")
                global_level_no = check_get_missing_key(layer_setting, "global_level_no")
                no_layer_type = check_get_missing_key(layer_setting, "no_layer_type")
                no_layer_settings = check_get_missing_key(layer_setting, "no_layer_settings")

                global_level_in = current_level
                global_level_out = global_level_decode
                current_level = global_level_out

                no_layer = get_no_layer(rcm,
                                        no_layer_type,
                                        global_level_in,
                                        global_level_no,
                                        global_level_out,
                                        precompute_encode=True,
                                        precompute_decode=True if layer_idx < len(model_dims_out)-1 else False,
                                        layer_settings=no_layer_settings,
                                        normalize_to_mask=(mask_as_embedding==False))
                
                if "mlp" in layer_setting["type"]:
                    embedder = get_embedder_from_dict(layer_setting)

                    if 'pre' in layer_setting["type"]:
                        layer = PreActivation_NOBlock(
                                    model_dim_in=model_dim_in,
                                    model_dim_out=model_dim_out,
                                    no_layer=no_layer,
                                    embedder=embedder,
                                    layer_type= layer_setting["layer_type"],
                                    with_gamma = layer_setting["with_gamma"],
                                    mask_as_embedding=mask_as_embedding
                                    )
                    else:
                        layer = NOBlock(
                                    model_dim_in=model_dim_in,
                                    model_dim_out=model_dim_out,
                                    no_layer=no_layer,
                                    embedder=embedder,
                                    layer_type= layer_setting["layer_type"],
                                    with_gamma = layer_setting["with_gamma"],
                                    mask_as_embedding=mask_as_embedding
                                    )
                

            if "var_att" in layer_setting["type"]:
                embedder_att = get_embedder_from_dict(layer_setting)
                embedder_mlp = get_embedder_from_dict(layer_setting)

                layer = VariableAttention(
                    model_dim_in,
                    model_dim_out,
                    grid_layers[str(current_level)],
                    layer_setting["n_head_channels"],
                    att_dim=layer_setting.get("att_dim",None),
                    p_dropout=layer_setting.get("p_dropout",0),
                    embedder = embedder_att,
                    embedder_mlp= embedder_mlp,
                    mask_as_embedding=mask_as_embedding
                )
            
            elif "spatial_att" in layer_setting["type"]:
                
                spatial_attention_configs = check_get_missing_key(layer_setting, "spatial_attention_configs")
                spatial_attention_configs["embedder_names"] = layer_setting.get("embedder_names",[[],[]])
                spatial_attention_configs["embed_confs"] = layer_setting.get("embed_confs",None)
                spatial_attention_configs["embed_mode"] = layer_setting.get("embed_mode","sum")

                layer = SpatialAttention(
                    model_dim_in,
                    model_dim_out,
                    grid_layers[str(current_level)],
                    layer_setting["n_head_channels"],
                    p_dropout=layer_setting.get("p_dropout",0),
                    spatial_attention_configs=spatial_attention_configs,
                    rotate_coord_system=rotate_coordinate_system,
                    mask_as_embedding=mask_as_embedding
                )
                
            self.layers.append(layer)
            
        self.output_level = current_level
        self.output_layer = nn.Linear(model_dims_out[-1], output_dim, bias=False) if output_dim is not None else nn.Identity()


    def forward(self, x, coords_in=None, coords_out=None, indices_sample=None, mask=None, emb=None):
        
        for layer_idx, layer in enumerate(self.layers):
            
            if isinstance(layer, NOBlock) or isinstance(layer, PreActivation_NOBlock):
                x, mask = layer(x, coords_encode=coords_in, coords_decode=coords_out, indices_sample=indices_sample, mask=mask, emb=emb)
            elif isinstance(layer, SpatialAttention):
                x = layer(x, indices_sample=indices_sample, mask=mask, emb=emb)
            else:
                x = layer(x, mask=mask, emb=emb)

        x = self.output_layer(x)

        return x, mask