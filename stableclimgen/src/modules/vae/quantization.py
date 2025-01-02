from typing import Optional, List, Dict
import torch
import torch.nn as nn
from stableclimgen.src.utils.helpers import check_value
from ..embedding.embedder import BaseEmbedder, EmbedderManager, EmbedderSequential

from ..icon_grids.grid_attention import GridAttention
from ..rearrange import RearrangeConvCentric
from ..cnn.conv import ConvBlockSequential
from ..cnn.resnet import ResBlockSequential
from ..transformer.attention import AdaptiveLayerNorm
from ..transformer.transformer_base import TransformerBlock
from ..utils import EmbedBlockSequential


class Quantization(nn.Module):
    """
    A quantization module for encoding and decoding data through various block types (conv, resnet, transformer).

    :param in_ch: Input channel size.
    :param z_ch: Latent space channel size for quantization.
    :param latent_ch: Latent space channel size for bottleneck processing.
    :param block_type: Block type for the quantization process ('conv', 'resnet', or 'transformer').
    :param spatial_dim_count: Number of spatial dimensions (2 or 3).
    """

    def __init__(
            self,
            in_ch: int,
            latent_ch: List[int],
            block_type: str,
            spatial_dim_count: int,
            blocks: List[str],
            embedder_names: List[List[str]] = None,
            embed_confs: Dict = None,
            embed_mode: str = "sum",
            dims: int = 2,
            **kwargs
    ):
        super().__init__()
        # Choose the block type based on provided configuration
        if block_type == "conv":
            # Define convolutional block for quantization
            self.quant = RearrangeConvCentric(
                EmbedBlockSequential(
                    nn.GroupNorm(32, in_ch),  # Normalize input channels
                    ConvBlockSequential(in_ch, [2 * l_ch for l_ch in latent_ch], blocks, dims=dims, **kwargs)
                ), spatial_dim_count, dims=dims
            )
            # Define convolutional block for post-quantization decoding
            self.post_quant = RearrangeConvCentric(
                ConvBlockSequential(latent_ch[-1], latent_ch[::-1][1:] + [in_ch], blocks, dims=dims, **kwargs),
                spatial_dim_count, dims=dims
            )
        elif block_type == "resnet":
            # Define ResNet block for quantization
            self.quant = RearrangeConvCentric(
                EmbedBlockSequential(
                    ResBlockSequential(in_ch, [2 * l_ch for l_ch in latent_ch], blocks,
                                       embedder_names=embedder_names, embed_confs=embed_confs, embed_mode=embed_mode, dims=dims,
                                       **kwargs)
                ), spatial_dim_count, dims=dims
            )
            # Define ResNet block for post-quantization decoding
            self.post_quant = RearrangeConvCentric(
                ResBlockSequential(latent_ch[-1], latent_ch[::-1][1:] + [in_ch], blocks,
                                   embedder_names=embedder_names, embed_confs=embed_confs, embed_mode=embed_mode, dims=dims,
                                   **kwargs),
                spatial_dim_count, dims=dims
            )
        elif block_type == "trans":
            # Use Transformer block for quantization and post-quantization
            self.quant = TransformerBlock(in_ch, [2 * l_ch for l_ch in latent_ch], blocks, spatial_dim_count=spatial_dim_count,
                                          embedder_names=embedder_names, embed_confs=embed_confs, embed_mode=embed_mode, **kwargs)
            self.post_quant = TransformerBlock(latent_ch[-1], latent_ch[::-1][1:] + [in_ch], blocks, spatial_dim_count=spatial_dim_count,
                                               embedder_names=embedder_names, embed_confs=embed_confs, embed_mode=embed_mode, **kwargs)
        elif block_type == "grid_trans":
            grid_layer = kwargs.pop("grid_layer")
            n_head_channels = check_value(kwargs.pop("n_head_channels"), len(blocks))
            rotate_coord_system = kwargs.pop("rotate_coord_system")
            n_params = kwargs.pop("n_params")

            spatial_attention_configs = kwargs
            spatial_attention_configs["embedder_names"] = embedder_names
            spatial_attention_configs["embed_confs"] = embed_confs
            spatial_attention_configs["embed_mode"] = embed_mode
            spatial_attention_configs["blocks"] = blocks

            self.quant = GridQuantizationEnc(n_params, grid_layer, in_ch, latent_ch[-1], n_head_channels, spatial_attention_configs.copy(), rotate_coord_system)
            self.post_quant = GridQuantizationDec(n_params[:len(self.quant.layers)][::-1], grid_layer, in_ch, latent_ch[-1], n_head_channels, spatial_attention_configs.copy(), rotate_coord_system)

    def quantize(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                 cond: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Encodes the input tensor x into a quantized latent space.

        :param x: Input tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Quantized tensor.
        """
        return self.quant(x, emb=emb, mask=mask, cond=cond, *args, **kwargs)

    def post_quantize(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                      cond: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        """
        Decodes the quantized tensor x back to the original space.

        :param x: Quantized tensor.
        :param emb: Optional embedding tensor.
        :param mask: Optional mask tensor.
        :param cond: Optional conditioning tensor.
        :return: Decoded tensor.
        """
        return self.post_quant(x, emb=emb, mask=mask, cond=cond, *args, **kwargs)


class GridQuantizationEnc(nn.Module):
    def __init__(
            self,
            n_params,
            grid_layer,
            model_channels: int,
            latent_ch: int,
            n_head_channels: List[int],
            spatial_attention_configs,
            rotate_coord_system
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for n_param in n_params:
            in_ch = n_param[0] * n_param[1]
            model_channels = model_channels // in_ch
            layer = GridQuantLayer(grid_layer, model_channels, in_ch, 1, n_head_channels, spatial_attention_configs, rotate_coord_system)
            if model_channels >= latent_ch:
                self.layers.append(layer)
        #self.log_var_layer = GridAttention(grid_layer, latent_ch, latent_ch, n_head_channels, spatial_attention_configs.copy(), rotate_coord_system)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, emb, mask, cond, *args, **kwargs)

        #log_var = self.log_var_layer(x, emb=emb, mask=mask, cond=cond, *args, **kwargs)
        #return torch.cat([x, log_var], dim=-1)
        return x


class GridQuantizationDec(nn.Module):
    def __init__(
            self,
            n_params,
            grid_layer,
            model_channels: int,
            latent_ch: int,
            n_head_channels: List[int],
            spatial_attention_configs,
            rotate_coord_system
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for n_param in n_params:
            out_ch = n_param[0] * n_param[1]
            latent_ch = latent_ch * out_ch
            layer = GridQuantLayer(grid_layer, latent_ch, 1, out_ch, n_head_channels, spatial_attention_configs, rotate_coord_system)
            if model_channels >= latent_ch:
                self.layers.append(layer)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, emb, mask, cond, *args, **kwargs)
        return x


class GridQuantLayer(nn.Module):
    def __init__(
            self,
            grid_layer,
            model_channels: int,
            in_ch: int,
            out_ch: int,
            n_head_channels: List[int],
            spatial_attention_configs,
            rotate_coord_system
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        emb_dict = nn.ModuleDict()
        emb: BaseEmbedder = EmbedderManager().get_embedder("VariableEmbedder", **spatial_attention_configs["embed_confs"]["VariableEmbedder"])
        emb_dict[emb.name] = emb
        self.embedder = EmbedderSequential(emb_dict, mode=spatial_attention_configs["embed_mode"], spatial_dim_count = 1)
        self.embedding_layer = torch.nn.Linear(self.embedder.get_out_channels, 2 * in_ch)
        self.layer_norm = torch.nn.LayerNorm(in_ch)

        self.linear_layer = nn.Linear(in_ch, out_ch)
        self.transformer = GridAttention(grid_layer, model_channels, model_channels, n_head_channels, spatial_attention_configs.copy(), rotate_coord_system)
        self.gamma = torch.nn.Parameter(torch.ones(out_ch) * 1E-6)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                 cond: Optional[torch.Tensor] = None, *args, **kwargs) -> torch.Tensor:
        b, t, g, v, c = x.shape
        x = x.view(b, t, g, v, -1, self.in_ch)

        if self.in_ch > self.out_ch:
            x_res = x.mean(-1, keepdim=True)
        else:
            x_res = x.repeat(1, 1, 1, 1, 1, self.out_ch // self.in_ch)

        if self.embedder:
            # Apply the embedding transformation (scale and shift)
            scale, shift = self.embedding_layer(self.embedder(emb)).unsqueeze(-2).chunk(2, dim=-1)
            x = self.layer_norm(x) * (scale + 1) + shift
        else:
            x = self.layer_norm(x)

        x = self.linear_layer(x)
        x = x_res + self.gamma * x
        x = x.view(b, t, g, v, -1)

        return self.transformer(x, emb=emb, mask=mask, cond=cond, *args, **kwargs)

