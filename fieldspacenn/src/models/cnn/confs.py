from typing import Any, List, Mapping, Optional, Sequence


class PatchEmbConfig:
    """
    Configuration class for defining diffusion blocks in the model.

    :param depth: Number of layers in the block.
    :param block_type: Type of block (e.g., 'ConvBlock', 'ResnetBlock').
    :param ch_mult: Channel multiplier for the block.
    :param sub_confs: Sub-configuration details specific to the block type.
    :param enc: Whether the block is an encoder block. Default is False.
    :param dec: Whether the block is a decoder block. Default is False.
    """

    def __init__(self,
                 block_type: str = "ConvBlock",
                 patch_emb_size: tuple[int, int] | tuple[int, int, int] = (1, 1, 1),
                 patch_emb_kernel: tuple[int, int] | tuple[int, int, int] = (1, 1, 1),
                 sub_confs=None):
        self.block_type = block_type
        self.patch_emb_size = patch_emb_size
        self.patch_emb_kernel = patch_emb_kernel
        self.sub_confs = sub_confs


class CNNBlockConfig:
    """
    Configuration class for defining the parameters of VAE blocks.

    :param depth: Number of layers in the block.
    :param block_type: Type of block (e.g., 'ConvBlock' or 'ResnetBlock').
    :param ch_mult: Channel multiplier for the block; can be an int or list of ints.
    :param blocks: List of block configurations specific to the block type.
    :param enc: Boolean indicating if this is an encoder block. Default is False.
    :param dec: Boolean indicating if this is a decoder block. Default is False.
    :param embedders: Optional list of embedder names per block.
    """

    def __init__(
        self,
        depth: int,
        block_type: str,
        ch_mult: int | List[int],
        sub_confs: Mapping[str, Any],
        enc: bool = False,
        dec: bool = False,
        embedders: Optional[Sequence[Sequence[str]]] = None,
    ) -> None:
        """
        Store the configuration for a CNN block stack.

        :param depth: Number of layers in the block.
        :param block_type: Type of block (e.g., 'ConvBlock' or 'ResnetBlock').
        :param ch_mult: Channel multiplier for the block; can be an int or list of ints.
        :param sub_confs: Block-type specific configuration.
        :param enc: Boolean indicating if this is an encoder block.
        :param dec: Boolean indicating if this is a decoder block.
        :param embedders: Optional list of embedder names per block.
        :return: None.
        """
        self.depth: int = depth
        self.block_type: str = block_type
        self.sub_confs: Mapping[str, Any] = sub_confs
        self.enc: bool = enc
        self.dec: bool = dec
        self.ch_mult: int | List[int] = ch_mult
        self.embedders: Optional[Sequence[Sequence[str]]] = embedders


class QuantConfig:
    """
    Configuration class for defining quantization parameters in the VAE model.

    :param z_ch: Number of latent channels in the quantized representation.
    :param latent_ch: Number of latent channels in the bottleneck.
    :param block_type: Block type used in quantization (e.g., 'ConvBlock' or 'ResnetBlock').
    """

    def __init__(
        self,
        latent_ch: Sequence[int],
        block_type: str,
        sub_confs: Mapping[str, Any],
    ) -> None:
        """
        Store quantization configuration parameters.

        :param latent_ch: Number of latent channels in the bottleneck.
        :param block_type: Block type used in quantization (e.g., 'ConvBlock' or 'ResnetBlock').
        :param sub_confs: Block-type specific configuration.
        :return: None.
        """
        self.latent_ch: Sequence[int] = latent_ch
        self.block_type: str = block_type
        self.sub_confs: Mapping[str, Any] = sub_confs
