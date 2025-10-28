from typing import List


class CNNBlockConfig:
    """
    Configuration class for defining the parameters of VAE blocks.

    :param depth: Number of layers in the block.
    :param block_type: Type of block (e.g., 'ConvBlock' or 'ResnetBlock').
    :param ch_mult: Channel multiplier for the block; can be an int or list of ints.
    :param blocks: List of block configurations specific to the block type.
    :param enc: Boolean indicating if this is an encoder block. Default is False.
    :param dec: Boolean indicating if this is a decoder block. Default is False.
    """

    def __init__(self, depth: int, block_type: str, ch_mult: int | List[int], sub_confs: dict, enc: bool = False,
                 dec: bool = False, embedders: List[List[str]] = None):
        self.depth = depth
        self.block_type = block_type
        self.sub_confs = sub_confs
        self.enc = enc
        self.dec = dec
        self.ch_mult = ch_mult
        self.embedders = embedders


class QuantConfig:
    """
    Configuration class for defining quantization parameters in the VAE model.

    :param z_ch: Number of latent channels in the quantized representation.
    :param latent_ch: Number of latent channels in the bottleneck.
    :param block_type: Block type used in quantization (e.g., 'ConvBlock' or 'ResnetBlock').
    """

    def __init__(self, latent_ch: List[int], block_type: str, sub_confs: dict):
        self.latent_ch = latent_ch
        self.block_type = block_type
        self.sub_confs = sub_confs