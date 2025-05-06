from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from stableclimgen.src.modules.embedding.embedder import BaseEmbedder, EmbedderManager, EmbedderSequential


class TemporalProcessor(nn.Module):
    """
    A PyTorch module that operates on 5D tensors (batch, time, grid, var, channel)
    to perform time reduction, time increase, or identity operations along the
    time dimension using 1D convolutions or interpolation.

    Includes adaptive Layer Normalization, optional residual connection,
    and optional skip connection.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 mode = 'identity',
                 time_factor: int = 7,
                 nh: int = 0,
                 residual: bool = True,
                 embedder_names: List[List[str]] = None,
                 embed_confs: Dict = None,
                 embed_mode: str = "sum"):
        """
        Initializes the TemporalProcessor module.

        Args:
            in_channels (int): Number of input channels (c).
            out_channels (int): Number of output channels (c').
            mode (Literal['reduce', 'increase', 'identity']): Operation mode.
                - 'reduce': Reduces time dimension by time_factor using Conv1d.
                - 'increase': Increases time dimension by time_factor using interpolation.
                - 'identity': Keeps time dimension the same (uses Conv1d if channels change).
            time_factor (int): Factor for time reduction/increase.
                               Must be >= 1. If mode is 'reduce', this defines
                               both kernel_size and stride. If mode is 'increase',
                               this is the upsampling factor. Ignored for 'identity'.
                               Defaults to 7.
            residual (bool): Whether to add a residual connection.
                                 Only applied if output time dim == input time dim.
                                 A projection is used if channels change. Defaults to True.
            use_skip (bool): Whether to add a skip connection from the original input.
                             A projection is used if channels or time dim change. Defaults to True.
        """
        super().__init__()

        if time_factor < 1:
            raise ValueError("time_factor must be >= 1")

        self.mode = mode
        self.time_factor = time_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = residual

        # 1. Layer Normalization (Adaptive - applied before main op)
        # Normalizes over the feature dimensions (g, v, c) for each batch and time step

        if embedder_names:
            emb_dict = nn.ModuleDict()
            for emb_name in embedder_names:
                emb: BaseEmbedder = EmbedderManager().get_embedder(emb_name, **embed_confs[emb_name])
                emb_dict[emb.name] = emb
            embedder_seq = EmbedderSequential(emb_dict, mode=embed_mode, spatial_dim_count=1)
            embedding_layer = torch.nn.Linear(embedder_seq.get_out_channels, 2 * self.in_channels)

            self.embedder = embedder_seq
            self.embedding_layer = embedding_layer
        else:
            self.embedder = self.embedding_layer = None
        self.norm = torch.nn.LayerNorm(self.in_channels, elementwise_affine=True)

        # 2. Main Temporal Operation Layer (Convolution or Placeholder)
        if self.mode == 'down':
            # Kernel size and stride = time_factor for reduction
            self.temporal_op = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.time_factor + nh,
                stride=self.time_factor,
                padding=nh
            )
        elif self.mode == 'up' or self.mode == 'identity':
            # Interpolation is done functionally in forward.
            # We might need a 1x1 conv if channels change.
            self.temporal_op = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1 + nh,
                stride=1
            )

        self.activation = nn.GELU()

        # 4. Projection layers for Residual and Skip connections (if needed)

        # Residual Projection: Only needed if channels change AND mode allows residual
        if (self.in_channels != self.out_channels) and self.residual:
            self.residual_proj = nn.Conv1d(
                self.in_channels, self.out_channels, kernel_size=1
            )
        else:
             self.residual_proj = nn.Identity() # Avoids None checks

    def _reshape_to_conv1d(self, x):
        """ Reshapes (b, t, g, v, c) -> (b, g*v*c, t) """
        b, t, g, v, c = x.shape
        # Permute to put time last: (b, g, v, c, t)
        x_permuted = x.permute(0, 2, 3, 4, 1)
        # Reshape to (b, C_in, t) where C_in = g*v*c
        x_reshaped = x_permuted.reshape(-1, c, t)
        return x_reshaped

    def _reshape_from_conv1d(self, x, t_out):
        """ Reshapes (b, g*v*c_out, t_out) -> (b, t_out, g, v, c_out) """
        b, _, _ = x.shape # Get batch size
        # Reshape to (b, g, v, c_out, t_out)
        x_unflattened = x.view(b, self.grid_size, self.num_vars, self.out_channels, t_out)
        # Permute back to (b, t_out, g, v, c_out)
        x_permuted = x_unflattened.permute(0, 4, 1, 2, 3)
        return x_permuted


    def forward(self, x: torch.Tensor, emb: Dict) -> torch.Tensor:
        b, t_in, g, v, c_in = x.shape

        # Store original input for skip connection
        residual = x

        if self.embedder:
            # Apply the embedding transformation (scale and shift)
            scale, shift = self.embedding_layer(self.embedder(emb)).chunk(2, dim=-1)
            x_norm = self.norm(x) * (scale + 1) + shift
        else:
            x_norm = self.norm(x)

        x_reshaped = self._reshape_to_conv1d(x_norm) # Shape: (b, C_in, t_in)

        if self.mode == 'down':
            if t_in % self.time_factor != 0:
                 # Warning or error? Let's warn for now. Padding could be added.
                 print(f"Warning: Input time dimension {t_in} is not divisible by time_factor {self.time_factor} for reduction.")
            processed = self.temporal_op(x_reshaped) # Shape: (b, C_out, t_out)
            t_out = processed.shape[-1] # Calculate actual t_out
        elif self.mode == 'up':
            t_out = t_in * self.time_factor
            # Interpolate first
            interp = F.interpolate(x_reshaped, size=t_out, mode='nearest-exact') # Use nearest-exact for better PyTorch>=1.7 compatibility
            # Adjust channels if necessary
            processed = self.temporal_op(interp) # Shape: (b, C_out, t_out)
        elif self.mode == 'identity':
            processed = self.temporal_op(x_reshaped) # Shape: (b, C_out, t_in)
            t_out = t_in

        output = self._reshape_from_conv1d(processed, t_out) # Shape: (b, t_out, g, v, c_out)

        output = self.activation(output)

        if self.residual:
            if self.in_channels != self.out_channels:
                # Project the original *unnormalized* input reshaped
                residual_reshaped = self._reshape_to_conv1d(residual)
                residual = self.residual_proj(residual_reshaped) # Shape: (b, C_out, t_in)
                residual = self._reshape_from_conv1d(residual, t_in) # Shape: (b, t_in, g, v, c_out)
            output = output + residual

        return output