import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from scipy.special import sph_harm_y

from ..grids.grid_utils import estimate_healpix_cell_radius_rad, rotate_coord_system
from ..grids.grid_layer import GridLayer


class RandomFourierLayer(nn.Module):
    """
    A neural network layer that applies a Random Fourier Feature transformation.

    :param in_features: Number of input features (default is 3).
    :param n_neurons: Number of neurons in the layer, which will determine the dimensionality of the output (default is 512).
    :param wave_length: Scaling factor for the input tensor, affecting the frequency of the sine and cosine functions (default is 1.0).
    """

    def __init__(
            self,
            in_features: int = 3,
            n_neurons: int = 512,
            wave_length: float = 1.0,
            wave_length_2: Optional[float] = None
    ) -> None:
        """
        Initialize the random Fourier feature layer.

        :param in_features: Number of input features.
        :param n_neurons: Number of neurons (output features).
        :param wave_length: Primary wavelength for random features.
        :param wave_length_2: Optional secondary wavelength for mixed scales.
        :return: None.
        """
        super().__init__()
        # Initialize the weights parameter with a random normal distribution

        if wave_length_2 is None:
            wave_length_2 = wave_length
  
        weights = torch.concat(
            (
                torch.randn(in_features, n_neurons // 4) / wave_length,
                torch.randn(in_features, n_neurons // 4) / wave_length_2,
            ),
            dim=1,
        )

        self.register_parameter(
            "weights",
            torch.nn.Parameter(
                weights, requires_grad=False
            )
        )
        self.weights: nn.Parameter
        # Scaling constant to normalize the output
        self.constant: float = math.sqrt(2 / n_neurons)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the layer, applying Random Fourier Feature transformation.

        :param in_tensor: Input tensor of shape ``(b, ..., in_features)``.
        :return: Transformed output tensor of shape ``(b, ..., n_neurons)``.
        """
        # Normalize input tensor by the wave length
        in_tensor = in_tensor
        # Apply a linear transformation using random weights
        out_tensor = 2 * torch.pi * in_tensor @ self.weights
        # Apply sine and cosine functions and concatenate the results
        out_tensor = self.constant * torch.cat(
            [torch.sin(out_tensor), torch.cos(out_tensor)], dim=-1
        )
        return out_tensor


class TimeScaleLayer(nn.Module):
    def __init__(
            self,
            in_features: int = 1,
            n_neurons: int = 512,
            # Pass periods in the unit of your input data (Hours)
            time_scales: Optional[Sequence[float]] = None,
            time_min: float = 0.0,
            time_max: float = 1.0,
            use_linear: bool = True
    ) -> None:
        """
        Initialize a time-scale embedding layer.

        :param in_features: Number of input features.
        :param n_neurons: Output feature count.
        :param time_scales: Periods for sinusoidal components.
        :param time_min: Minimum time value for normalization.
        :param time_max: Maximum time value for normalization.
        :param use_linear: Whether to include a linear trend component.
        :return: None.
        """
        super().__init__()

        self.use_linear: bool = use_linear
        # 1. Setup Periodic Components
        # We need 2 features (sin + cos) per scale
        if time_scales is None:
            time_scales = [168.0, 8766.0]

        periodic_scales = torch.tensor(time_scales, dtype=torch.float32)

        self.register_buffer('periodic_scales', periodic_scales.view(1, 1, -1))
        self.periodic_scales: torch.Tensor

        n_periodic_features = len(time_scales) * 2

        # 2. Setup Linear Trend Component
        # The rest of the neurons go to the linear trend
        
        n_linear_features = n_neurons - n_periodic_features if use_linear else 0

        self.time_min: float = time_min
        self.time_range: float = time_max - time_min + 1e-8

        # Projection for the linear trend (Global warming/Decadal shifts)
        self.linear_trend: nn.Module = nn.Linear(in_features, n_linear_features) if use_linear else nn.Identity()

        # We also want to learn how to mix the sin/cos features
        out_features_periodic = n_periodic_features if use_linear else n_neurons
        self.periodic_projection: nn.Module = nn.Linear(n_periodic_features, out_features_periodic)


    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """
        Encode time values with periodic and optional linear components.

        :param times: Tensor of shape ``(b, t)`` or ``(b, t, 1)`` with time values.
        :return: Embedded tensor of shape ``(b, t, n_neurons)``.
        """
       # times = time_zooms[max(time_zooms.keys())]

        if times.dim() == 2:
            times = times.unsqueeze(-1)  # Ensure (B, S, 1)

        # --- A. Linear Trend (Global Time) ---
        # Normalize to 0-1 range for stability
        if self.use_linear:
            normalized_time = (times - self.time_min) / self.time_range
            trend_embed = self.linear_trend(normalized_time)

        # Shape magic to broadcast: (B, S, 1) / (Num_Scales) -> (B, S, Num_Scales)
        phase = 2 * torch.pi * times / self.periodic_scales

        # Compute Sin and Cos to fix phase ambiguity
        sin_features = torch.sin(phase)
        cos_features = torch.cos(phase)

        # Concatenate and project
        periodic_raw = torch.cat([sin_features, cos_features], dim=-1)
        periodic_embed = self.periodic_projection(periodic_raw)

        if self.use_linear:
            return torch.cat([trend_embed, periodic_embed], dim=-1)
        else:
            return periodic_embed


class SinusoidalLayer(nn.Module):
    """
    Sinusoidal timestep embeddings for diffusion steps.

    This function generates sinusoidal positional embeddings for each diffusion step,
    where the frequencies are controlled by `max_period`. The embeddings are intended to
    be used in models that benefit from positional information of diffusion steps.

    :param in_channels: Number of input features.
    :param dim: Dimensionality of the output embeddings.
    :param max_period: Controls the minimum frequency of the embeddings.
                       Higher values lead to more gradual frequency changes.
    """
    def __init__(self, in_channels: int, max_period: int = 10000) -> None:
        """
        Initialize sinusoidal embeddings.

        :param in_channels: Number of input features.
        :param max_period: Controls the minimum frequency of the embeddings.
        :return: None.
        """
        super().__init__()
        self.in_channels: int = in_channels
        self.freqs: torch.Tensor = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=in_channels // 2,
                                                 dtype=torch.float32) / (in_channels // 2)
        )

    def forward(self, diffusion_steps: torch.Tensor) -> torch.Tensor:
        """
        :param diffusion_steps: A 1-D tensor of shape ``(b,)`` with diffusion steps.
        :return: Positional embeddings of shape ``(b, in_channels)``.
        """

        # Calculate arguments for sine and cosine functions
        args = diffusion_steps.unsqueeze(-1).float() @ self.freqs.unsqueeze(0).to(diffusion_steps.device)
        # Combine sine and cosine embeddings along the last dimension
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Pad with a zero column if `dim` is odd
        if self.in_channels % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return embedding


def get_mg_embeddings(mg_emb_confs: Dict[str, Any], grid_layers: Dict[str, GridLayer]) -> nn.ParameterDict:
    """
    Build multigrid embeddings for each zoom.

    :param mg_emb_confs: Configuration dict containing zooms, features, variables, and init methods.
    :param grid_layers: Mapping from zoom string to grid layer.
    :return: ParameterDict of embeddings by zoom.
    """
    mg_emeddings = nn.ParameterDict()
    diff_mode = mg_emb_confs.get('diff_mode', True)
    
    amplitude = 1
    wavelength_max = None
    for zoom, features, n_variables, init_method in zip(
        mg_emb_confs['zooms'],
        mg_emb_confs['features'],
        mg_emb_confs["n_variables"],
        mg_emb_confs['init_methods'],
    ):
        
        wavelength_min = estimate_healpix_cell_radius_rad(grid_layers[str(zoom)].adjc.shape[0])

        mg_emeddings[str(zoom)] = get_mg_embedding(
            grid_layers[str(zoom)],
            features,
            n_variables,
            init_mode=init_method,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            amplitude=amplitude)
        
        if diff_mode:
            wavelength_max = wavelength_min
            amplitude = 1e-3
        else:
            wavelength_max = None
            amplitude = 1

    return mg_emeddings


def get_mg_embedding(
        grid_layer_emb: GridLayer,
        features: int,
        n_variables: int,
        init_mode: str = 'fourier_sphere',
        wavelength: float = 1,
        wavelength_min: Optional[float] = None,
        wavelength_max: Optional[float] = None,
        random_rotation: bool = False,
        amplitude: float = 1
) -> nn.Parameter:
    """
    Generate a multigrid embedding parameter tensor.

    :param grid_layer_emb: Grid layer providing coordinates.
    :param features: Number of embedding features.
    :param n_variables: Number of variables to repeat embeddings for.
    :param init_mode: Initialization strategy.
    :param wavelength: Base wavelength for Fourier features.
    :param wavelength_min: Minimum wavelength for spherical harmonics.
    :param wavelength_max: Maximum wavelength for spherical harmonics.
    :param random_rotation: Whether to randomize the coordinate system.
    :param amplitude: Amplitude scaling for embeddings.
    :return: Trainable embedding parameter of shape ``(n_variables, n_cells, features)``.
    """
    
    coords = grid_layer_emb.get_coordinates()       

    clon, clat = coords[...,0], coords[...,1]

    if init_mode == 'random':
        embs = amplitude*torch.randn(1, coords.shape[-2], features)
    
    elif 'fourier_sphere' == init_mode:
        fourier_layer = RandomFourierLayer(in_features=2, n_neurons=features, wave_length=2*wavelength*torch.pi)
        embs = amplitude*fourier_layer(coords).squeeze(dim=-2)

    elif 'fourier' == init_mode:

        x = torch.cos(clat) * torch.cos(clon)
        y = torch.cos(clat) * torch.sin(clon)
        z = torch.sin(clat)

        coords_3d = torch.stack((x, y, z), dim=-1).float()

        fourier_layer = RandomFourierLayer(in_features=3, n_neurons=features, wave_length=wavelength)
        embs = amplitude*fourier_layer(coords_3d)
    
    elif "spherical_harmonics" == init_mode:

        if wavelength_min is not None:
            L_nyq = int(math.floor(math.pi / (2.0 * wavelength_min)))
        else:
            dtheta = estimate_healpix_cell_radius_rad(grid_layer_emb.adjc.shape[0])
            L_nyq = int(math.floor(math.pi / (2.0 * dtheta)))

        l_min = 0 if wavelength_max is None else math.floor(math.pi / (2.0 * wavelength_max))
        l_max = max(l_min + 1, L_nyq)


        ls = torch.randint(l_min, int(l_max), (features,))
        # sample m in [-l, l], excluding the empty interval if l=0
        ms = torch.stack([
                (-int(l.item()) + 2 * torch.randint(0, int(l.item()) + 1, (1,))).squeeze(0)
                for l in ls
            ])
        
        embs = torch.zeros(1, coords.shape[1], features)

        for k in range(features):
            l = int(ls[k].item())
            m = int(ms[k].item())

            if random_rotation:
                rotation_lon = torch.rand((1,))*torch.pi
                rotation_lat = torch.rand((1,))*torch.pi/2-torch.pi/4
                clon, clat = rotate_coord_system(clon, clat, rotation_lon, rotation_lat)
                clon, clat = clon.unsqueeze(dim=-1),clat.unsqueeze(dim=-1)

            Ylm = sph_harm_y(l, abs(m), clat, clon)

            if m == 0:
                Y_real = torch.as_tensor(Ylm.real, dtype=torch.float32)
            elif m > 0:
                Y_real = math.sqrt(2.0) * torch.as_tensor(Ylm.real, dtype=torch.float32)
            else:  
                Y_real = math.sqrt(2.0) * torch.as_tensor(Ylm.imag, dtype=torch.float32)

            embs[..., k] = amplitude * Y_real.view(1, -1)


    embs = embs.repeat_interleave(n_variables, dim=0)
    
    embs = nn.Parameter(embs, requires_grad=True)

    return embs
