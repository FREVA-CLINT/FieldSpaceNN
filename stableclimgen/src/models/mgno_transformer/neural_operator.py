import torch
import torch.nn.functional as F
import torch.nn as nn

from .attention import ChannelVariableAttention,ResLayer
from ...utils.grid_utils_icon import get_distance_angle
from .icon_grids import RelativeCoordinateManager

def von_mises(d_phi, kappa):
    vm = torch.exp(kappa * torch.cos(d_phi))
    return vm

def cosine(d_phi):
    return torch.cos(d_phi)

def normal_dist(d_dists, sigma):
    nd = torch.exp(-0.5 * (d_dists / sigma) ** 2)
    return nd


class NoLayer(nn.Module):
    """
    The NoLayer class implements a custom neural network layer that processes
    hierarchical grid-based data using spatial transformations, such as 
    distance-based and angular-based weighting. This class is useful for tasks
    requiring structured data analysis, particularly when working with grid-like
    data structures and requiring intricate spatial relationships.

    Attributes:
        grid_layers (dict): Dictionary containing class GridLayer
        global_level_out (int): The output global level index
        global_level (int): The input global level index 
        n_phi (int): Number of angular partitions for computing angular weights.
        n_dist (int): Number of distance partitions for computing distance weights.
        n_sigma (int): Number of sigma values for computing Gaussian distance weights.
        kappa_init (float): Initial value for kappa in von Mises distribution.
        use_von_mises (bool): Flag indicating whether to use von Mises distribution 
            for angular calculations.
        phi_channel_attention (bool): Flag to enable/disable angular channel attention.
        dist_channel_attention (bool): Flag to enable/disable distance channel attention.
        sigma_channel_attention (bool): Flag to enable/disable sigma channel attention.
        with_mean_res (bool): Whether to use residual connections with mean feature maps.
        with_channel_res (bool): Whether to apply residual connections within channel 
            attention layers.
        dist_learnable (bool): Flag to determine if distances are learnable parameters.
        sigma_learnable (bool): Flag to determine if sigma values are learnable.
        nh_projection (bool): Flag for non-hierarchical projection.
        angular_dist_calc (bool): Indicates if angular distance calculations are active.
        dist_dist_calc (bool): Indicates if distance calculations are active.
        periodic_fov (bool or None): Specifies if the field of view is periodic.
        polar (bool): Specifies if the coordinate system is polar.
        coord_system (str): The type of coordinate system used ('polar' by default).
        sigma (nn.Parameter): Learnable sigma values for Gaussian distance calculations.
        dists (nn.Parameter): Learnable distance values.
        phi (nn.Parameter): Angular partitions for calculating angular weights.
        kappa (nn.Parameter): Kappa parameter for von Mises distribution (if applicable).
        dist_channel_attention (nn.Module): Channel attention module for distance-based weights.
        sigma_channel_attention (nn.Module): Channel attention module for sigma-based weights.
        phi_channel_attention (nn.Module): Channel attention module for angular-based weights.
        res_layer_dist (nn.Module): Residual layer for distance-based attention.
        res_layer_sigma (nn.Module): Residual layer for sigma-based attention.
        res_layer_phi (nn.Module): Residual layer for angular-based attention.
        gamma_res_dist (nn.Parameter): Learnable scaling factor for distance-based residuals.
        gamma_res_sigma (nn.Parameter): Learnable scaling factor for sigma-based residuals.
        gamma_res_phi (nn.Parameter): Learnable scaling factor for angular-based residuals.
    """
    def __init__(self,
                 grid_layers,
                 global_level_in,
                 global_level_out, 
                 model_dim_in,
                 model_dim_out,
                 kernel_settings,
                 n_head_channels: int=16,
                 precompute_rel_coordinates: bool=False
                ) -> None: 
        """
        Initializes the NoLayer class with parameters related to grid-based 
        spatial data processing.

        :param grid_layers: Dictionary containing grid layer configurations.
        :param global_level_in: The input global level for processing.
        :param global_level_out: The output global level for processing.
        :param model_dim_in: The dimensionality of the input features.
        :param model_dim_out: The dimensionality of the output features.
        :param kernel_settings: A dictionary specifying settings for distance 
            and angular kernel computations, such as 'n_phi', 'n_dists', etc.
        :param n_head_channels: Number of channels per head in attention mechanisms.
        """
        super().__init__()
        
        self.grid_layers = grid_layers
        self.global_level_out = global_level_out
        self.global_level = global_level_in

        self.rel_coord_mngr = RelativeCoordinateManager(
            self.grid_layers[str(global_level_in)],
            self.grid_layers[str(global_level_out)],
            nh_input= kernel_settings['nh_projection'],
            precompute=precompute_rel_coordinates,
            coord_system='polar')
  
        # Kernel settings and initialization
        n_phi = kernel_settings['n_phi']
        n_dist = kernel_settings['n_dists']
        n_sigma = kernel_settings['n_sigma']
        kappa_init = kernel_settings['kappa_init']
        use_von_mises = kernel_settings['use_von_mises']
        phi_channel_attention = kernel_settings['phi_att']
        dist_channel_attention = kernel_settings['dist_att']
        sigma_channel_attention = kernel_settings['sigma_att']
        with_mean_res = kernel_settings['with_mean_res']
        with_channel_res = kernel_settings['with_channel_res']

        dist_learnable = kernel_settings['dists_learnable']
        sigma_learnable = kernel_settings['sigma_learnable']
        nh_projection = kernel_settings['nh_projection']

        min_sigma = 1e-4
        dist_max = self.grid_layers[str(global_level_out)].min_dist

        # Flags and parameters
        self.with_mean_res = with_mean_res
        self.nh_projection= nh_projection
        self.use_von_mises = use_von_mises

        if use_von_mises:
            phi_min = -torch.pi
        else:
            phi_min = 0

        self.angular_dist_calc = False
        self.dist_dist_calc = False
    
        self.periodic_fov=None
        self.polar=True
        self.coord_system = 'polar'
        model_dim_enhanced = model_dim_in

        # Distance, sigma, and angular kernel initialization
        if n_dist>1:
            model_dim_enhanced *= n_dist
            dist = torch.linspace(0, dist_max, n_dist)
            self.dists = nn.Parameter(dist, requires_grad=dist_learnable)
            self.dist_dist_calc = True
            n_sigma = n_sigma if n_sigma >=1 else 1


        if n_sigma>1:
            model_dim_enhanced *= n_sigma
            sigma = torch.linspace(min_sigma, dist_max/2, n_sigma)
            self.sigma = nn.Parameter(sigma, requires_grad=sigma_learnable)
            self.dist_dist_calc = True

        elif n_sigma==1:
            sigma = torch.tensor(dist_max/2)
            self.sigma = nn.Parameter(sigma, requires_grad=sigma_learnable)
            self.dist_dist_calc = True

        if n_phi>1:
            model_dim_enhanced *= n_phi
            phi = torch.linspace(phi_min, torch.pi, n_phi+1)[:-1]
            self.phi = nn.Parameter(torch.tensor(phi), requires_grad=False)
            self.angular_dist_calc = True
            if use_von_mises:
                self.kappa = nn.Parameter(torch.tensor(kappa_init), requires_grad=False)

        # Initialize channel attention mechanisms
        if dist_channel_attention and n_dist>1:
            model_dim = model_dim_enhanced // n_dist
            self.dist_channel_attention = ChannelVariableAttention(model_dim, 1, n_head_channels, with_res=with_channel_res)

            if with_mean_res:
                self.gamma_res_dist = nn.Parameter(torch.ones(model_dim_in)*1e-6, requires_grad=True)
                self.res_layer_dist = ResLayer(model_dim)

        if sigma_channel_attention and n_sigma>1:
            model_dim = model_dim_enhanced // n_sigma
            self.sigma_channel_attention = ChannelVariableAttention(model_dim, 1, n_head_channels, with_res=with_channel_res)
            
            if with_mean_res:
                self.gamma_res_sigma = nn.Parameter(torch.ones(model_dim_in)*1e-6, requires_grad=True)
                self.res_layer_sigma = ResLayer(model_dim)

        if phi_channel_attention and n_phi>1:
            model_dim = model_dim_enhanced // n_phi
            self.phi_channel_attention = ChannelVariableAttention(model_dim, 1, n_head_channels, with_res=with_channel_res)
            
            if with_mean_res:
                self.gamma_res_phi = nn.Parameter(torch.ones(model_dim_in)*1e-6, requires_grad=True)
                self.res_layer_phi = ResLayer(model_dim)
        
        if model_dim_enhanced != model_dim_out:
            self.lin_layer_out = nn.Linear(model_dim_enhanced, model_dim_out, bias=False)
        else:
            self.lin_layer_out = nn.Identity()


    def forward(self, x, indices_layers=None, coordinates=None, coordinates_out=None, sample_dict=None, mask=None):
        """
        Forward pass of the NoLayer class. This method processes input features by
        computing spatial relationships and applying a custom projection using
        hierarchical grid-based data and distance-angle calculations.

        :param x: Input tensor of shape (B, N, NV, F), where B is the batch size, 
                N is the sequence length, NV is the number of variables, 
                and F is the feature size.
        :param indices_layers: Dictionary containing grid indices for different 
                            hierarchical levels. Used to derive coordinates if 
                            `coordinates` is not provided. Default is None.
        :param coordinates: Tensor of input coordinates with shape (N_C, B, Seq_In, ...).
                            If not provided, coordinates are generated from `indices_layers`.
                            Default is None.
        :param coordinates_out: Tensor of output coordinates with shape (N_C, B, Seq_Out, ...).
                                If not provided, coordinates are derived from `indices_layers`
                                for the output level. Default is None.
        :param sample_dict: Dictionary containing additional data for non-hierarchical 
                            (nh) projection, used if `nh_projection` is True. Default is None.
        :param mask: Optional tensor mask for indicating valid inputs. Its shape may 
                    change depending on the input transformations. Default is None.

        :return: A tuple containing:
                - x: Transformed tensor after projection, with shape (B, N, NV, F).
                - mask: Updated mask tensor if provided, or None.
        """
        
        nh_projection = self.nh_projection

        coordinates_rel = self.rel_coord_mngr(indices_in=indices_layers[self.global_level],
                                              indices_out=indices_layers[self.global_level_out],
                                              coordinates_in=coordinates,
                                              coordinates_out=coordinates_out)
  
        if nh_projection:
            x, mask = self.grid_layers[str(self.global_level)].get_nh(x, 
                                                                    indices_layers[self.global_level], 
                                                                    sample_dict, 
                                                                    mask=mask)
            
        b, n, seq_out, seq_in = coordinates_rel[0].shape
        nv, nc = x.shape[-2:]
        x = x.view(b,n,seq_in,nv,nc)
        
        if mask is not None:
            mask = mask.view(b,n,seq_in,nv)
        # Project `x` using the relative coordinates
        x, mask = self.project(x, coordinates_rel, mask=mask)      
                       
        return x, mask


    def project(self, x: torch.Tensor, coordinates_rel: tuple,  mask=None):
        """
        Applies a projection to the input tensor `x` using relative distance and 
        angle weights. The method calculates weighted features using both distance 
        and angular relationships, applying various attention mechanisms to 
        emphasize important channels.

        :param x: Input tensor of shape (B, N, seq_in, NV, F), where:
                - B: Batch size
                - N: Number of sequences
                - seq_in: Input sequence length
                - NV: Number of variables
                - F: Feature dimension
        :param coordinates_rel: A tuple containing two tensors (dists, phis) with dimensions:
                (B, N, seq_out, seq_in), where:
                - seq_out: Output sequence length
        :param mask: Optional tensor used to mask out invalid values in `x`. 
                    The shape should be compatible with the shape of `x`. 
                    If provided, it is used to modify the weights to ignore 
                    masked elements.

        :return: A tuple containing:
                - x: Transformed tensor with shape (B, N*seq_out, NV, ...), where seq_out 
                    represents the transformed sequence dimension. The exact 
                    shape of the output tensor depends on the input coordinates.
                - mask: Updated mask tensor, if provided, with shape (B, N*seq_out, NV).
        """
        b, n, seq_in, nv, f = x.shape

        if mask is not None and self.with_mean_res:
            weights_res = torch.ones(x.shape[:-1], device=x.device)
            weights_res.masked_fill_(mask, 0)
            x_res = (x * weights_res.unsqueeze(dim=-1)).sum(dim=-3, keepdim=True).unsqueeze(dim=2)

        elif self.with_mean_res:
            x_res = (x).sum(dim=-3, keepdim=True).unsqueeze(dim=2)

        else:
            x_res = None

        x = x.view(b,n,seq_in,nv,1,1,1,f)
        
        sigma = self.sigma.clamp(min=1e-4)

        angular_weights = dist_weights = None

        dists, phis = coordinates_rel

        if self.dist_dist_calc:
            if hasattr(self, "dists"):
                d_dists = dists.unsqueeze(dim=-1) - self.dists.unsqueeze(dim=0)
            else:
                d_dists = dists.unsqueeze(dim=-1)
            dist_weights = normal_dist(d_dists.unsqueeze(dim=-1), sigma.unsqueeze(dim=0))
            dist_weights = dist_weights.unsqueeze(dim=-1)

        if self.angular_dist_calc:
            d_phis = phis.unsqueeze(dim=-1) - self.phi.unsqueeze(dim=0)
            if self.use_von_mises:
                angular_weights = von_mises(d_phis, self.kappa)
            else:
                angular_weights = cosine(d_phis)

            angular_weights = angular_weights.masked_fill(dists.unsqueeze(dim=-1)<1e-10, 1)

            angular_weights = angular_weights.unsqueeze(dim=-2).unsqueeze(dim=-3)


        if angular_weights is not None and dist_weights is not None:
            weights = dist_weights*angular_weights

        elif dist_weights is not None:
            weights = dist_weights

        elif angular_weights is not None:
            weights = angular_weights

        weights = weights.unsqueeze(dim=4).repeat_interleave(nv, dim=4)
        
        sng = weights.sign()

        if mask is not None:

            weights = weights.masked_fill(mask.view(b,n,1,seq_in,nv,1,1,1), -1e30 if x.dtype == torch.float32 else -1e4)

        weights = F.softmax(weights, dim=3)

        weights = weights*sng
        
        #inflate target dimension in x and feature dimension in weights
        x = (x.unsqueeze(dim=2) * weights.unsqueeze(dim=-1)).sum(dim=3)


        b, n, seq_out, nv, n_dist, n_sigma, n_phi, nc = x.shape
        x = x.view(b, n*seq_out, nv, n_dist, n_sigma, n_phi, nc)

        
        if mask is not None:
            mask = mask.sum(dim=-2)==mask.shape[-2]
            mask = mask.unsqueeze(dim=-2).repeat_interleave(seq_out, dim=-2)
            mask = mask.view(b,n*seq_out,nv)

        # Apply channel attention mechanisms if available
        if hasattr(self, 'phi_channel_attention'):
            x = x.transpose(3,-2).reshape(b, n*seq_out, nv*n_phi, -1)
            mask_phi=None
            if mask is not None:
                mask_phi = mask.unsqueeze(dim=-1).repeat_interleave(n_phi, dim=-1).view(b, n*seq_out, nv*n_phi)
            x = self.phi_channel_attention(x, mask=mask_phi)[0]
            x = x.view(b,n*seq_out, nv*n_phi, n_sigma, n_dist, nc)

            if x_res is not None:
                x = (x.view(b,n,seq_out,nv,n_phi,n_sigma,n_dist,nc) * self.gamma_res_phi + x_res.view(b,n,1,nv,1,1,1,nc))
                x = x.view(b,n,seq_out, nv*n_phi, -1)
                x = self.res_layer_phi(x)
                
            x = x.view(b, n*seq_out, nv, n_phi, n_sigma, n_dist, nc)
            x = x.transpose(3,-2)


        if hasattr(self, 'dist_channel_attention'):
            x = x.reshape(b, n*seq_out, nv*n_dist, -1)
            mask_dist=None
            if mask is not None:
                mask_dist = mask.unsqueeze(dim=-1).repeat_interleave(n_dist, dim=-1).view(b, n*seq_out, nv*n_dist)
            x = self.dist_channel_attention(x, mask=mask_dist)[0]

            x = x.view(b,n*seq_out, nv*n_dist, n_sigma, n_phi, nc)
            if x_res is not None:
                x = (x.view(b,n,seq_out,nv,n_dist,n_sigma,n_phi,nc) * self.gamma_res_dist + x_res.view(b,n,1,nv,1,1,1,nc))
                x = x.view(b, n, seq_out, nv*n_dist, -1)
                x = self.res_layer_dist(x)
            x = x.view(b,n*seq_out, nv, n_dist, n_sigma, n_phi, nc)


        if hasattr(self, 'sigma_channel_attention'):
            x = x.transpose(3,4).reshape(b,n*seq_out, nv*n_sigma, -1)
            mask_sigma=None
            if mask is not None:
                mask_sigma = mask.unsqueeze(dim=-1).repeat_interleave(n_sigma, dim=-1).view(b, n*seq_out, nv*n_sigma)
            x = self.sigma_channel_attention(x, mask=mask_sigma)[0]
            x = x.view(b,n*seq_out, nv*n_sigma, n_dist, n_phi, nc)

            if x_res is not None:
                x = (x.view(b,n,seq_out, nv, n_sigma, n_dist, n_phi, nc) * self.gamma_res_sigma + x_res.view(b,n,1,nv,1,1,1,nc))
                x = x.view(b,n,seq_out, nv*n_sigma, -1)
                x = self.res_layer_sigma(x)

            x = x.view(b,n*seq_out, nv,n_sigma, n_dist, n_phi, nc)
            x = x.transpose(3,4)

        x = x.reshape(b,n*seq_out, nv, -1)
        x = self.lin_layer_out(x)

        return x, mask