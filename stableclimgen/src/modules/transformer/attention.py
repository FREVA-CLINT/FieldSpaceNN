import torch
import torch.nn as nn

from .transformer_modules import MultiHeadAttentionBlock

class ResLayer(nn.Module):
    def __init__(self, model_dim):
        """
        A residual layer that applies a learnable scalar weighting (gamma) to an MLP-based transformation of the input.

        :param model_dim: Dimensionality of the input and output of the MLP layer.
        """

        super().__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim)*1e-6, requires_grad=True)

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(model_dim, elementwise_affine=True),
            nn.Linear(model_dim, model_dim , bias=False),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim, bias=False)
        )
    
    def forward(self, x):
        return x + self.gamma * self.mlp_layer(x)

class ChannelVariableAttention(nn.Module):
    def __init__(self, model_dim, n_chunks_channel, n_head_channels, model_dim_out=None, with_res=True):
        super().__init__()
        """
        Channel-wise variable attention module for applying multi-head attention across variable dimensions.

        :param model_dim: Dimensionality of the input feature space.
        :param n_chunks_channel: Number of chunks to divide the channel dimension into for attention.
        :param n_head_channels: Number of channels per attention head.
        :param model_dim_out: Dimensionality of the output feature space (if different from model_dim).
        :param with_res: Whether to use a residual connection with the attention mechanism.
        """
        
        self.n_chunks_channels = n_chunks_channel

        self.with_res = with_res

        if model_dim_out is None:
            model_dim_out = model_dim
            self.res_lin_layer = nn.Identity()

        elif with_res:
            self.res_lin_layer = nn.Linear(model_dim, model_dim_out, bias=False)

        model_dim_att = model_dim // n_chunks_channel
        model_dim_att_out = model_dim_out // n_chunks_channel

        self.layer_norm = nn.LayerNorm(model_dim_att, elementwise_affine=True)

        n_heads = model_dim_att//n_head_channels if model_dim_att > n_head_channels else 1
        self.MHA = MultiHeadAttentionBlock(
            model_dim_att, model_dim_att_out, n_heads, input_dim=model_dim_att, qkv_proj=True
            )   

        if with_res:
            self.gamma = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
            self.res_layer = ResLayer(model_dim_out)
    

    def forward(self, x, mask=None):
        b,n,nv,f=x.shape
        x = x.view(b,n,nv,f)

        if self.with_res:
            x_res = self.res_lin_layer(x)

        x = x.view(b*n,nv,f)
        x = x.view(b*n,nv*self.n_chunks_channels,-1)

        x = self.layer_norm(x)
        
        mask_chunk=mask
        if mask_chunk is not None:
            mask_chunk = mask_chunk.view(b*n,nv).repeat_interleave(self.n_chunks_channels,dim=1)
            
        x = self.MHA(q=x, k=x, v=x, mask=mask_chunk)
        x = x.view(b*n,nv,-1)
        x = x.view(b,n,nv,-1)

        if self.with_res:
            x = x_res + self.gamma * x
            x = self.res_layer(x)

        if mask is not None:
            mask = mask.view(b, x.shape[1], -1)
            mask[mask.sum(dim=-1)!=mask.shape[-1]] = False
            mask = mask.view(b,n,nv)

        return x, mask
    

class MultiGridChannelAttention(nn.Module):
    """
    A neural network module for multi-grid channel attention. This class combines 
    an attention mechanism across multiple input channels with an optional output 
    layer for further transformation.

    Attributes:
        att_layer (nn.Module): The channel variable attention layer.
        mlp_layer_out (nn.Module): The output MLP layer or an identity mapping if 
                                   output_layer is set to False.
    """

    def __init__(self, model_dims_in: torch.Tensor, model_dim_out: int, n_chunks: int = 2, n_head_channels: int = 16) -> None:
        """
        Initializes the MultiGridChannelAttention module.

        :param model_dims_in: Tensor containing the dimensions of input models.
        :param model_dim_out: The output dimension for the model.
        :param n_chunks: Number of chunks for splitting the attention heads. Defaults to 4.
        :param n_head_channels: Number of channels per attention head. Defaults to 16.
        :param output_layer: Flag to determine if the output layer is used. Defaults to False.
        """
        super().__init__()

        # Calculate the total input model dimensions
        model_dims_in_total = int(model_dims_in.sum())

        # Determine the output dimension for the attention layer
        
        model_dim_att_out = model_dim_out

        # Initialize the channel variable attention layer
        self.att_layer = ChannelVariableAttention(
            model_dims_in_total, len(model_dims_in), n_head_channels, model_dim_out=model_dim_att_out
        )


    def forward(self, x_levels: list, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the MultiGridChannelAttention module.

        :param x_levels: List of input tensors for different levels.
        :param mask: Optional mask tensor. Defaults to None.
        :return: Transformed output tensor after applying attention and MLP layers.
        """
        # Concatenate input levels and apply the attention layer
        x = self.att_layer(torch.concat(x_levels, dim=-1), mask=mask)[0]

        # Apply the output MLP layer or identity
        return x

        
class MultiGridAttention_masked(nn.Module):
    def __init__(self, model_dims_in, model_dim_out, n_head_channels, n_chunks: int = 2, with_channel_att=False):
        super().__init__()
        """
        Channel-wise variable attention module for applying multi-head attention across variable dimensions.

        :param model_dim: Dimensionality of the input feature space.
        :param n_chunks_channel: Number of chunks to divide the channel dimension into for attention.
        :param n_head_channels: Number of channels per attention head.
        :param model_dim_out: Dimensionality of the output feature space (if different from model_dim).
        :param with_res: Whether to use a residual connection with the attention mechanism.
        """
        
        n_grids = len(model_dims_in)

        self.n_chunks = n_chunks
        model_dim_att = int(model_dims_in[0]) // (n_chunks)
        model_dim_att_out = model_dim_out // (n_grids*n_chunks)

        self.layer_norm = nn.LayerNorm(model_dim_att, elementwise_affine=True)

        n_heads = model_dim_att//n_head_channels if model_dim_att > n_head_channels else 1

        self.MHA = MultiHeadAttentionBlock(
            model_dim_att, model_dim_att_out, n_heads, input_dim=model_dim_att, qkv_proj=True
            )   

        self.gamma = nn.Parameter(torch.ones(model_dim_out)*1e-6, requires_grad=True)
       
        self.res_layer = ResLayer(model_dim_out)
    

    def forward(self, x_levels, mask_levels=None):
        if mask_levels is not None:
            mask = torch.stack(mask_levels, dim=-1)
        else:
            mask =None

        x = torch.stack(x_levels, dim=-2)

        b,n,nv,ng,f=x.shape
        
        if mask is not None:
            weights = (mask==False) + 1e-6
            weights = weights/(weights.sum(dim=-1,keepdim=True))
            x_res = (x * weights.view(b,n,nv,ng,1)).sum(dim=-2)
        else:
            x_res = x.mean(dim=-2, keepdim=True)


        x = x.view(b*n,nv,ng,-1)
        x = x.view(b*n,nv,ng,self.n_chunks,-1)
        x = x.view(b*n,nv*ng*self.n_chunks,-1)
        
        x = self.layer_norm(x)

        mask_chunk=mask
        if mask_chunk is not None:
            mask_chunk = mask_chunk.view(b*n,nv,ng,1).repeat_interleave(self.n_chunks,dim=-1)
            mask_chunk = mask_chunk.view(b*n,nv*ng*self.n_chunks)
            
        x = self.MHA(q=x, k=x, v=x, mask=mask_chunk)
        x = x.view(b*n,nv,-1)
        x = x.view(b,n,nv,-1)

        x = x_res + self.gamma * x
        x = self.res_layer(x)

        if mask is not None:
            mask = mask.view(b,n, nv, ng)
            mask = mask.sum(dim=-1)==mask.shape[-1]
            mask = mask.view(b,n,nv)

        return x, mask_chunk



    

