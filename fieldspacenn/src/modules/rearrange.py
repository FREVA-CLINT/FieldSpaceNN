from typing import Any, Dict, Optional

from einops import rearrange
import torch

from ..modules.grids.grid_layer import GridLayer
from ..modules.cnn.cnn_base import EmbedBlock
from ..modules.base import IdentityLayer

class RearrangeBlock(EmbedBlock):
    """
    A block that rearranges input tensor shapes before and after applying a specified function.

    :param fn: The function to apply to the rearranged input.
    :param pattern: The rearrangement pattern for the input tensor.
    :param reverse_pattern: The reverse pattern to apply after the function is executed.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn: Any, 
                 pattern: str, 
                 reverse_pattern: str, 
                 spatial_dim_count: int = 1, 
                 seq_len: Optional[int] = None, 
                 proj_layer_q: Optional[torch.nn.Module] = None,
                 proj_layer_kv: Optional[torch.nn.Module] = None,
                 out_layer: Optional[torch.nn.Module] = None):
        """
        Initialize the rearrangement block.

        :param fn: Function or module to apply to the rearranged input.
        :param pattern: Einops pattern for rearranging the input.
        :param reverse_pattern: Einops pattern for restoring the output.
        :param spatial_dim_count: Number of spatial dimensions.
        :param seq_len: Optional sequence length for chunked processing.
        :param proj_layer_q: Optional projection layer for queries.
        :param proj_layer_kv: Optional projection layer for keys/values.
        :param out_layer: Optional output projection layer.
        :return: None.
        """
        
        super().__init__()
        self.fn: Any = fn
        self.pattern: str = pattern
        self.reverse_pattern: str = reverse_pattern
        self.seq_len: Optional[int] = seq_len
        self.proj_layer_q: torch.nn.Module = proj_layer_q if proj_layer_q else IdentityLayer()
        self.proj_layer_kv: torch.nn.Module = proj_layer_kv if proj_layer_kv else IdentityLayer()
        self.out_layer: torch.nn.Module = out_layer if out_layer else IdentityLayer()



        if spatial_dim_count == 2:
            # Replace the variable dimension 'g' with height and width dimensions for 2D spatial arrangements
            self.pattern = self.pattern.replace('g', 'h w')
            self.reverse_pattern = self.reverse_pattern.replace('g', 'h w')


    def forward(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args: Any, **kwargs: Any):
        """
        Forward pass for the RearrangeBlock.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)`` or ``(b, v, t, g, f)``.
        :param emb: Optional embedding dictionary.
        :param mask: Optional mask tensor of shape ``(b, v, t, n, d, m)`` or broadcastable to x.
        :param cond: Optional conditioning tensor of shape ``(b, v, t, n, d, c)``.
        :param args: Additional positional arguments forwarded to fn.
        :param kwargs: Additional keyword arguments forwarded to fn.
        :return: Tensor after rearrangement and reverse rearrangement, shape matches x except for feature dim.
        """
        # Determine the input dimensionality and extract batch, time, and variable dimensions

        dim = x.dim()
        if dim == 6:
            b, v, t, h, w, c = x.shape
        else:
            b, v, t, g, c = x.shape

        # Project input to query/key-value space if projection layers are provided.
        q = self.proj_layer_q(x, emb=emb).view(*x.shape[:dim-1],-1)
        kv = self.proj_layer_kv(x, emb=emb).view(*x.shape[:dim-1],-1)
        x = torch.concat((q,kv), dim=-1)
        
        x = x.reshape(*x.shape[:dim-1],-1)

        # Rearrange input and optional tensors according to the specified pattern
        x, mask, cond = [
            rearrange(tensor, self.pattern) if torch.is_tensor(tensor) else tensor
            for tensor in (x, mask, cond)
        ]

        s = x.shape[1]
        if self.seq_len and self.seq_len < s:
            n = s // self.seq_len
            x, mask, cond = [
                rearrange(tensor, 'b (n s) ... -> (b n) s ...', n=n, s=self.seq_len) if torch.is_tensor(tensor) else tensor
                for tensor in (x, mask, cond)
            ]

        # Apply the specified function (fn) to the rearranged tensor
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, *args)
        else:
            x = self.fn(x, *args)

        if self.seq_len and self.seq_len < s:
            x, mask, cond = [
                rearrange(tensor, '(b n) s  ... -> b (n s)  ...', n=n, s=self.seq_len) if torch.is_tensor(tensor) else tensor
                for tensor in (x, mask, cond)
            ]

        # Reverse rearrange to restore the original shape
        if dim == 6:
            x = rearrange(x, self.reverse_pattern, b=b, t=t, h=h, w=w, v=v)
        else:
            x = rearrange(x, self.reverse_pattern, b=b, t=t, g=g, v=v)

        x = self.out_layer(x, emb=emb)
        return x


class RearrangeTimeCentric(RearrangeBlock):
    """
    Rearranges input to make the time dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn: Any, spatial_dim_count=1, seq_len: Optional[int] = None, proj_layer_q: Optional[torch.nn.Module] = None, proj_layer_kv: Optional[torch.nn.Module] = None, out_layer: Optional[torch.nn.Module] = None, **kwargs: Any):
        """
        Initialize a time-centric rearrangement block.

        :param fn: Function or module to apply after rearrangement.
        :param spatial_dim_count: Number of spatial dimensions.
        :param seq_len: Optional sequence length for chunking.
        :param proj_layer_q: Optional projection layer for queries.
        :param proj_layer_kv: Optional projection layer for keys/values.
        :param out_layer: Optional output projection layer.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(
            fn,
            pattern='b v t g c -> (b v g) t c',
            reverse_pattern='(b v g) t c -> b v t g c',
            spatial_dim_count=spatial_dim_count,
            seq_len=seq_len,
            proj_layer_q=proj_layer_q,
            proj_layer_kv=proj_layer_kv,
            out_layer=out_layer
        )


class RearrangeSpaceCentric(RearrangeBlock):
    """
    Rearranges input to make spatial dimensions centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn: Any, spatial_dim_count=1, seq_len: Optional[int] = None, proj_layer_q: Optional[torch.nn.Module] = None, proj_layer_kv: Optional[torch.nn.Module] = None, out_layer: Optional[torch.nn.Module] = None, **kwargs: Any):
        """
        Initialize a space-centric rearrangement block.

        :param fn: Function or module to apply after rearrangement.
        :param spatial_dim_count: Number of spatial dimensions.
        :param seq_len: Optional sequence length for chunking.
        :param proj_layer_q: Optional projection layer for queries.
        :param proj_layer_kv: Optional projection layer for keys/values.
        :param out_layer: Optional output projection layer.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(
            fn,
            pattern='b v t g c  -> (b v t) (g) c',
            reverse_pattern='(b v t) (g) c -> b v t g c',
            spatial_dim_count=spatial_dim_count,
            seq_len=seq_len,
            proj_layer_q=proj_layer_q,
            proj_layer_kv=proj_layer_kv,
            out_layer=out_layer
        )


class RearrangeVarCentric(RearrangeBlock):
    """
    Rearranges input to make the variable dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn: Any, spatial_dim_count=1, seq_len: Optional[int] = None, proj_layer_q: Optional[torch.nn.Module] = None, proj_layer_kv: Optional[torch.nn.Module] = None, out_layer: Optional[torch.nn.Module] = None, **kwargs: Any):
        """
        Initialize a variable-centric rearrangement block.

        :param fn: Function or module to apply after rearrangement.
        :param spatial_dim_count: Number of spatial dimensions.
        :param seq_len: Optional sequence length for chunking.
        :param proj_layer_q: Optional projection layer for queries.
        :param proj_layer_kv: Optional projection layer for keys/values.
        :param out_layer: Optional output projection layer.
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        super().__init__(
            fn,
            pattern='b v t g c -> (b t g) v c',
            reverse_pattern='(b t g) v c -> b v t g c',
            spatial_dim_count=spatial_dim_count,
            seq_len=seq_len,
            proj_layer_q=proj_layer_q,
            proj_layer_kv=proj_layer_kv,
            out_layer=out_layer
        )


class RearrangeNHCentric(RearrangeBlock):
    """
    Rearranges input to make the variable dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn: Any, spatial_dim_count=1, seq_len: Optional[int] = None, proj_layer_q: Optional[torch.nn.Module] = None, proj_layer_kv: Optional[torch.nn.Module] = None, out_layer: Optional[torch.nn.Module] = None, grid_layer: Optional[GridLayer] = None):
        """
        Initialize a neighborhood-centric rearrangement block.

        :param fn: Function or module to apply after rearrangement.
        :param spatial_dim_count: Number of spatial dimensions.
        :param seq_len: Optional sequence length for chunking.
        :param proj_layer_q: Optional projection layer for queries.
        :param proj_layer_kv: Optional projection layer for keys/values.
        :param out_layer: Optional output projection layer.
        :param grid_layer: Grid layer used to gather neighborhoods.
        :return: None.
        """
        super().__init__(
            fn,
            pattern='b v t g c  -> (b v t) g c',
            reverse_pattern='(b v t) g c -> b v t g c',
            spatial_dim_count=1,
            seq_len=None,
            proj_layer_q=proj_layer_q,
            proj_layer_kv=proj_layer_kv,
            out_layer=out_layer
        )
        self.grid_layer: Optional[GridLayer] = grid_layer

      #  self.pattern = 'b t s v c -> (b t s v) 1 c'
        self.pattern: str = 'b v t s n c -> (b v t s) (n) c'
        self.nh_mask_pattern: str = 'b v t s n 1 -> (b v t s) 1 1 n'
        self.reverse_pattern: str = '(b v t s) n c -> b v t (s n) c'

    
    def forward(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, mask: Optional[torch.Tensor] = None, sample_configs: Optional[Dict[str, Any]] = None):
        """
        Apply neighborhood-centric rearrangement and attention.

        :param x: Input tensor of shape ``(b, v, t, s, n, f)``.
        :param emb: Optional embedding dictionary.
        :param mask: Optional mask tensor of shape ``(b, v, t, s, n, m)``.
        :param sample_configs: Sampling configuration for neighborhoods.
        :return: Output tensor of shape ``(b, v, t, s * n, f_out)``.
        """
        

        # Project into query and key/value spaces and split into q and kv parts.
        q = self.proj_layer_q(x, emb=emb)
        kv = self.proj_layer_kv(x, emb=emb).view(*x.shape[:4],-1)
        x = torch.concat((q,kv), dim=-1)
        
        x = x.reshape(*x.shape[:4],-1)

        x, x_nh = x.split([x.shape[-1]//3, 2*x.shape[-1]//3],dim=-1)

        # Gather neighborhood values using the grid layer.
        x_nh, mask_nh = self.grid_layer.get_nh(x_nh, **sample_configs, with_nh=True, mask=mask)

        x = x.view(*x_nh.shape[:4],-1, x.shape[-1])

        b, v, t, s, n, c = x.shape

        x = rearrange(x, self.pattern)
        x_nh = rearrange(x_nh, self.pattern)
        
        if mask_nh is not None:
            mask_nh = rearrange(mask_nh, self.nh_mask_pattern)

        x = self.fn(x, x_nh, mask=mask_nh)

        x = rearrange(x, self.reverse_pattern, b=b, v=v, t=t, s=s, n=n, c=c)

        x = self.out_layer(x, emb=emb)

        return x

class RearrangeVarNHCentric(RearrangeNHCentric):
    """
    Rearranges input to make the variable dimension centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn: Any, spatial_dim_count=1, seq_len: Optional[int] = None, proj_layer_q: Optional[torch.nn.Module] = None, proj_layer_kv: Optional[torch.nn.Module] = None, out_layer: Optional[torch.nn.Module] = None, grid_layer: Optional[GridLayer] = None):
        """
        Initialize a variable+neighborhood-centric rearrangement block.

        :param fn: Function or module to apply after rearrangement.
        :param spatial_dim_count: Number of spatial dimensions.
        :param seq_len: Optional sequence length for chunking.
        :param proj_layer_q: Optional projection layer for queries.
        :param proj_layer_kv: Optional projection layer for keys/values.
        :param out_layer: Optional output projection layer.
        :param grid_layer: Grid layer used to gather neighborhoods.
        :return: None.
        """
        super().__init__(
            fn,
            proj_layer_q=proj_layer_q,
            proj_layer_kv=proj_layer_kv,
            out_layer=out_layer,
            grid_layer=grid_layer
        )
      

      #  self.pattern = 'b t s v c -> (b t s v) 1 c'
        self.pattern: str = 'b v t s n c -> (b t s) (v n) c'
        self.nh_mask_pattern: str = 'b v t s n 1-> (b t s) 1 1 (v n)'
        self.reverse_pattern: str = '(b t s) (v n) c -> b v t (s n) c'

    
    
class RearrangeConvCentric(RearrangeBlock):
    """
    Rearranges input to make the convolutional channels centric before applying a function, then reverses it.

    :param fn: The function to apply to the rearranged input.
    :param spatial_dim_count: Determines the number of spatial dimensions, adjusting the rearrangement accordingly.
    """

    def __init__(self, fn: Any, spatial_dim_count=1, seq_len: Optional[int] = None, dims: int = 2, **kwargs: Any):
        """
        Initialize a convolution-centric rearrangement block.

        :param fn: Function or module to apply after rearrangement.
        :param spatial_dim_count: Number of spatial dimensions.
        :param seq_len: Optional sequence length for chunking.
        :param dims: Dimensionality of the convolution (1, 2, or 3).
        :param kwargs: Additional keyword arguments (unused).
        :return: None.
        """
        assert dims == 1 or dims == 2 or dims == 3
        if dims - spatial_dim_count == 0:
            pattern = 'b v t g c -> (b v t) c g'
            reverse_pattern = '(b v t) c g -> b v t g c'
        elif dims - spatial_dim_count == 1:
            pattern = 'b v t g c -> (b v) c t g'
            reverse_pattern = '(b v) c t g -> b v t g c'
        else:
            pattern = 'b v t g c -> b c v t g'
            reverse_pattern = 'b c v t g -> b v t g c'

        super().__init__(
            fn,
            pattern=pattern,
            reverse_pattern=reverse_pattern,
            spatial_dim_count=spatial_dim_count,
            seq_len=seq_len
        )

    def forward(self, x: torch.Tensor, emb: Optional[Dict] = None, mask: Optional[torch.Tensor] = None,
                cond: Optional[torch.Tensor] = None, *args, **kwargs):
        """
        Forward pass for the RearrangeConvCentric.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dict.
        :param mask: Optional mask tensor of shape ``(b, v, t, n, d, m)``.
        :param cond: Optional conditioning tensor of shape ``(b, v, t, n, d, c)``.
        :param args: Additional positional arguments forwarded to fn.
        :param kwargs: Additional keyword arguments forwarded to fn.
        :return: Tensor after rearrangement and reverse rearrangement, shape matches x except for feature dim.
        """
        b, v, t, c = x.shape[0], x.shape[1], x.shape[2], x.shape[-1]

        # Rearrange input and optional tensors according to the specified pattern
        x, mask, cond = [
            rearrange(tensor, self.pattern) if torch.is_tensor(tensor) else tensor
            for tensor in (x, mask, cond)
        ]

        # Apply the function (fn) to the rearranged tensor
        if isinstance(self.fn, EmbedBlock):
            x = self.fn(x, emb, mask, cond, *args, **kwargs)
        else:
            x = self.fn(x, *args, **kwargs)

        # Reverse rearrange to restore the original shape
        return rearrange(x, self.reverse_pattern, b=b, v=v)
