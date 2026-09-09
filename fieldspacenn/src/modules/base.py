import copy
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..utils.helpers import check_get

import torch
import torch.nn as nn

from .factorization import (
    TuckerFacLayer,
    broadcast_indexed_tensor,
    normalize_indexed_dims,
)


def _get_layer_variable_indices(emb: Optional[Dict[str, Any]]) -> Any:
    if emb is None:
        raise KeyError("Embedding dictionary is required for variable-wise parameter selection.")
    if "variables_sampled" in emb:
        return emb["variables_sampled"]
    if "VariableEmbedder" in emb:
        return emb["VariableEmbedder"]
    raise KeyError("Expected `variables_sampled` (or fallback `VariableEmbedder`) in embedding dictionary.")


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        n_variables: int = 1,
        indexed_dims: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        """
        Custom LayerNorm supporting optional variable-specific parameters.

        :param normalized_shape: Shape over which normalization is applied.
        :param n_variables: Number of variables for variable-wise affine params.
        :param eps: Small value to avoid division by zero.
        :param elementwise_affine: Whether to learn gamma and beta.
        """
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
            
        self.normalized_shape: tuple = tuple(normalized_shape)
        self.eps: float = eps
        self.elementwise_affine: bool = elementwise_affine
        self.n_variables: int = n_variables
        self.indexed_dims = normalize_indexed_dims(
            indexed_dims=indexed_dims,
            n_variables=n_variables,
        )

        self.weight: Optional[nn.Parameter] = None
        self.bias: Optional[nn.Parameter] = None
        if elementwise_affine:
            indexed_shape = [spec["n_features"] for spec in self.indexed_dims.values()]
            if not indexed_shape:
                self.weight: nn.Parameter = nn.Parameter(torch.ones(self.normalized_shape))
                self.bias: nn.Parameter = nn.Parameter(torch.zeros(self.normalized_shape))
            else:
                self.weight = nn.Parameter(torch.ones(*indexed_shape, *self.normalized_shape))
                self.bias = nn.Parameter(torch.zeros(*indexed_shape, *self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def apply_weight_bias(self, x_hat: torch.Tensor, emb: Optional[Dict[str, Any]]):
        """
        Apply learnable affine parameters, optionally per variable.

        :param x_hat: Normalized tensor of shape ``(b, v, t, n, d, f)``.
        :param emb: Optional embedding dict containing variable indices.
        :return: Affine-transformed tensor of the same shape.
        """
        if not self.indexed_dims:
            return self.weight * x_hat + self.bias
        else:
            weight = broadcast_indexed_tensor(self.weight, self.indexed_dims, x_hat, emb=emb)
            bias = broadcast_indexed_tensor(self.bias, self.indexed_dims, x_hat, emb=emb)
            return weight * x_hat + bias

    def forward(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, x_stats: Optional[torch.Tensor] = None):
        """
        Normalize across the last ``normalized_shape`` dimensions.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)`` or compatible.
        :param emb: Optional embedding dict for variable-wise affine params.
        :param x_stats: Optional tensor to compute mean/var from.
        :return: Normalized tensor of the same shape as x.
        """
        if x_stats is None:
            x_stats = x

        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x_stats.mean(dim=dims, keepdim=True)
        var = x_stats.var(dim=dims, unbiased=False, keepdim=True)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            return self.apply_weight_bias(x_hat, emb=emb)
        else:
            return x_hat


class IdentityLayer(nn.Module):
    """
    Identity layer with optional variable-aware tensor selection.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, **kwargs: Any):
        """
        Forward pass returning the input tensor unchanged.

        :param x: Input tensor.
        :param kwargs: Additional keyword arguments (unused).
        :return: Unmodified input tensor.
        """
        return x
    
    def get_tensor(self, tensor: torch.Tensor, emb: Dict[str, Any]):
        """
        Select a variable-specific tensor if variable indexing is enabled.

        :param tensor: Input tensor of shape ``(v, ...)`` when using variable selection.
        :param emb: Embedding dict containing variable indices.
        :return: Selected tensor.
        """
        if self.n_variables==1:
            return tensor
        else:
            return tensor[_get_layer_variable_indices(emb)]

    
class MLP_fac(nn.Module):
    """
    MLP wrapper that supports factorized layers and optional gamma scaling.

    :param in_features: Input feature size(s).
    :param out_features: Output feature size(s).
    :param mult: Hidden dimension multiplier.
    :param hidden_dim: Optional explicit hidden dimension.
    :param dropout: Dropout probability.
    :param layer_confs: Layer configuration dictionary.
    :param gamma: Whether to apply learnable output scaling.
    """
    def __init__(self,
                 in_features: Union[int, List[int]], 
                 out_features: Union[int, List[int]],
                 mult: int = 1,
                 hidden_dim: Optional[Union[int, List[int]]] = None,
                 dropout: float = 0,
                 layer_confs: Optional[Dict[str, Any]] = None,
                 ranks: Optional[List[Optional[int]]] = None,
                 n_variables: int = 1,
                 indexed_dims: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
                 indexed_dims_layer1: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
                 indexed_dims_layer2: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
                 fac_mode: str = "Tucker",
                 gamma: bool = False
                ) -> None: 
      
        super().__init__() 
        
        if hidden_dim is None:
            if isinstance(in_features, list):
                out_features_1 = [int(in_feat*mult) for in_feat in in_features]
            else:
                out_features_1 = int(in_features*mult)
        else:
            if isinstance(hidden_dim, list) or not isinstance(in_features, list):
                out_features_1 = hidden_dim
            else:
                out_features_1 = [int(in_feat) for in_feat in in_features]
                out_features_1[1] = hidden_dim       

        layer_confs = {} if layer_confs is None else copy.deepcopy(layer_confs)
        if ranks is None:
            ranks = layer_confs.get("ranks", [None])
        n_variables = layer_confs.get("n_variables", n_variables)
        indexed_dims = layer_confs.get("indexed_dims", indexed_dims)
        indexed_dims_layer1 = layer_confs.get(
            "indexed_dims_layer1",
            indexed_dims if indexed_dims_layer1 is None else indexed_dims_layer1,
        )
        indexed_dims_layer2 = layer_confs.get(
            "indexed_dims_layer2",
            indexed_dims if indexed_dims_layer2 is None else indexed_dims_layer2,
        )
        fac_mode = layer_confs.get("fac_mode", fac_mode)
        self.layer1 = get_layer(
            in_features,
            out_features_1,
            ranks=ranks,
            n_variables=n_variables,
            indexed_dims=indexed_dims_layer1,
            fac_mode=fac_mode,
            bias=True,
        )
        self.layer2 = get_layer(
            out_features_1,
            out_features,
            ranks=ranks,
            n_variables=n_variables,
            indexed_dims=indexed_dims_layer2,
            fac_mode=fac_mode,
            bias=True,
        )
        self.dropout: nn.Module = nn.Dropout(p=dropout) if dropout>0 else nn.Identity()
        self.activation: nn.Module = nn.SiLU()

        if gamma:
            self.gamma: torch.nn.Parameter = torch.nn.Parameter(torch.ones(out_features) * 1E-6)
            self.rtn_fcn: Any = self.rtn_w_gamma
        else:
            self.rtn_fcn: Any = self.rtn

    def rtn_w_gamma(self, x: torch.Tensor):
        """
        Apply learnable gamma scaling to the output.

        :param x: Input tensor.
        :return: Scaled tensor.
        """
        return x * self.gamma
    
    def rtn(self, x: torch.Tensor):
        """
        Return the tensor unchanged.

        :param x: Input tensor.
        :return: Input tensor.
        """
        return x

    def forward(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, **kwargs: Any):
        """
        Apply MLP transformation with optional embeddings.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)`` or flattened.
        :param emb: Optional embedding dictionary.
        :param kwargs: Additional keyword arguments forwarded to layers.
        :return: Output tensor with updated feature dimension.
        """
        
        x = self.layer1(x, emb=emb, **kwargs)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x, emb=emb, **kwargs)

        return self.rtn_fcn(x)


def get_layer(
        in_features: Union[int, List[int]],
        out_features: Union[int, List[int]],
        layer_confs: Optional[Dict[str, Any]] = None,
        ranks: Optional[List[Optional[int]]] = None,
        n_variables: int = 1,
        indexed_dims: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
        fac_mode: str = "Tucker",
        rank_variables: Optional[int] = None,
        **kwargs: Any
        ):  
    """
    Build a linear or factorized layer based on configuration.

    :param in_features: Input feature size(s).
    :param out_features: Output feature size(s).
    :param layer_confs: Layer configuration dictionary.
    :param kwargs: Additional overrides (ranks, n_variables, bias, fac_mode).
    :return: Instantiated layer module.
    """

    layer_confs = {} if layer_confs is None else copy.deepcopy(layer_confs)
    if ranks is None:
        ranks = layer_confs.get("ranks", [None])
    else:
        ranks = copy.deepcopy(ranks)
    n_variables = layer_confs.get("n_variables", n_variables)
    indexed_dims = layer_confs.get("indexed_dims", indexed_dims)
    fac_mode = layer_confs.get("fac_mode", fac_mode)
    if rank_variables is None:
        rank_variables = layer_confs.get("rank_variables", None)
    layer_kwargs = copy.deepcopy(kwargs)
    bias = check_get([layer_kwargs, {'bias': False}], 'bias')
    layer_kwargs.pop("bias", None)

    ranks_not_none = [rank is not None for rank in ranks]

    indexed_dims_norm = normalize_indexed_dims(
        indexed_dims=indexed_dims,
        n_variables=n_variables,
        rank_variables=rank_variables,
    )

    if not any(ranks_not_none) and not indexed_dims_norm:
        # Use a standard linear layer when no factorization is requested.
        layer = LinearLayer(
                in_features,
                out_features,
                bias=bias
                )

    elif fac_mode=='Tucker':
        # Use Tucker factorization for parameter-efficient linear transforms.
        layer = TuckerFacLayer(
            in_features,
            out_features,
            bias = bias,
            ranks=ranks,
            n_variables=n_variables,
            indexed_dims=indexed_dims_norm,
            rank_variables=rank_variables,
            **layer_kwargs,
        )
        
    else:
        raise NotImplementedError

    return layer


class LinearLayer(nn.Module):
    """
    Linear layer supporting multi-dimensional feature shapes.

    :param in_features: Input feature shape(s).
    :param out_features: Output feature shape(s).
    :param bias: Whether to include a bias term.
    :param skip_dims: Optional boolean mask of dimensions to skip.
    """
    def __init__(self, 
                 in_features: Union[int, List[int]], 
                 out_features: Union[int, List[int]],
                 bias: bool = False,
                 skip_dims: Optional[List[bool]] = None,
                 **kwargs: Any):

        super().__init__()

        if isinstance(in_features,int):
            in_features = [in_features]

        if isinstance(out_features,int):
            out_features = [out_features]

       # self.out_features = int(torch.tensor(out_features).prod())
        self.in_features: List[int] = in_features
        self.out_features: List[int] = out_features
        
        self.in_shapes: List[int] = in_features
        self.out_shapes: List[int] = out_features

        if skip_dims is not None:
            self.in_features  = []
            self.out_features = []
            for skip_dim, in_feat, out_feat in zip(skip_dims, in_features, out_features):
                if not skip_dim:
                    self.in_features.append(in_feat)
                    self.out_features.append(out_feat)

        self.in_features_tot: int = math.prod(self.in_features)

        self.layer: nn.Linear = nn.Linear(self.in_features_tot , math.prod(self.out_features), bias=bias)

    def forward(self, x: torch.Tensor, **kwargs: Any):
        """
        Apply the linear projection across the last feature dimensions.

        :param x: Input tensor of shape ``(b, v, t, n, d, f)``.
        :param kwargs: Additional keyword arguments (unused).
        :return: Output tensor of shape ``(b, v, t, n, d, f_out)``.
        """
        x_dims = x.shape[:5]

        # Flatten the feature dimensions, apply linear, then reshape.
        x = x.reshape(*x_dims, self.in_features_tot)
        x = self.layer(x)
        x = x.view(*x_dims, *self.out_features)

        return x
