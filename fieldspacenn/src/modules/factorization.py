from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
import math

import torch
import torch.nn as nn


def _get_layer_variable_indices(emb: Optional[Dict[str, Any]]) -> Any:
    if emb is None:
        raise KeyError("Embedding dictionary is required for variable-wise parameter selection.")
    if "variables_sampled" in emb:
        return emb["variables_sampled"]
    if "VariableEmbedder" in emb:
        return emb["VariableEmbedder"]
    raise KeyError("Expected `variables_sampled` (or fallback `VariableEmbedder`) in embedding dictionary.")


def get_ranks(shape: Sequence[int], rank: Union[int, float], rank_decay: float = 0):
    """
    Compute per-dimension ranks with optional decay.

    :param shape: Input tensor shape as a sequence of ints.
    :param rank: Base rank value (absolute or relative).
    :param rank_decay: Linear decay applied across dimensions.
    :return: List of computed ranks per dimension.
    """
    rank_ = []
    for k in range(len(shape)):
        r = rank * (1 - rank_decay*k/(max([1,len(shape)-1]))) 
        if k < len(shape)-1:
            rank_.append(r)
        else:
            if len(rank_)>0:
                rank_.append(float(torch.tensor(rank_).mean()))
            else:
                rank_.append(float(rank))

    if rank > 1:
        ranks = [min([dim, int(rank_[k])]) for k, dim in enumerate(shape)]
    else:
        ranks = [max([1,int(dim * rank_[k])]) for k, dim in enumerate(shape)]
    
    return ranks


def get_fac_matrix(dim: int, rank: Union[int, float]):
    """
    Initialize a factor matrix with orthogonal columns.

    :param dim: Input dimension.
    :param rank: Factorization rank (absolute or relative).
    :return: Learnable parameter matrix of shape ``(dim, rank)``.
    """
    if isinstance(rank, float):
        rank = int(rank * dim)

    rank = int(max([rank, 1]))
    m = torch.empty(dim, rank)
    nn.init.orthogonal_(m)

    return nn.Parameter(m, requires_grad=True)


class TuckerFacLayer(nn.Module):
    """
    Tucker factorization layer supporting variable-wise cores.

    :param in_features: Input feature shape(s).
    :param out_features: Output feature shape(s).
    :param ranks: Per-dimension ranks for factorization.
    :param rank_variables: Optional rank for variable factorization.
    :param n_variables: Number of variables for variable-wise cores.
    :param bias: Whether to include a bias term.
    """
    def __init__(self,
                 in_features: Union[List[int], int], 
                 out_features: Union[List[int], int], 
                 ranks: Optional[List[Union[int, float]]] = None,
                 rank_variables: Optional[int] = None,
                 n_variables: int = 1,
                 bias: bool = False,
                 **kwargs: Any):

        super().__init__()

        # contract_dims: List[bool]
        # rank_feat: List[int]
        # ranks: None does not factorize,   

        if isinstance(in_features, int):
            in_features = [in_features]

        if isinstance(out_features, int):
            out_features = [out_features]

        assert len(in_features)==len(out_features), f"unmachting len of in_features {in_features} and out_features {out_features}"
        #contract_features = [rank_feat > 0 for k in in_features]
        self.factor_letters: Iterator[str] = iter("aefghijklmopqruwxyz")
        self.core_letters: Iterator[str] = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ") 

        
        self.n_variables: int = int(n_variables)
        factorize_vars = rank_variables is not None

        self.subscripts: Dict[str, Any] = {
            'factors': [],
            'core': '',
            'x_in': 'bvtnd',
            'x_out': 'bvtnd',
            }
        
        self.core_dims: List[int] = []
        self.factor_vars: Optional[nn.Parameter] = None
        self.get_var_fac_fcn: Callable[..., List[torch.Tensor]]
        self.get_core_fcn: Callable[..., torch.Tensor]
        self.get_bias_fcn: Callable[..., Optional[torch.Tensor]]

        if factorize_vars:
            self.factor_vars = get_fac_matrix(self.n_variables, rank_variables)
            self.core_dims.append(rank_variables)
            self.get_var_fac_fcn = self.get_variable_factors
            self.get_core_fcn = self.get_core

            sub_c = next(self.core_letters)
            self.subscripts['core'] = sub_c
            self.subscripts['factors'].append('bv' + sub_c)
        else:
            self.core_dims.append(self.n_variables)
            self.get_var_fac_fcn = self.get_empty1
            self.get_core_fcn = self.get_core_from_var_idx
            self.subscripts['core'] = 'bv'

        self.factors: nn.ParameterList = nn.ParameterList()
        in_dims = []
        for rank, f_in in zip(ranks, in_features):

            x_sub, core_dim = self.add(rank, f_in)
            self.subscripts['x_in'] += x_sub
            in_dims.append(core_dim)

        out_dims = []
        for rank, f_out in zip(ranks, out_features):
            
            x_sub, core_dim = self.add(rank, f_out)
            self.subscripts['x_out'] += x_sub
            out_dims.append(core_dim)

        self.in_features: List[int] = in_features
        self.out_features: List[int] = out_features

        fan_in = math.prod(in_dims)
        bound = 1.0 / math.sqrt(fan_in)
        core = torch.empty(self.core_dims)
        nn.init.uniform_(core, -bound, bound)
        self.core: nn.Parameter = nn.Parameter(core, requires_grad=True)

        self.bias = None
        if bias:
            bias_ = torch.empty([n_variables, *out_features])
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(bias_, -bound, bound)

            self.bias = nn.Parameter(bias_, requires_grad=True)
            self.get_bias_fcn = self.get_bias_from_var_idx
        else:
            self.get_bias_fcn = self.get_none
    
    def add(self, rank: Optional[Union[int, float]], features: int):
        """
        Register a factor matrix or passthrough dimension for a given feature size.

        :param rank: Rank to use for this feature (None to skip factorization).
        :param features: Feature dimension size.
        :return: Tuple of (subscript, core_dim) for einsum construction.
        """

        core_sub = next(self.core_letters)

        if rank is not None  and rank < features and rank > 0:
            fac_sub = next(self.factor_letters)
            self.factors.append(get_fac_matrix(rank, features))
            self.subscripts['factors'] += [core_sub + fac_sub]
            core_dim = rank
            x_sub = fac_sub

        else:
            core_sub = next(self.core_letters)
            core_dim = features
            x_sub = core_sub

        self.subscripts['core'] += core_sub
        self.core_dims.append(core_dim)

        return x_sub, core_dim

    def get_core_from_var_idx(self, emb: Optional[Dict[str, Any]] = None):
        """
        Select a variable-specific core tensor.

        :param emb: Embedding dict containing variable indices.
        :return: Core tensor for the selected variables.
        """
        if self.n_variables <= 1:
            return self.core.unsqueeze(0)

        return self.core[_get_layer_variable_indices(emb)]

    def get_core(self, **kwargs: Any):
        """
        Return the full core tensor.
        """
        return self.core
    
    def get_empty1(self, **kwargs: Any):
        """
        Return an empty list for cases without variable factors.
        """
        return []
    
    def get_variable_factors(self, emb: Dict[str, Any]):
        """
        Return variable-specific factor matrices.

        :param emb: Embedding dict containing variable indices.
        :return: List with variable factor matrix.
        """
        if self.n_variables <= 1:
            return [self.factor_vars.unsqueeze(0)]

        return [self.factor_vars[_get_layer_variable_indices(emb)]]
        
    def get_bias(self, **kwargs: Any):
        """
        Return the shared bias tensor.
        """
        return self.bias

    def get_bias_from_var_idx(self, emb: Dict[str, Any]):
        """
        Return variable-specific bias selected by variable indices.
        """
        if self.n_variables <= 1:
            return self.bias[0]

        return self.bias[_get_layer_variable_indices(emb)]

    def get_none(self, **kwargs: Any):
        """
        Return None for optional tensors (e.g., disabled bias).
        """
        return None

    def add_bias(self, x: torch.Tensor, bias: torch.Tensor):
        """
        Add shared or variable-specific bias with correct broadcasting over ``(t, n, d)``.
        """
        if bias.ndim == len(self.out_features):
            bias = bias.view(*([1] * 5), *self.out_features)
        elif bias.ndim == len(self.out_features) + 2:
            bias = bias.view(*bias.shape[:2], *([1] * 3), *self.out_features)
        return x + bias
    

    def forward(self, x: torch.Tensor, emb: Optional[Dict[str, Any]] = None, sample_configs: Dict[str, Any] = {}):
        """
        Apply Tucker factorized transformation.

        :param x: Input tensor of shape ``(b, v, t, n, d, f_in...)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Optional sampling configuration (unused).
        :return: Output tensor of shape ``(b, v, t, n, d, f_out...)``.
        """
        
        f_v = self.get_var_fac_fcn(emb=emb)

        core = self.get_core_fcn(emb=emb)
        
        x_shape = list(x.shape[:5])
        x_dims_in = x_shape + self.in_features 
        x_dims_out = x_shape + self.out_features 

        x = x.reshape(x_dims_in)
        
        lhs = [self.subscripts['x_in'], self.subscripts['core'], *self.subscripts['factors']]
        
        einsum_eq = (
            f"{','.join(lhs)}"
            f"->{self.subscripts['x_out']}"
        )

        factors = f_v + list(self.factors)

        # Contract input, core, and factors using the constructed einsum equation.
        x = torch.einsum(einsum_eq, x, core, *factors)
        
        x = x.reshape(x_dims_out)

        bias = self.get_bias_fcn(emb=emb)
        if bias is not None:
            x = self.add_bias(x, bias)

        return x
