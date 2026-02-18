from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union
import math

import torch
import torch.nn as nn

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
        
        # variables -----
        scale = 0
        if factorize_vars and n_variables>1:
            self.factor_vars = get_fac_matrix(n_variables, rank_variables)
            self.core_dims.append(rank_variables)
            self.get_var_fac_fcn = self.get_variable_factors
            self.get_core_fcn = self.get_core

            sub_c = next(self.core_letters)
            self.subscripts['core'] = sub_c
            self.subscripts['factors'].append('bv' + sub_c)

        elif n_variables>1:
            self.core_dims.append(n_variables)
            self.get_core_fcn = self.get_core_from_var_idx
            self.get_var_fac_fcn = self.get_empty1
            self.subscripts['core'] = 'bv'

        else:
            self.get_core_fcn = self.get_core
            self.get_var_fac_fcn = self.get_empty1

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

        core = torch.empty(self.core_dims)

        if n_variables == 1:
            core = core.reshape(math.prod(in_dims),math.prod(out_dims))
            nn.init.kaiming_normal_(core)
            core = core.reshape(self.core_dims)
        else:
            for k, c_ in enumerate(core):
                c_ = c_.reshape(math.prod(in_dims),math.prod(out_dims))
                nn.init.kaiming_normal_(c_)
                core[k] = c_.reshape(self.core_dims[1:])

        self.core: nn.Parameter = nn.Parameter(core, requires_grad=True)

        self.bias: Optional[nn.Parameter] = None
        if bias:
            if len(out_features)==1:
                bias = torch.randn(out_features)
            else:
                bias = torch.empty(out_features)
                nn.init.kaiming_uniform_(bias)
                
            self.bias: nn.Parameter = nn.Parameter(bias)
            self.return_fcn: Callable[[torch.Tensor], torch.Tensor] = self.return_w_bias
        else:
            self.bias = None
            self.return_fcn = self.return_wo_bias
    
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
        return self.core[emb['VariableEmbedder']]
    
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
        return [self.factor_vars[emb['VariableEmbedder']]]
        
    def return_w_bias(self, x: torch.Tensor):
        """
        Add bias to the output tensor.
        """
        return x + self.bias
    
    def return_wo_bias(self, x: torch.Tensor):
        """
        Return the output tensor unchanged.
        """
        return x
    

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

        return self.return_fcn(x)
