from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


AXIS_ORDER: tuple[str, ...] = ("v", "t", "n", "d")
AXIS_TO_DIM: Dict[str, int] = {"v": 1, "t": 2, "n": 3, "d": 4}
DIM_TO_AXIS: Dict[int, str] = {dim: axis for axis, dim in AXIS_TO_DIM.items()}
LEGACY_ALIAS_TO_AXIS: Dict[str, str] = {
    "n_variables": "v",
    "rank_variables": "v",
    "n_times": "t",
    "rank_time": "t",
    "n_space": "n",
    "rank_space": "n",
    "n_depths": "d",
    "rank_depth": "d",
}


def _get_layer_variable_indices(emb: Optional[Dict[str, Any]]) -> Any:
    if emb is None:
        raise KeyError("Embedding dictionary is required for variable-wise parameter selection.")
    if "variables_sampled" in emb:
        return emb["variables_sampled"]
    if "VariableEmbedder" in emb:
        return emb["VariableEmbedder"]
    raise KeyError("Expected `variables_sampled` (or fallback `VariableEmbedder`) in embedding dictionary.")


def _axis_runtime_size(reference: Union[torch.Tensor, Sequence[int]], dim: int) -> int:
    shape = reference.shape if isinstance(reference, torch.Tensor) else reference
    return int(shape[dim])


def _ensure_int_or_none(value: Optional[Union[int, float]]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def build_indexed_dims(
    n_variables: int = 1,
    rank_variables: Optional[int] = None,
    same_values_variables: bool = False,
    n_times: int = 1,
    rank_time: Optional[int] = None,
    same_values_times: bool = False,
    n_space: int = 1,
    rank_space: Optional[int] = None,
    same_values_space: bool = False,
    n_depths: int = 1,
    rank_depth: Optional[int] = None,
    same_values_depths: bool = False,
) -> "OrderedDict[str, Dict[str, Any]]":
    indexed_dims: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    values = {
        "v": (n_variables, rank_variables, True, same_values_variables),
        "t": (n_times, rank_time, False, same_values_times),
        "n": (n_space, rank_space, False, same_values_space),
        "d": (n_depths, rank_depth, False, same_values_depths),
    }
    for axis in AXIS_ORDER:
        n_features, rank, use_emb_indices, same_values = values[axis]
        n_features = int(n_features)
        if n_features <= 1:
            continue
        indexed_dims[axis] = {
            "dim": AXIS_TO_DIM[axis],
            "n_features": n_features,
            "rank": _ensure_int_or_none(rank),
            "use_emb_indices": use_emb_indices,
            "same_values": bool(same_values),
        }
    return indexed_dims


def normalize_indexed_dims(
    indexed_dims: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
    n_variables: int = 1,
    rank_variables: Optional[int] = None,
) -> "OrderedDict[str, Dict[str, Any]]":
    if indexed_dims is None:
        return build_indexed_dims(n_variables=n_variables, rank_variables=rank_variables)

    normalized: MutableMapping[str, Dict[str, Any]] = {}
    for key, value in indexed_dims.items():
        if isinstance(key, str):
            if key in LEGACY_ALIAS_TO_AXIS:
                axis = LEGACY_ALIAS_TO_AXIS[key]
            else:
                axis = key
        else:
            axis = DIM_TO_AXIS[int(key)]

        if axis not in AXIS_TO_DIM:
            raise ValueError(f"Unsupported indexed axis `{key}`.")

        spec = dict(value)
        spec["dim"] = int(spec.get("dim", AXIS_TO_DIM[axis]))
        spec["n_features"] = int(spec["n_features"])
        spec["rank"] = _ensure_int_or_none(spec.get("rank"))
        spec["use_emb_indices"] = bool(spec.get("use_emb_indices", axis == "v"))
        spec["same_values"] = bool(spec.get("same_values", False))

        if spec["dim"] != AXIS_TO_DIM[axis]:
            raise ValueError(
                f"Indexed axis `{axis}` must map to dim {AXIS_TO_DIM[axis]}, got {spec['dim']}."
            )
        if spec["n_features"] > 1:
            normalized[axis] = spec

    ordered: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for axis in AXIS_ORDER:
        if axis in normalized:
            ordered[axis] = normalized[axis]
    return ordered


def get_runtime_index_tensors(
    reference: torch.Tensor,
    indexed_dims: Optional[Mapping[str, Mapping[str, Any]]],
    emb: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    indexed_dims = normalize_indexed_dims(indexed_dims=indexed_dims)
    runtime_indices: Dict[str, torch.Tensor] = {}
    prefix_shape = list(reference.shape[:5])
    device = reference.device

    for axis, spec in indexed_dims.items():
        axis_dim = AXIS_TO_DIM[axis]
        axis_size = _axis_runtime_size(reference, axis_dim)
        if axis_size > int(spec["n_features"]):
            raise ValueError(
                f"Indexed axis `{axis}` has runtime size {axis_size}, which exceeds configured "
                f"n_features={spec['n_features']}."
            )

        if spec["use_emb_indices"]:
            axis_indices = _get_layer_variable_indices(emb).to(device=device, dtype=torch.long)
            if axis_indices.ndim == 1:
                axis_indices = axis_indices.unsqueeze(0)
            if axis_indices.shape[0] == 1 and prefix_shape[0] > 1:
                axis_indices = axis_indices.expand(prefix_shape[0], -1)
            expected_shape = (prefix_shape[0], prefix_shape[axis_dim])
            if tuple(axis_indices.shape) != expected_shape:
                raise ValueError(
                    f"Expected variable indices with shape {expected_shape}, got {tuple(axis_indices.shape)}."
                )
            if axis_indices.numel() > 0 and (
                int(axis_indices.min().item()) < 0
                or int(axis_indices.max().item()) >= int(spec["n_features"])
            ):
                raise ValueError(
                    f"Variable indices must be in [0, {int(spec['n_features']) - 1}], got "
                    f"min={int(axis_indices.min().item())}, max={int(axis_indices.max().item())}."
                )
        else:
            axis_indices = torch.arange(axis_size, device=device, dtype=torch.long)

        view_shape = [1, 1, 1, 1, 1]
        if spec["use_emb_indices"]:
            view_shape[0] = prefix_shape[0]
        view_shape[axis_dim] = axis_size
        runtime_indices[axis] = axis_indices.view(*view_shape)

    return runtime_indices


def select_indexed_tensor(
    tensor: torch.Tensor,
    indexed_dims: Optional[Mapping[str, Mapping[str, Any]]],
    reference: torch.Tensor,
    emb: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    indexed_dims = normalize_indexed_dims(indexed_dims=indexed_dims)
    if not indexed_dims:
        return tensor

    runtime_indices = get_runtime_index_tensors(reference, indexed_dims=indexed_dims, emb=emb)
    index = tuple(runtime_indices[axis] for axis in indexed_dims.keys())
    return tensor[index]


def broadcast_indexed_tensor(
    tensor: torch.Tensor,
    indexed_dims: Optional[Mapping[str, Mapping[str, Any]]],
    reference: torch.Tensor,
    emb: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    tensor = select_indexed_tensor(tensor, indexed_dims=indexed_dims, reference=reference, emb=emb)
    while tensor.ndim < reference.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor


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
        r = rank * (1 - rank_decay * k / (max([1, len(shape) - 1])))
        if k < len(shape) - 1:
            rank_.append(r)
        else:
            if len(rank_) > 0:
                rank_.append(float(torch.tensor(rank_).mean()))
            else:
                rank_.append(float(rank))

    if rank > 1:
        ranks = [min([dim, int(rank_[k])]) for k, dim in enumerate(shape)]
    else:
        ranks = [max([1, int(dim * rank_[k])]) for k, dim in enumerate(shape)]

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

    rank = int(max(rank, 1))
    m = torch.empty(dim, rank)
    nn.init.orthogonal_(m)

    return nn.Parameter(m, requires_grad=True)


def get_indexed_fac_matrix(
    dim: int,
    rank: Union[int, float],
    same_values: bool = False,
):
    """
    Initialize an indexed factor matrix with row-wise normalization over rank only.

    :param dim: Number of indexed entries.
    :param rank: Factorization rank.
    :param same_values: Whether every indexed entry should share the same row values.
    :return: Learnable parameter matrix of shape ``(dim, rank)``.
    """
    if isinstance(rank, float):
        rank = int(rank * dim)

    rank = int(max(rank, 1))
    if same_values:
        row = F.normalize(torch.ones(1, rank), dim=-1)
        m = row.expand(dim, -1).clone()
    else:
        m = F.normalize(torch.randn(dim, rank), dim=-1)

    return nn.Parameter(m, requires_grad=True)


class TuckerFacLayer(nn.Module):
    """
    Tucker factorization layer supporting indexed tensor dimensions.

    :param in_features: Input feature shape(s).
    :param out_features: Output feature shape(s).
    :param ranks: Per-feature ranks for factorization.
    :param indexed_dims: Optional indexed axis specification for ``(v, t, n, d)``.
    :param n_variables: Legacy alias for indexed variable dims.
    :param rank_variables: Legacy alias for indexed variable rank.
    :param bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: Union[List[int], int],
        out_features: Union[List[int], int],
        ranks: Optional[List[Optional[Union[int, float]]]] = None,
        indexed_dims: Optional[Mapping[Union[str, int], Mapping[str, Any]]] = None,
        rank_variables: Optional[int] = None,
        n_variables: int = 1,
        bias: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        if isinstance(in_features, int):
            in_features = [in_features]
        if isinstance(out_features, int):
            out_features = [out_features]
        if ranks is None:
            ranks = [None] * len(in_features)

        assert len(in_features) == len(out_features), (
            f"unmachting len of in_features {in_features} and out_features {out_features}"
        )

        self.factor_letters: Iterator[str] = iter("aefghijklmopqruwxyz")
        self.core_letters: Iterator[str] = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.prefix_subscripts: str = "bvtnd"
        self.in_features: List[int] = list(in_features)
        self.out_features: List[int] = list(out_features)
        self.n_variables: int = int(n_variables)
        self.indexed_dims = normalize_indexed_dims(
            indexed_dims=indexed_dims,
            n_variables=n_variables,
            rank_variables=rank_variables,
        )
        self.bias_indexed_dims = OrderedDict(
            (
                axis,
                {
                    "dim": spec["dim"],
                    "n_features": spec["n_features"],
                    "rank": None,
                    "use_emb_indices": spec["use_emb_indices"],
                },
            )
            for axis, spec in self.indexed_dims.items()
        )

        self.indexed_factors: nn.ParameterList = nn.ParameterList()
        self.indexed_factor_subscripts: List[str] = []
        self.indexed_rank_subscripts: str = ""
        self.feature_factor_subscripts: List[str] = []
        self.input_subscripts: str = self.prefix_subscripts
        self.output_subscripts: str = self.prefix_subscripts
        self.core_input_subscripts: str = ""
        self.core_output_subscripts: str = ""
        self.core_dims: List[int] = []

        for axis, spec in self.indexed_dims.items():
            rank = spec["rank"]
            if rank is not None and rank > 0:
                rank_letter = next(self.core_letters)
                factor = get_indexed_fac_matrix(
                    int(spec["n_features"]),
                    min(int(spec["n_features"]), int(rank)),
                    same_values=bool(spec.get("same_values", False)),
                )
                self.indexed_factors.append(factor)
                self.indexed_factor_subscripts.append(self.prefix_subscripts + rank_letter)
                self.indexed_rank_subscripts += rank_letter
                self.core_dims.append(int(factor.shape[-1]))
            else:
                self.core_dims.append(int(spec["n_features"]))

        self.factors: nn.ParameterList = nn.ParameterList()
        in_dims: List[int] = []
        for rank, f_in in zip(ranks, self.in_features):
            x_sub, core_dim = self._add_feature_factor(rank, f_in, is_input=True)
            self.input_subscripts += x_sub
            in_dims.append(core_dim)

        out_dims: List[int] = []
        for rank, f_out in zip(ranks, self.out_features):
            x_sub, core_dim = self._add_feature_factor(rank, f_out, is_input=False)
            self.output_subscripts += x_sub
            out_dims.append(core_dim)

        self.core_subscripts: str = (
            self.prefix_subscripts
            + self.indexed_rank_subscripts
            + self.core_input_subscripts
            + self.core_output_subscripts
        )

        fan_in = math.prod(in_dims) if in_dims else 1
        bound = 1.0 / math.sqrt(fan_in)
        core = torch.empty(self.core_dims)
        nn.init.uniform_(core, -bound, bound)
        self.core: nn.Parameter = nn.Parameter(core, requires_grad=True)

        if bias:
            if self.bias_indexed_dims:
                bias_shape = [spec["n_features"] for spec in self.bias_indexed_dims.values()]
                bias_shape.extend(self.out_features)
            else:
                bias_shape = list(self.out_features)
            bias_ = torch.empty(bias_shape)
            nn.init.uniform_(bias_, -bound, bound)
            self.bias = nn.Parameter(bias_, requires_grad=True)
        else:
            self.register_parameter("bias", None)

    def _add_feature_factor(
        self,
        rank: Optional[Union[int, float]],
        features: int,
        is_input: bool,
    ) -> tuple[str, int]:
        core_sub = next(self.core_letters)

        if rank is not None and rank < features and rank > 0:
            fac_sub = next(self.factor_letters)
            self.factors.append(get_fac_matrix(int(rank), features))
            self.feature_factor_subscripts.append(core_sub + fac_sub)
            core_dim = int(rank)
            x_sub = fac_sub
        else:
            core_dim = features
            x_sub = core_sub

        if is_input:
            self.core_input_subscripts += core_sub
        else:
            self.core_output_subscripts += core_sub

        self.core_dims.append(core_dim)
        return x_sub, core_dim

    def _select_core(self, reference: torch.Tensor, emb: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if not self.indexed_dims:
            return self.core.view(*([1] * 5), *self.core.shape)

        runtime_indices = get_runtime_index_tensors(reference, indexed_dims=self.indexed_dims, emb=emb)
        has_runtime_selection = False
        index = []
        for axis, spec in self.indexed_dims.items():
            rank = spec["rank"]
            if rank is not None and rank > 0:
                index.append(slice(None))
            else:
                has_runtime_selection = True
                index.append(runtime_indices[axis])

        if not has_runtime_selection:
            return self.core.view(*([1] * 5), *self.core.shape)

        return self.core[tuple(index)]

    def _get_indexed_factor_tensors(
        self,
        reference: torch.Tensor,
        emb: Optional[Dict[str, Any]] = None,
    ) -> List[torch.Tensor]:
        if not self.indexed_dims or len(self.indexed_factors) == 0:
            return []

        runtime_indices = get_runtime_index_tensors(reference, indexed_dims=self.indexed_dims, emb=emb)
        factor_tensors: List[torch.Tensor] = []
        factor_idx = 0
        for axis, spec in self.indexed_dims.items():
            rank = spec["rank"]
            if rank is None or rank <= 0:
                continue
            factor_tensors.append(self.indexed_factors[factor_idx][runtime_indices[axis]])
            factor_idx += 1

        return factor_tensors

    def add_bias(self, x: torch.Tensor, bias: torch.Tensor):
        """
        Add shared or indexed bias with correct broadcasting over ``(v, t, n, d)``.
        """
        if bias.ndim == len(self.out_features):
            bias = bias.view(*([1] * 5), *self.out_features)
        return x + bias

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[Dict[str, Any]] = None,
        sample_configs: Dict[str, Any] = {},
    ):
        """
        Apply Tucker factorized transformation.

        :param x: Input tensor of shape ``(b, v, t, n, d, f_in...)``.
        :param emb: Optional embedding dictionary.
        :param sample_configs: Optional sampling configuration (unused).
        :return: Output tensor of shape ``(b, v, t, n, d, f_out...)``.
        """

        x_prefix = list(x.shape[:5])
        x = x.reshape(*x_prefix, *self.in_features)

        core = self._select_core(x, emb=emb)
        indexed_factors = self._get_indexed_factor_tensors(x, emb=emb)

        lhs = [self.input_subscripts, self.core_subscripts, *self.indexed_factor_subscripts, *self.feature_factor_subscripts]
        einsum_eq = f"{','.join(lhs)}->{self.output_subscripts}"
        x = torch.einsum(einsum_eq, x, core, *indexed_factors, *self.factors)
        x = x.reshape(*x_prefix, *self.out_features)

        if self.bias is not None:
            bias = select_indexed_tensor(self.bias, self.bias_indexed_dims, reference=x, emb=emb)
            x = self.add_bias(x, bias)

        return x
