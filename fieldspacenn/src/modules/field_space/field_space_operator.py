"""Learned operators over the canonical field-space axes.

This module deliberately lives next to, rather than inside, field-space
attention.  Ordinary feature projections still use :func:`get_layer` and the
existing ``TuckerFacLayer``; ``TuckerOperatorWeight`` only parameterizes the
operator acting on variable, outer-time, or outer-space positions.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from einops import rearrange
from omegaconf import ListConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import get_layer
from ..embedding.embedder import get_embedder
from ..grids.grid_layer import GridLayer
from .field_space_base import (
    GLOBAL_EMBEDDER_CACHE_KEY,
    LinEmbLayer,
    Tokenizer,
    add_depth_overlap_from_neighbor_patches,
    add_time_overlap_from_neighbor_patches,
    align_time_embeddings_to_tokens,
)


OperatorName = Literal["variable", "time", "space"]
ConstraintName = Literal["unconstrained", "softmax", "signed_softmax"]
_DEPENDENCY_ORDER = ("variable", "time", "space")
_FIELD_DIM_NAMES = ("batch", "variable", "time", "space", "depth")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple, ListConfig))


def _aligned_values(
    value: Optional[Sequence[Any]],
    n_operators: int,
    name: str,
    default: Any,
) -> List[Any]:
    """Return an exact-length operator setting without scalar broadcasting."""
    if value is None:
        return [copy.deepcopy(default) for _ in range(n_operators)]
    if not _is_sequence(value):
        raise ValueError(
            f"{name} must be a list of length {n_operators}; scalar broadcasting "
            "is not supported"
        )
    values = list(value)
    if len(values) != n_operators:
        raise ValueError(
            f"{name} must have length {n_operators}, got {len(values)}"
        )
    return values


def _axis_values(value: Any, keys: Sequence[int], name: str) -> Dict[int, Any]:
    """Normalize a scalar, mapping, or exact-length sequence over zoom keys."""
    keys = [int(key) for key in keys]
    if isinstance(value, Mapping):
        normalized = {int(key): item for key, item in value.items()}
        missing = [key for key in keys if key not in normalized]
        extra = [key for key in normalized if key not in keys]
        if missing or extra:
            raise ValueError(
                f"{name} keys must match {keys}; missing={missing}, extra={extra}"
            )
        return {key: normalized[key] for key in keys}
    if _is_sequence(value):
        values = list(value)
        if len(values) != len(keys):
            raise ValueError(
                f"{name} must have length {len(keys)}, got {len(values)}"
            )
        return dict(zip(keys, values))
    return {key: value for key in keys}


def _group_values(value: Any, count: int, name: str) -> List[Any]:
    if _is_sequence(value):
        values = list(value)
        if len(values) != count:
            raise ValueError(f"{name} must have length {count}, got {len(values)}")
        return values
    return [value for _ in range(count)]


def _contains_configured_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_configured_value(item) for item in value.values())
    if _is_sequence(value):
        return any(_contains_configured_value(item) for item in value)
    return value is not None


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _mode_product(tensor: torch.Tensor, factor: torch.Tensor, mode: int) -> torch.Tensor:
    """Replace ``tensor`` mode ``mode`` using a ``(full, rank)`` factor."""
    result = torch.tensordot(factor, tensor, dims=([1], [mode]))
    permutation = [*range(1, mode + 1), 0, *range(mode + 1, tensor.ndim)]
    return result.permute(permutation)


def _orthogonal_factor(
    size: int,
    rank: int,
    num_heads: Optional[int] = None,
) -> nn.Parameter:
    if num_heads is None:
        factor = torch.empty(size, rank)
        nn.init.orthogonal_(factor)
        return nn.Parameter(factor)

    factor = torch.empty(num_heads, size, rank)
    for head in range(num_heads):
        nn.init.orthogonal_(factor[head])
    return nn.Parameter(factor)


class _TuckerOperatorParameters(nn.Module):
    """One Tucker parameterization, used once or twice by a constrained weight."""

    def __init__(
        self,
        dependency_sizes: Mapping[str, int],
        num_heads: int,
        out_size: int,
        in_size: int,
        rank_out: Optional[int],
        rank_in: Optional[int],
        dependency_ranks: Mapping[str, Optional[int]],
        *,
        logits: bool,
        zero_core: bool,
        share_factors_across_heads: bool = False,
    ) -> None:
        super().__init__()
        self.dependency_names = tuple(dependency_sizes)
        self.dependency_sizes = OrderedDict(
            (name, int(size)) for name, size in dependency_sizes.items()
        )
        self.num_heads = int(num_heads)
        self.out_size = int(out_size)
        self.in_size = int(in_size)
        self.rank_out = None if rank_out is None else int(rank_out)
        self.rank_in = None if rank_in is None else int(rank_in)
        self.share_factors_across_heads = bool(share_factors_across_heads)
        self.dependency_ranks = {
            name: None if dependency_ranks.get(name) is None
            else int(dependency_ranks[name])
            for name in self.dependency_names
        }

        self.dependency_factors = nn.ParameterDict()
        core_shape: List[int] = []
        for name, size in self.dependency_sizes.items():
            rank = self.dependency_ranks[name]
            if rank is None:
                core_shape.append(size)
            else:
                if rank <= 0 or rank > size:
                    raise ValueError(
                        f"rank_{name} must be in [1, {size}], got {rank}"
                    )
                self.dependency_factors[name] = _orthogonal_factor(
                    size,
                    rank,
                    None if self.share_factors_across_heads else self.num_heads,
                )
                core_shape.append(rank)

        core_shape.append(self.num_heads)
        if self.rank_out is None:
            core_shape.append(self.out_size)
            self.register_parameter("out_factor", None)
        else:
            if self.rank_out <= 0 or self.rank_out > self.out_size:
                raise ValueError(
                    f"rank_out must be in [1, {self.out_size}], got {self.rank_out}"
                )
            self.out_factor = _orthogonal_factor(
                self.out_size,
                self.rank_out,
                None if self.share_factors_across_heads else self.num_heads,
            )
            core_shape.append(self.rank_out)

        if self.rank_in is None:
            core_shape.append(self.in_size)
            self.register_parameter("in_factor", None)
        else:
            if self.rank_in <= 0 or self.rank_in > self.in_size:
                raise ValueError(
                    f"rank_in must be in [1, {self.in_size}], got {self.rank_in}"
                )
            self.in_factor = _orthogonal_factor(
                self.in_size,
                self.rank_in,
                None if self.share_factors_across_heads else self.num_heads,
            )
            core_shape.append(self.rank_in)

        core = torch.empty(core_shape)
        if zero_core:
            nn.init.zeros_(core)
        elif logits:
            nn.init.normal_(core, mean=0.0, std=1e-2)
        else:
            fan_in = self.rank_in if self.rank_in is not None else self.in_size
            nn.init.uniform_(core, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))
        self.core = nn.Parameter(core)

    @property
    def n_dependencies(self) -> int:
        return len(self.dependency_names)

    def _normalize_ids(
        self,
        indices: Optional[torch.Tensor],
        size: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if indices is None:
            return torch.arange(size, device=device, dtype=torch.long)
        indices = indices.to(device=device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError(f"Shared {name} indices must be one-dimensional")
        if indices.numel() and (
            int(indices.min().item()) < 0 or int(indices.max().item()) >= size
        ):
            raise ValueError(f"{name} indices must be in [0, {size - 1}]")
        return indices

    def _factor_for_head(
        self,
        factor: torch.Tensor,
        head: int,
    ) -> torch.Tensor:
        return factor if self.share_factors_across_heads else factor[head]

    def _selected_core_head(
        self,
        dependency_indices: Mapping[str, torch.Tensor],
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
        head: int,
    ) -> torch.Tensor:
        core = self.core.select(self.n_dependencies, head)
        for mode, name in enumerate(self.dependency_names):
            indices = dependency_indices[name]
            rank = self.dependency_ranks[name]
            if rank is None:
                core = torch.index_select(core, mode, indices)
            else:
                factor = torch.index_select(
                    self._factor_for_head(
                        self.dependency_factors[name], head
                    ),
                    0,
                    indices,
                )
                core = _mode_product(core, factor, mode)

        out_mode = self.n_dependencies
        in_mode = self.n_dependencies + 1
        if self.rank_out is None:
            core = torch.index_select(core, out_mode, output_indices)
        if self.rank_in is None:
            core = torch.index_select(core, in_mode, input_indices)
        return core

    def raw_tile(
        self,
        dependency_indices: Mapping[str, torch.Tensor],
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize only the selected dependency/output/input tile."""
        head_tiles: List[torch.Tensor] = []
        out_mode = self.n_dependencies
        in_mode = self.n_dependencies + 1
        for head in range(self.num_heads):
            core = self._selected_core_head(
                dependency_indices, output_indices, input_indices, head
            )
            if self.rank_out is not None:
                out_factor = torch.index_select(
                    self._factor_for_head(self.out_factor, head),
                    0,
                    output_indices,
                )
                core = _mode_product(core, out_factor, out_mode)
            if self.rank_in is not None:
                in_factor = torch.index_select(
                    self._factor_for_head(self.in_factor, head),
                    0,
                    input_indices,
                )
                core = _mode_product(core, in_factor, in_mode)
            head_tiles.append(core)
        return torch.stack(head_tiles, dim=-3)

    def contract_unconstrained(
        self,
        values: torch.Tensor,
        dependency_indices: Mapping[str, torch.Tensor],
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Contract without constructing the dense output-by-input operator."""
        head_results: List[torch.Tensor] = []
        for head in range(self.num_heads):
            core = self._selected_core_head(
                dependency_indices, output_indices, input_indices, head
            )
            head_values = values.select(-3, head)
            if self.rank_in is not None:
                in_factor = torch.index_select(
                    self._factor_for_head(self.in_factor, head),
                    0,
                    input_indices,
                )
                head_values = torch.einsum(
                    "b...ic,ir->b...rc", head_values, in_factor
                )
            result = torch.einsum(
                "...oi,b...ic->b...oc", core, head_values
            )
            if self.rank_out is not None:
                out_factor = torch.index_select(
                    self._factor_for_head(self.out_factor, head),
                    0,
                    output_indices,
                )
                result = torch.einsum("b...rc,or->b...oc", result, out_factor)
            head_results.append(result)
        return torch.stack(head_results, dim=-3)


class TuckerOperatorWeight(nn.Module):
    """Tucker-capable learned operator over physical/token sequence axes.

    Values passed to :meth:`contract` have shape
    ``(batch, *dependencies, heads, source, head_channels)``.  The returned
    tensor replaces ``source`` with ``target``.  Head channels are never mixed.

    Identity initialization represents an unconstrained operator as a fixed
    self-selection map plus a zero-initialized Tucker correction.  Constrained
    softmax operators use the same zero correction in logit space together
    with a fixed self-logit bias.  Tucker factors are independent per head by
    default; ``share_factors_across_heads=True`` enables the compressed shared-
    subspace variant.
    """

    _SOFTMAX_CHUNK_SIZE = 256

    def __init__(
        self,
        *,
        dependency_sizes: Optional[Mapping[str, int]] = None,
        num_heads: int,
        out_size: int,
        in_size: int,
        rank_out: Optional[int] = None,
        rank_in: Optional[int] = None,
        dependency_ranks: Optional[Mapping[str, Optional[int]]] = None,
        constraint: ConstraintName = "unconstrained",
        initialization: str = "identity",
        identity_probability: float = 0.99,
        self_indices: Optional[torch.Tensor] = None,
        share_factors_across_heads: bool = False,
    ) -> None:
        super().__init__()
        dependency_sizes = {} if dependency_sizes is None else dependency_sizes
        dependency_ranks = {} if dependency_ranks is None else dependency_ranks
        ordered_sizes = OrderedDict(
            (name, int(dependency_sizes[name]))
            for name in _DEPENDENCY_ORDER
            if name in dependency_sizes
        )
        unknown = sorted(set(dependency_sizes).difference(_DEPENDENCY_ORDER))
        if unknown:
            raise ValueError(f"Unsupported operator dependencies: {unknown}")
        if constraint not in {"unconstrained", "softmax", "signed_softmax"}:
            raise ValueError(f"Unsupported operator constraint {constraint!r}")
        if initialization not in {"identity", "random"}:
            raise ValueError(
                "Operator initialization must be 'identity' or 'random'"
            )
        if int(num_heads) <= 0 or int(out_size) <= 0 or int(in_size) <= 0:
            raise ValueError("num_heads, out_size, and in_size must be positive")
        if not 0.0 < float(identity_probability) < 1.0:
            raise ValueError("identity_probability must be strictly between 0 and 1")

        self.dependency_names = tuple(ordered_sizes)
        self.dependency_sizes = ordered_sizes
        self.num_heads = int(num_heads)
        self.out_size = int(out_size)
        self.in_size = int(in_size)
        self.constraint = constraint
        self.initialization = initialization
        self.identity_probability = float(identity_probability)
        self.share_factors_across_heads = bool(share_factors_across_heads)
        self._softmax_chunk_size = self._SOFTMAX_CHUNK_SIZE

        if self_indices is None:
            if self.out_size > self.in_size and initialization == "identity":
                raise ValueError(
                    "Identity initialization requires one self source for every target"
                )
            self_indices = (
                torch.arange(self.out_size, dtype=torch.long) % self.in_size
            )
        else:
            self_indices = torch.as_tensor(self_indices, dtype=torch.long)
        if self_indices.ndim != 1 or self_indices.numel() != self.out_size:
            raise ValueError(
                f"self_indices must have shape ({self.out_size},), got "
                f"{tuple(self_indices.shape)}"
            )
        if self_indices.numel() and (
            int(self_indices.min().item()) < 0
            or int(self_indices.max().item()) >= self.in_size
        ):
            raise ValueError(
                f"self_indices must be in [0, {self.in_size - 1}]"
            )
        self.register_buffer("self_indices", self_indices, persistent=True)

        parameter_kwargs = dict(
            dependency_sizes=ordered_sizes,
            num_heads=self.num_heads,
            out_size=self.out_size,
            in_size=self.in_size,
            rank_out=rank_out,
            rank_in=rank_in,
            dependency_ranks=dependency_ranks,
            zero_core=initialization == "identity",
            share_factors_across_heads=self.share_factors_across_heads,
        )
        if constraint == "signed_softmax":
            self.positive = _TuckerOperatorParameters(**parameter_kwargs, logits=True)
            self.negative = _TuckerOperatorParameters(**parameter_kwargs, logits=True)
            positive_gain = 1.0 if initialization == "identity" else 0.1
            negative_gain = 1e-4 if initialization == "identity" else 0.1
            self.positive_gain_raw = nn.Parameter(
                torch.full((self.num_heads,), _inverse_softplus(positive_gain))
            )
            self.negative_gain_raw = nn.Parameter(
                torch.full((self.num_heads,), _inverse_softplus(negative_gain))
            )
            self.parameters_single = None
        else:
            self.parameters_single = _TuckerOperatorParameters(
                **parameter_kwargs,
                logits=constraint == "softmax",
            )
            self.positive = None
            self.negative = None
            self.register_parameter("positive_gain_raw", None)
            self.register_parameter("negative_gain_raw", None)

    def _self_selection_contract(
        self,
        values: torch.Tensor,
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the fixed self-selection map without materializing a matrix."""
        lookup = torch.full(
            (self.in_size,),
            -1,
            device=values.device,
            dtype=torch.long,
        )
        lookup.scatter_(
            0,
            input_indices,
            torch.arange(input_indices.numel(), device=values.device),
        )
        source_positions = lookup[self.self_indices[output_indices]]
        present = source_positions >= 0
        selected = torch.index_select(
            values,
            -2,
            source_positions.clamp_min(0),
        )
        present_shape = [1] * selected.ndim
        present_shape[-2] = present.numel()
        return selected * present.view(present_shape).to(selected.dtype)

    def _self_logit_bias(
        self,
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
        n_runtime_sources: int,
    ) -> torch.Tensor:
        self_sources = self.self_indices[output_indices].to(device=device)
        is_self = self_sources.unsqueeze(-1) == input_indices.unsqueeze(0)
        n_sources = int(n_runtime_sources)
        if n_sources <= 1:
            bias_value = 0.0
        else:
            bias_value = math.log(
                self.identity_probability
                * (n_sources - 1)
                / (1.0 - self.identity_probability)
            )
        shape = [1] * len(self.dependency_names) + [1, *is_self.shape]
        return is_self.to(device=device, dtype=dtype).view(shape) * bias_value

    def _raw_logit_tile(
        self,
        parameters: _TuckerOperatorParameters,
        dependency_indices: Mapping[str, torch.Tensor],
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
        *,
        with_identity_bias: bool,
        n_runtime_sources: int,
    ) -> torch.Tensor:
        logits = parameters.raw_tile(
            dependency_indices, output_indices, input_indices
        )
        if with_identity_bias and self.initialization == "identity":
            logits = logits + self._self_logit_bias(
                output_indices,
                input_indices,
                dtype=logits.dtype,
                device=logits.device,
                n_runtime_sources=n_runtime_sources,
            )
        return logits

    def _normalize_runtime_indices(
        self,
        values: torch.Tensor,
        dependency_indices: Optional[Mapping[str, torch.Tensor]],
        output_indices: Optional[torch.Tensor],
        input_indices: Optional[torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        dependency_indices = (
            {} if dependency_indices is None else dict(dependency_indices)
        )
        unknown = sorted(set(dependency_indices).difference(self.dependency_names))
        if unknown:
            raise ValueError(f"Indices supplied for inactive dependencies: {unknown}")
        device = values.device
        normalized: Dict[str, torch.Tensor] = {}
        for offset, name in enumerate(self.dependency_names):
            indices = dependency_indices.get(name)
            if indices is None:
                indices = torch.arange(
                    values.shape[1 + offset], device=device, dtype=torch.long
                )
            else:
                indices = indices.to(device=device, dtype=torch.long)
            if indices.ndim not in {1, 2}:
                raise ValueError(
                    f"{name} dependency indices must be one- or two-dimensional"
                )
            if indices.shape[-1] != values.shape[1 + offset]:
                raise ValueError(
                    f"{name} dependency index length {indices.shape[-1]} does not "
                    f"match runtime size {values.shape[1 + offset]}"
                )
            if indices.ndim == 2 and indices.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Batch-specific {name} indices must have batch size "
                    f"{values.shape[0]}, got {indices.shape[0]}"
                )
            size = self.dependency_sizes[name]
            if indices.numel() and (
                int(indices.min().item()) < 0 or int(indices.max().item()) >= size
            ):
                raise ValueError(f"{name} dependency indices must be in [0, {size - 1}]")
            normalized[name] = indices

        def normalize_sequence(
            indices: Optional[torch.Tensor], size: int, runtime: int, name: str
        ) -> torch.Tensor:
            if indices is None:
                if runtime != size:
                    raise ValueError(
                        f"Runtime {name} size is {runtime}, configured size is {size}; "
                        f"explicit {name}_indices are required"
                    )
                return torch.arange(size, device=device, dtype=torch.long)
            indices = indices.to(device=device, dtype=torch.long)
            if indices.ndim not in {1, 2} or indices.shape[-1] != runtime:
                raise ValueError(
                    f"{name}_indices must end in runtime size {runtime}, got "
                    f"{tuple(indices.shape)}"
                )
            if indices.ndim == 2 and indices.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Batch-specific {name}_indices must have batch size {values.shape[0]}"
                )
            if indices.numel() and (
                int(indices.min().item()) < 0 or int(indices.max().item()) >= size
            ):
                raise ValueError(f"{name}_indices must be in [0, {size - 1}]")
            return indices

        input_indices = normalize_sequence(
            input_indices, self.in_size, int(values.shape[-2]), "input"
        )
        runtime_out = (
            self.out_size if output_indices is None else int(output_indices.shape[-1])
        )
        output_indices = normalize_sequence(
            output_indices, self.out_size, runtime_out, "output"
        )
        return normalized, output_indices, input_indices

    @staticmethod
    def _has_batch_indices(
        dependency_indices: Mapping[str, torch.Tensor],
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
    ) -> bool:
        return any(index.ndim == 2 for index in dependency_indices.values()) or (
            output_indices.ndim == 2 or input_indices.ndim == 2
        )

    def _shared_indices(
        self,
        parameters: _TuckerOperatorParameters,
        dependency_indices: Mapping[str, torch.Tensor],
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        deps = {
            name: parameters._normalize_ids(
                dependency_indices.get(name),
                parameters.dependency_sizes[name],
                parameters.core.device,
                name,
            )
            for name in parameters.dependency_names
        }
        out_ids = parameters._normalize_ids(
            output_indices,
            parameters.out_size,
            parameters.core.device,
            "output",
        )
        in_ids = parameters._normalize_ids(
            input_indices,
            parameters.in_size,
            parameters.core.device,
            "input",
        )
        return deps, out_ids, in_ids

    def _softmax_contract_shared(
        self,
        parameters: _TuckerOperatorParameters,
        values: torch.Tensor,
        dependency_indices: Mapping[str, torch.Tensor],
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
        *,
        with_identity_bias: bool,
    ) -> torch.Tensor:
        deps, out_ids, in_ids = self._shared_indices(
            parameters, dependency_indices, output_indices, input_indices
        )
        output_chunks: List[torch.Tensor] = []
        chunk_size = self._softmax_chunk_size
        for output_start in range(0, out_ids.numel(), chunk_size):
            out_chunk = out_ids[output_start : output_start + chunk_size]
            running_max: Optional[torch.Tensor] = None
            denominator: Optional[torch.Tensor] = None
            for input_start in range(0, in_ids.numel(), chunk_size):
                in_chunk = in_ids[input_start : input_start + chunk_size]
                logits = self._raw_logit_tile(
                    parameters,
                    deps,
                    out_chunk,
                    in_chunk,
                    with_identity_bias=with_identity_bias,
                    n_runtime_sources=int(in_ids.numel()),
                )
                chunk_max = logits.amax(dim=-1)
                if running_max is None:
                    running_max = chunk_max
                    denominator = torch.exp(
                        logits - running_max.unsqueeze(-1)
                    ).sum(dim=-1)
                else:
                    new_max = torch.maximum(running_max, chunk_max)
                    assert denominator is not None
                    denominator = (
                        denominator * torch.exp(running_max - new_max)
                        + torch.exp(logits - new_max.unsqueeze(-1)).sum(dim=-1)
                    )
                    running_max = new_max

            assert running_max is not None and denominator is not None
            numerator: Optional[torch.Tensor] = None
            for input_start in range(0, in_ids.numel(), chunk_size):
                in_chunk = in_ids[input_start : input_start + chunk_size]
                logits = self._raw_logit_tile(
                    parameters,
                    deps,
                    out_chunk,
                    in_chunk,
                    with_identity_bias=with_identity_bias,
                    n_runtime_sources=int(in_ids.numel()),
                )
                weights = torch.exp(logits - running_max.unsqueeze(-1))
                value_chunk = values[..., input_start : input_start + in_chunk.numel(), :]
                contribution = torch.einsum(
                    "...hoi,b...hic->b...hoc", weights, value_chunk
                )
                numerator = (
                    contribution if numerator is None else numerator + contribution
                )
            output_chunks.append(
                numerator / denominator.unsqueeze(0).unsqueeze(-1)
            )
        return torch.cat(output_chunks, dim=-2)

    def _contract_shared(
        self,
        values: torch.Tensor,
        dependency_indices: Mapping[str, torch.Tensor],
        output_indices: torch.Tensor,
        input_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.constraint == "unconstrained":
            assert self.parameters_single is not None
            deps, out_ids, in_ids = self._shared_indices(
                self.parameters_single,
                dependency_indices,
                output_indices,
                input_indices,
            )
            delta = self.parameters_single.contract_unconstrained(
                values, deps, out_ids, in_ids
            )
            if self.initialization != "identity":
                return delta
            return delta + self._self_selection_contract(
                values, out_ids, in_ids
            )
        if self.constraint == "softmax":
            assert self.parameters_single is not None
            return self._softmax_contract_shared(
                self.parameters_single,
                values,
                dependency_indices,
                output_indices,
                input_indices,
                with_identity_bias=True,
            )

        assert self.positive is not None and self.negative is not None
        positive = self._softmax_contract_shared(
            self.positive,
            values,
            dependency_indices,
            output_indices,
            input_indices,
            with_identity_bias=True,
        )
        negative = self._softmax_contract_shared(
            self.negative,
            values,
            dependency_indices,
            output_indices,
            input_indices,
            with_identity_bias=False,
        )
        gain_shape = [1] * positive.ndim
        gain_shape[-3] = self.num_heads
        positive_gain = F.softplus(self.positive_gain_raw).view(gain_shape)
        negative_gain = F.softplus(self.negative_gain_raw).view(gain_shape)
        return positive_gain * positive - negative_gain * negative

    def contract(
        self,
        values: torch.Tensor,
        *,
        dependency_indices: Optional[Mapping[str, torch.Tensor]] = None,
        output_indices: Optional[torch.Tensor] = None,
        input_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the operator without mixing the final head-channel dimension."""
        expected_ndim = 4 + len(self.dependency_names)
        if values.ndim != expected_ndim:
            raise ValueError(
                f"Expected values with {expected_ndim} dimensions "
                f"(batch, dependencies, heads, source, channels), got "
                f"shape {tuple(values.shape)}"
            )
        if values.shape[-3] != self.num_heads:
            raise ValueError(
                f"Expected {self.num_heads} heads, got {values.shape[-3]}"
            )
        deps, out_ids, in_ids = self._normalize_runtime_indices(
            values, dependency_indices, output_indices, input_indices
        )
        if not self._has_batch_indices(deps, out_ids, in_ids):
            return self._contract_shared(values, deps, out_ids, in_ids)

        outputs = []
        for batch_index in range(values.shape[0]):
            batch_deps = {
                name: index[batch_index] if index.ndim == 2 else index
                for name, index in deps.items()
            }
            batch_out = out_ids[batch_index] if out_ids.ndim == 2 else out_ids
            batch_in = in_ids[batch_index] if in_ids.ndim == 2 else in_ids
            outputs.append(
                self._contract_shared(
                    values[batch_index : batch_index + 1],
                    batch_deps,
                    batch_out,
                    batch_in,
                )
            )
        return torch.cat(outputs, dim=0)

    def to_dense(self, max_elements: Optional[int] = 10_000_000) -> torch.Tensor:
        """Materialize the effective operator for small diagnostics and tests."""
        element_count = (
            math.prod(self.dependency_sizes.values())
            * self.num_heads
            * self.out_size
            * self.in_size
        )
        if max_elements is not None and element_count > max_elements:
            raise ValueError(
                f"Dense operator would contain {element_count} elements; "
                f"limit is {max_elements}"
            )
        device = next(self.parameters()).device
        deps = {
            name: torch.arange(size, device=device)
            for name, size in self.dependency_sizes.items()
        }
        out_ids = torch.arange(self.out_size, device=device)
        in_ids = torch.arange(self.in_size, device=device)

        def constrained_dense(
            parameters: _TuckerOperatorParameters,
            *,
            with_identity_bias: bool,
        ) -> torch.Tensor:
            logits = self._raw_logit_tile(
                parameters,
                deps,
                out_ids,
                in_ids,
                with_identity_bias=with_identity_bias,
                n_runtime_sources=int(in_ids.numel()),
            )
            return torch.softmax(logits, dim=-1)

        if self.constraint == "unconstrained":
            assert self.parameters_single is not None
            dense = self.parameters_single.raw_tile(deps, out_ids, in_ids)
            if self.initialization == "identity":
                self_dense = (
                    self.self_indices[out_ids].unsqueeze(-1)
                    == in_ids.unsqueeze(0)
                ).to(dtype=dense.dtype, device=dense.device)
                shape = [1] * len(self.dependency_names) + [1, *self_dense.shape]
                dense = dense + self_dense.view(shape)
            return dense
        if self.constraint == "softmax":
            assert self.parameters_single is not None
            return constrained_dense(
                self.parameters_single,
                with_identity_bias=True,
            )
        assert self.positive is not None and self.negative is not None
        shape = [1] * (len(self.dependency_names) + 3)
        shape[-3] = self.num_heads
        return (
            F.softplus(self.positive_gain_raw).view(shape)
            * constrained_dense(self.positive, with_identity_bias=True)
            - F.softplus(self.negative_gain_raw).view(shape)
            * constrained_dense(self.negative, with_identity_bias=False)
        )


class FieldSpaceOperator(nn.Module):
    """Pack and apply one atomic operator to projected canonical values."""

    def __init__(
        self,
        *,
        operator: OperatorName,
        operator_dim: int,
        num_heads: int,
        n_variables: int,
        n_times: int,
        token_zoom: int,
        sequence_zoom: Optional[int],
        include_neighbors: bool,
        include_dependencies: Mapping[str, bool],
        dependency_ranks: Mapping[str, Optional[int]],
        rank_in: Optional[int],
        rank_out: Optional[int],
        constraint: ConstraintName,
        initialization: str = "identity",
        share_factors_across_heads: bool = False,
        grid_layers: Mapping[str, GridLayer],
    ) -> None:
        super().__init__()
        if operator not in _DEPENDENCY_ORDER:
            raise ValueError(f"Unsupported atomic operator {operator!r}")
        if operator_dim % num_heads != 0:
            raise ValueError(
                f"operator_dim ({operator_dim}) must be divisible by num_heads "
                f"({num_heads}) for {operator!r}"
            )
        self.operator = operator
        self.operator_dim = int(operator_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.operator_dim // self.num_heads
        self.n_variables = int(n_variables)
        self.n_times = int(n_times)
        self.token_zoom = int(token_zoom)
        self.sequence_zoom = sequence_zoom
        self.include_neighbors = bool(include_neighbors)
        self.dependency_names = tuple(
            name for name in _DEPENDENCY_ORDER if include_dependencies.get(name, False)
        )

        self.grid_layer_field: Optional[GridLayer] = None
        self.grid_layer_sequence: Optional[GridLayer] = None
        self.local_space = operator == "space" and sequence_zoom is not None and sequence_zoom >= 0
        self.global_space = operator == "space" and sequence_zoom == -1
        n_space = 1
        n_regions = 1
        self_indices: Optional[torch.Tensor] = None
        global_space_ids = torch.empty(0, dtype=torch.long)
        region_space_ids = torch.empty(0, dtype=torch.long)
        if operator == "space" or "space" in self.dependency_names:
            if token_zoom < 0:
                raise ValueError("Spatial operators/dependencies require token_zoom >= 0")
            self.grid_layer_field = grid_layers[str(token_zoom)]
            n_space = int(self.grid_layer_field.adjc.shape[0])
            global_space_ids = (
                self.grid_layer_field.get_idx_of_patch().reshape(-1).to(torch.long)
            )
            if global_space_ids.numel() != n_space:
                raise ValueError("Grid metadata returned an invalid number of token IDs")
        if self.local_space:
            if int(sequence_zoom) > token_zoom:
                raise ValueError("sequence_zoom cannot exceed token_zoom")
            self.grid_layer_sequence = grid_layers[str(int(sequence_zoom))]
            n_regions = int(self.grid_layer_sequence.adjc.shape[0])
            region_space_ids = (
                self.grid_layer_sequence.get_idx_of_patch().reshape(-1).to(torch.long)
            )
            if region_space_ids.numel() != n_regions:
                raise ValueError("Grid metadata returned an invalid number of region IDs")
            if n_space % n_regions != 0:
                raise ValueError("Token grid cannot be partitioned into sequence regions")
            out_size = n_space // n_regions
            source_indices, target_indices = self._build_local_spatial_indices(
                n_space=n_space,
                n_regions=n_regions,
                out_size=out_size,
                global_space_ids=global_space_ids,
            )
            in_size = int(source_indices.shape[-1]) if include_neighbors else out_size
            if not include_neighbors:
                source_indices = target_indices.clone()
            self.register_buffer("source_indices", source_indices, persistent=False)
            self.register_buffer("target_indices", target_indices, persistent=False)
            matches = source_indices.unsqueeze(-2) == target_indices.unsqueeze(-1)
            if not matches.any(dim=-1).all():
                raise ValueError("Every local target must occur in its source context")
            self.register_buffer(
                "self_source_positions",
                matches.to(torch.int64).argmax(dim=-1),
                persistent=False,
            )
            if not torch.equal(
                self.self_source_positions,
                self.self_source_positions[:1].expand_as(
                    self.self_source_positions
                ),
            ):
                raise ValueError(
                    "The current local operator parameterization requires the "
                    "grid-derived central source positions to be identical across "
                    "sequence regions"
                )
            self_indices = self.self_source_positions[0].clone()
        elif self.global_space:
            out_size = in_size = n_space
            indices = global_space_ids.view(1, n_space)
            self.register_buffer("source_indices", indices, persistent=False)
            self.register_buffer("target_indices", indices.clone(), persistent=False)
            self.register_buffer(
                "self_source_positions",
                torch.arange(n_space).view(1, n_space),
                persistent=False,
            )
            self_indices = torch.arange(n_space)
        elif operator == "variable":
            out_size = in_size = self.n_variables
            self_indices = torch.arange(out_size)
        else:
            out_size = in_size = self.n_times
            self_indices = torch.arange(out_size)

        self.register_buffer("global_space_ids", global_space_ids, persistent=False)
        self.register_buffer("region_space_ids", region_space_ids, persistent=False)

        dependency_sizes: "OrderedDict[str, int]" = OrderedDict()
        for name in self.dependency_names:
            if name == "variable":
                dependency_sizes[name] = self.n_variables
            elif name == "time":
                dependency_sizes[name] = self.n_times
            elif self.local_space and operator == "space":
                dependency_sizes[name] = n_regions
            else:
                dependency_sizes[name] = n_space

        self.weight = TuckerOperatorWeight(
            dependency_sizes=dependency_sizes,
            num_heads=self.num_heads,
            out_size=out_size,
            in_size=in_size,
            rank_out=rank_out,
            rank_in=rank_in,
            dependency_ranks=dependency_ranks,
            constraint=constraint,
            initialization=initialization,
            self_indices=self_indices,
            share_factors_across_heads=share_factors_across_heads,
        )

    def _build_local_spatial_indices(
        self,
        *,
        n_space: int,
        n_regions: int,
        out_size: int,
        global_space_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = global_space_ids.view(n_regions, out_size)
        if not self.include_neighbors:
            return target.clone(), target
        assert self.grid_layer_sequence is not None
        sentinel = global_space_ids.view(1, 1, 1, n_space, 1, 1)
        gathered, _ = self.grid_layer_sequence.get_nh(
            sentinel, input_zoom=self.token_zoom
        )
        source = gathered.reshape(n_regions, -1).to(torch.long)
        return source, target

    @staticmethod
    def _variable_ids(
        values: torch.Tensor, emb: Optional[Mapping[str, Any]]
    ) -> torch.Tensor:
        ids = None
        if emb is not None:
            ids = emb.get("variables_sampled", emb.get("VariableEmbedder"))
        if ids is None:
            return torch.arange(values.shape[1], device=values.device)
        ids = ids.to(device=values.device, dtype=torch.long)
        if ids.ndim == 1:
            if ids.numel() != values.shape[1]:
                raise ValueError("Variable index count does not match the field")
            return ids
        if ids.ndim == 2:
            if tuple(ids.shape) != (values.shape[0], values.shape[1]):
                raise ValueError(
                    "Batch variable indices must have shape "
                    f"{(values.shape[0], values.shape[1])}, got {tuple(ids.shape)}"
                )
            return ids
        raise ValueError("Variable indices must be one- or two-dimensional")

    @staticmethod
    def _expand_batch_indices(
        indices: torch.Tensor,
        passive_shape: Sequence[int],
    ) -> torch.Tensor:
        if indices.ndim == 1:
            return indices
        view_shape = [indices.shape[0], *([1] * len(passive_shape)), indices.shape[1]]
        expand_shape = [indices.shape[0], *passive_shape, indices.shape[1]]
        return indices.view(view_shape).expand(expand_shape).reshape(-1, indices.shape[1])

    def _pack(
        self,
        values: torch.Tensor,
        dim_names: Sequence[str],
        sequence_name: str,
        variable_ids: torch.Tensor,
        *,
        spatial_dependency_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Dict[str, Any],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        axis = {name: index for index, name in enumerate(dim_names)}
        passive = [
            name
            for name in dim_names
            if name not in {"batch", sequence_name, *self.dependency_names}
        ]
        current_names = [
            "batch",
            *passive,
            *self.dependency_names,
            "head",
            sequence_name,
            "channel",
        ]
        full_axis = {
            **axis,
            "head": len(dim_names),
            "channel": len(dim_names) + 1,
        }
        permutation = [full_axis[name] for name in current_names]
        packed = values.permute(permutation)
        batch_size = int(values.shape[axis["batch"]])
        passive_shape = [int(values.shape[axis[name]]) for name in passive]
        dependency_shape = [
            int(values.shape[axis[name]]) for name in self.dependency_names
        ]
        sequence_size = int(values.shape[axis[sequence_name]])
        packed = packed.reshape(
            batch_size * math.prod(passive_shape),
            *dependency_shape,
            self.num_heads,
            sequence_size,
            self.head_dim,
        )

        dependency_indices: Dict[str, torch.Tensor] = {}
        for name in self.dependency_names:
            if name == "variable":
                dependency_indices[name] = self._expand_batch_indices(
                    variable_ids, passive_shape
                )
            elif name == "space" and spatial_dependency_ids is not None:
                dependency_indices[name] = spatial_dependency_ids
            else:
                dependency_indices[name] = torch.arange(
                    values.shape[axis[name]], device=values.device
                )

        if sequence_name == "variable":
            sequence_indices = self._expand_batch_indices(
                variable_ids, passive_shape
            )
        else:
            sequence_indices = torch.arange(sequence_size, device=values.device)

        metadata = {
            "dim_names": list(dim_names),
            "sequence_name": sequence_name,
            "current_names": current_names,
            "batch_size": batch_size,
            "passive_shape": passive_shape,
            "dependency_shape": dependency_shape,
        }
        return (
            packed,
            metadata,
            dependency_indices,
            sequence_indices,
            sequence_indices,
        )

    def _unpack(self, values: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
        current_names = list(metadata["current_names"])
        dim_names = list(metadata["dim_names"])
        sequence_name = str(metadata["sequence_name"])
        values = values.reshape(
            metadata["batch_size"],
            *metadata["passive_shape"],
            *metadata["dependency_shape"],
            self.num_heads,
            values.shape[-2],
            self.head_dim,
        )
        desired_names = [*dim_names, "head", "channel"]
        # The output target occupies the same named slot as the source sequence.
        if sequence_name not in desired_names:
            raise RuntimeError("Sequence name was lost while unpacking operator output")
        permutation = [current_names.index(name) for name in desired_names]
        return values.permute(permutation)

    def forward(
        self,
        values: torch.Tensor,
        *,
        emb: Optional[Mapping[str, Any]] = None,
        sample_config: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply the configured operator to ``(b,v,T,N,D,c)`` values."""
        if values.ndim != 6 or values.shape[-1] != self.operator_dim:
            raise ValueError(
                "FieldSpaceOperator expects (b,v,T,N,D,operator_dim), got "
                f"{tuple(values.shape)}"
            )
        if (
            self.operator == "time" or "time" in self.dependency_names
        ) and values.shape[2] != self.n_times:
            raise ValueError(
                f"Configured n_times={self.n_times}, but the tokenized outer time "
                f"axis has size {values.shape[2]}"
            )
        if "space" in self.dependency_names:
            configured_space = self.weight.dependency_sizes["space"]
            runtime_space = (
                values.shape[3] // self.weight.out_size
                if self.local_space and self.operator == "space"
                else values.shape[3]
            )
            if runtime_space != configured_space:
                raise ValueError(
                    "Spatially dependent operators require grid-derived global "
                    f"indices for every runtime position; expected {configured_space} "
                    f"positions, got {runtime_space}"
                )
        variable_ids = self._variable_ids(values, emb)
        values = values.view(*values.shape[:-1], self.num_heads, self.head_dim)

        if self.local_space:
            assert self.grid_layer_sequence is not None
            b, v, t, n, d, heads, channels = values.shape
            expected_full = int(self.target_indices.numel())
            config = {} if sample_config is None else dict(sample_config)
            if n == expected_full:
                source, _ = self.grid_layer_sequence.get_nh(
                    values,
                    input_zoom=self.token_zoom,
                    **config,
                ) if self.include_neighbors else (None, None)
                target = values.reshape(
                    b, v, t, self.target_indices.shape[0],
                    self.target_indices.shape[1], d, heads, channels
                )
                if not self.include_neighbors:
                    source = target
                spatial_ids = self.region_space_ids
            else:
                if "space" in self.dependency_names:
                    raise ValueError(
                        "Spatially dependent local operators currently require a full grid"
                    )
                if not config:
                    raise ValueError(
                        "A partial local spatial sequence requires sampling metadata"
                    )
                source, _ = self.grid_layer_sequence.get_nh(
                    values, input_zoom=self.token_zoom, **config
                ) if self.include_neighbors else (None, None)
                n_regions = n // self.weight.out_size
                target = values.reshape(
                    b, v, t, n_regions, self.weight.out_size,
                    d, heads, channels
                )
                if not self.include_neighbors:
                    source = target
                spatial_ids = None

            source = source.permute(0, 1, 2, 3, 5, 4, 6, 7)
            packed, metadata, dep_ids, _, _ = self._pack(
                source,
                ["batch", "variable", "time", "space", "depth", "sequence"],
                "sequence",
                variable_ids,
                spatial_dependency_ids=spatial_ids,
            )
            output = self.weight.contract(
                packed,
                dependency_indices=dep_ids,
                input_indices=torch.arange(packed.shape[-2], device=values.device),
                output_indices=torch.arange(self.weight.out_size, device=values.device),
            )
            output = self._unpack(output, metadata)
            output = output.permute(0, 1, 2, 3, 5, 4, 6, 7)
            output = output.reshape(b, v, t, -1, d, heads, channels)
        else:
            sequence_name = self.operator
            packed, metadata, dep_ids, output_ids, input_ids = self._pack(
                values,
                _FIELD_DIM_NAMES,
                sequence_name,
                variable_ids,
                spatial_dependency_ids=(
                    self.global_space_ids
                    if "space" in self.dependency_names
                    else None
                ),
            )
            if self.global_space and values.shape[3] != self.weight.in_size:
                raise ValueError(
                    "Global spatial operators require the complete token grid; "
                    f"expected {self.weight.in_size} tokens, got {values.shape[3]}"
                )
            if self.global_space:
                output_ids = self.global_space_ids
                input_ids = self.global_space_ids
            output = self.weight.contract(
                packed,
                dependency_indices=dep_ids,
                output_indices=output_ids,
                input_indices=input_ids,
            )
            output = self._unpack(output, metadata)

        return output.reshape(*output.shape[:-2], self.operator_dim)


class FieldSpaceOperatorConfig:
    """Configuration object consumed by the multigrid model builder."""

    def __init__(
        self,
        token_zoom: int,
        operators: Sequence[OperatorName],
        num_heads: Optional[Sequence[int]] = None,
        *,
        in_zooms: Union[Sequence[int], int] = -1,
        target_zooms: Optional[Sequence[int]] = None,
        out_zooms: Optional[Sequence[int]] = None,
        groups: Union[Sequence[bool], int] = -1,
        sequence_zooms: Optional[Sequence[Optional[int]]] = None,
        include_neighbors: Optional[Sequence[Optional[bool]]] = None,
        ranks_in: Optional[Sequence[Optional[int]]] = None,
        ranks_out: Optional[Sequence[Optional[int]]] = None,
        include_variable_dependency: Optional[Sequence[bool]] = None,
        include_space_dependency: Optional[Sequence[bool]] = None,
        include_time_dependency: Optional[Sequence[bool]] = None,
        ranks_variable: Optional[Sequence[Optional[int]]] = None,
        ranks_space: Optional[Sequence[Optional[int]]] = None,
        ranks_time: Optional[Sequence[Optional[int]]] = None,
        constraints: Optional[Sequence[ConstraintName]] = None,
        initializations: Optional[Sequence[str]] = None,
        share_factors_across_heads: Optional[Sequence[bool]] = None,
        operator_dim: Optional[int] = None,
        n_times: int = 1,
        token_len_time: Any = 1,
        token_len_depth: Any = 1,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: Any = False,
        token_overlap_mlp_time: bool = False,
        token_overlap_mlp_depth: Any = False,
        operator_projection_rank_time: Any = None,
        operator_projection_rank_space: Any = None,
        operator_projection_rank_depth: Any = None,
        operator_projection_rank_features: Any = None,
        operator_projection_rank_variables: Any = None,
        include_variable_dependency_operator_projection: bool = False,
        mlp_projection_rank_time: Any = None,
        mlp_projection_rank_space: Any = None,
        mlp_projection_rank_depth: Any = None,
        mlp_projection_rank_features: Any = None,
        mlp_projection_rank_variables: Any = None,
        include_variable_dependency_mlp: bool = False,
        update: Literal["shift", "shift_scale"] = "shift",
        layer_norm: bool = True,
        separate_mlp_norm: bool = True,
        mlp_residual_from_operators: bool = False,
        embed_confs: Optional[Dict[str, Any]] = None,
        emb_modulation_mode: str = "shift_scale",
        dropout: Optional[float] = None,
        fac_mode: str = "Tucker",
        n_groups_variables: Optional[Sequence[int]] = None,
        n_groups_depths: Optional[Sequence[int]] = None,
        **kwargs: Any,
    ) -> None:
        removed_chunk_settings = {
            name for name in ("contraction_chunk_size", "chunk_size")
            if name in kwargs
        }
        if removed_chunk_settings:
            names = ", ".join(sorted(removed_chunk_settings))
            raise TypeError(
                f"{names} is internal to operator contraction and is no longer "
                "a FieldSpaceOperatorConfig setting"
            )
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected FieldSpaceOperatorConfig settings: {names}")
        operators = list(operators)
        if not operators:
            raise ValueError("operators must contain at least one atomic operator")
        unsupported = [name for name in operators if name not in _DEPENDENCY_ORDER]
        if unsupported:
            raise ValueError(
                f"Unsupported operators {unsupported}; supported values are "
                f"{list(_DEPENDENCY_ORDER)}"
            )
        n_operators = len(operators)
        num_heads = _aligned_values(num_heads, n_operators, "num_heads", 1)
        sequence_defaults = [-1 if name == "space" else None for name in operators]
        if sequence_zooms is None:
            sequence_zooms = sequence_defaults
        else:
            sequence_zooms = _aligned_values(
                sequence_zooms, n_operators, "sequence_zooms", None
            )
        include_neighbors = _aligned_values(
            include_neighbors, n_operators, "include_neighbors", False
        )
        ranks_in = _aligned_values(ranks_in, n_operators, "ranks_in", None)
        ranks_out = _aligned_values(ranks_out, n_operators, "ranks_out", None)
        include_variable_dependency = _aligned_values(
            include_variable_dependency,
            n_operators,
            "include_variable_dependency",
            False,
        )
        include_time_dependency = _aligned_values(
            include_time_dependency,
            n_operators,
            "include_time_dependency",
            False,
        )
        include_space_dependency = _aligned_values(
            include_space_dependency,
            n_operators,
            "include_space_dependency",
            False,
        )
        ranks_variable = _aligned_values(
            ranks_variable, n_operators, "ranks_variable", None
        )
        ranks_time = _aligned_values(ranks_time, n_operators, "ranks_time", None)
        ranks_space = _aligned_values(ranks_space, n_operators, "ranks_space", None)
        constraints = _aligned_values(
            constraints, n_operators, "constraints", "unconstrained"
        )
        initializations = _aligned_values(
            initializations, n_operators, "initializations", "identity"
        )
        share_factors_across_heads = _aligned_values(
            share_factors_across_heads,
            n_operators,
            "share_factors_across_heads",
            False,
        )

        if update not in {"shift", "shift_scale"}:
            raise ValueError("update must be either 'shift' or 'shift_scale'")
        if int(token_zoom) < -1:
            raise ValueError("token_zoom must be -1 or non-negative")
        if int(n_times) <= 0:
            raise ValueError("n_times must be positive")
        if operator_dim is not None and int(operator_dim) <= 0:
            raise ValueError("operator_dim must be positive when configured")
        if (
            not include_variable_dependency_operator_projection
            and _contains_configured_value(operator_projection_rank_variables)
        ):
            raise ValueError(
                "operator_projection_rank_variables requires "
                "include_variable_dependency_operator_projection=True"
            )
        if (
            not include_variable_dependency_mlp
            and _contains_configured_value(mlp_projection_rank_variables)
        ):
            raise ValueError(
                "mlp_projection_rank_variables requires "
                "include_variable_dependency_mlp=True"
            )

        dependency_flags = {
            "variable": include_variable_dependency,
            "time": include_time_dependency,
            "space": include_space_dependency,
        }
        dependency_ranks = {
            "variable": ranks_variable,
            "time": ranks_time,
            "space": ranks_space,
        }
        for index, operator in enumerate(operators):
            if int(num_heads[index]) <= 0:
                raise ValueError(f"num_heads[{index}] must be positive")
            if operator_dim is not None and int(operator_dim) % int(num_heads[index]):
                raise ValueError(
                    f"operator_dim ({operator_dim}) must be divisible by "
                    f"num_heads[{index}] ({num_heads[index]})"
                )
            if constraints[index] not in {
                "unconstrained", "softmax", "signed_softmax"
            }:
                raise ValueError(
                    f"Unsupported constraints[{index}]={constraints[index]!r}"
                )
            if initializations[index] not in {"identity", "random"}:
                raise ValueError(
                    "Initialization must be 'identity' or 'random'; "
                    f"got initializations[{index}]={initializations[index]!r}"
                )
            for rank_name, rank_values in {
                "ranks_in": ranks_in,
                "ranks_out": ranks_out,
                **{f"ranks_{name}": values for name, values in dependency_ranks.items()},
            }.items():
                rank = rank_values[index]
                if rank is not None and int(rank) <= 0:
                    raise ValueError(f"{rank_name}[{index}] must be positive or None")

            if operator == "space":
                if int(token_zoom) < 0:
                    raise ValueError("A spatial operator requires token_zoom >= 0")
                sequence_zoom = sequence_zooms[index]
                if sequence_zoom is None or int(sequence_zoom) < -1:
                    raise ValueError(
                        f"sequence_zooms[{index}] must be -1 or non-negative for space"
                    )
                sequence_zooms[index] = int(sequence_zoom)
                if sequence_zooms[index] > int(token_zoom):
                    raise ValueError(
                        f"sequence_zooms[{index}] cannot exceed token_zoom"
                    )
                if include_neighbors[index] and sequence_zooms[index] < 0:
                    raise ValueError(
                        "include_neighbors is only valid for a local spatial operator"
                    )
                if (
                    include_space_dependency[index]
                    and sequence_zooms[index] == -1
                ):
                    raise ValueError(
                        "A global spatial operator cannot also use space as a dependency"
                    )
            else:
                if sequence_zooms[index] is not None:
                    raise ValueError(
                        f"sequence_zooms[{index}] is only used by spatial operators"
                    )
                if include_neighbors[index] not in {False, None}:
                    raise ValueError(
                        f"include_neighbors[{index}] is only used by spatial operators"
                    )
                include_neighbors[index] = False

            if operator == "variable" and include_variable_dependency[index]:
                raise ValueError(
                    "A variable operator cannot also use variable as a dependency"
                )
            if operator == "time" and include_time_dependency[index]:
                raise ValueError("A time operator cannot also use time as a dependency")
            if include_space_dependency[index] and int(token_zoom) < 0:
                raise ValueError("A spatial dependency requires token_zoom >= 0")

            for dependency in _DEPENDENCY_ORDER:
                if (
                    not dependency_flags[dependency][index]
                    and dependency_ranks[dependency][index] is not None
                ):
                    raise ValueError(
                        f"ranks_{dependency}[{index}] requires "
                        f"include_{dependency}_dependency[{index}]=True"
                    )

        self.token_zoom = int(token_zoom)
        self.operators = operators
        self.num_heads = [int(value) for value in num_heads]
        self.in_zooms = in_zooms
        self.target_zooms = None if target_zooms is None else list(target_zooms)
        self.out_zooms = None if out_zooms is None else list(out_zooms)
        self.groups = groups
        self.sequence_zooms = list(sequence_zooms)
        self.include_neighbors = [bool(value) for value in include_neighbors]
        self.ranks_in = ranks_in
        self.ranks_out = ranks_out
        self.include_variable_dependency = [
            bool(value) for value in include_variable_dependency
        ]
        self.include_time_dependency = [bool(value) for value in include_time_dependency]
        self.include_space_dependency = [
            bool(value) for value in include_space_dependency
        ]
        self.ranks_variable = ranks_variable
        self.ranks_time = ranks_time
        self.ranks_space = ranks_space
        self.constraints = list(constraints)
        self.initializations = list(initializations)
        self.share_factors_across_heads = [
            bool(value) for value in share_factors_across_heads
        ]
        self.operator_dim = None if operator_dim is None else int(operator_dim)
        self.n_times = int(n_times)
        self.token_len_time = token_len_time
        self.token_len_depth = token_len_depth
        self.token_overlap_space = bool(token_overlap_space)
        self.token_overlap_time = bool(token_overlap_time)
        self.token_overlap_depth = token_overlap_depth
        self.token_overlap_mlp_time = bool(token_overlap_mlp_time)
        self.token_overlap_mlp_depth = token_overlap_mlp_depth
        self.operator_projection_rank_time = operator_projection_rank_time
        self.operator_projection_rank_space = operator_projection_rank_space
        self.operator_projection_rank_depth = operator_projection_rank_depth
        self.operator_projection_rank_features = (
            operator_projection_rank_features
        )
        self.operator_projection_rank_variables = (
            operator_projection_rank_variables
        )
        self.include_variable_dependency_operator_projection = bool(
            include_variable_dependency_operator_projection
        )
        self.mlp_projection_rank_time = mlp_projection_rank_time
        self.mlp_projection_rank_space = mlp_projection_rank_space
        self.mlp_projection_rank_depth = mlp_projection_rank_depth
        self.mlp_projection_rank_features = mlp_projection_rank_features
        self.mlp_projection_rank_variables = mlp_projection_rank_variables
        self.include_variable_dependency_mlp = bool(
            include_variable_dependency_mlp
        )
        self.update = update
        self.layer_norm = bool(layer_norm)
        self.separate_mlp_norm = bool(separate_mlp_norm)
        self.mlp_residual_from_operators = bool(mlp_residual_from_operators)
        if embed_confs is not None:
            self.embed_confs = embed_confs
        self.emb_modulation_mode = emb_modulation_mode
        if dropout is not None:
            self.dropout = float(dropout)
        self.fac_mode = fac_mode
        self.n_groups_variables = n_groups_variables
        self.n_groups_depths = n_groups_depths


class _FieldSpaceOperatorBranch(nn.Module):
    """Projection, operator, projection, and physical update for one operator."""

    def __init__(
        self,
        *,
        grid_layers: Mapping[str, GridLayer],
        in_zooms: Sequence[int],
        target_zooms: Sequence[int],
        in_features: Mapping[int, int],
        token_zoom: int,
        token_len_time: Mapping[int, int],
        token_len_depth: int,
        token_overlap_space: bool,
        token_overlap_time: bool,
        token_overlap_depth: bool,
        operator_dim: int,
        operator: OperatorName,
        num_heads: int,
        n_variables: int,
        n_times: int,
        sequence_zoom: Optional[int],
        include_neighbors: bool,
        include_dependencies: Mapping[str, bool],
        dependency_ranks: Mapping[str, Optional[int]],
        rank_in: Optional[int],
        rank_out: Optional[int],
        constraint: ConstraintName,
        initialization: str,
        share_factors_across_heads: bool,
        operator_projection_ranks_by_zoom: Mapping[
            int, Sequence[Optional[int]]
        ],
        operator_projection_rank_variables_by_zoom: Mapping[
            int, Optional[int]
        ],
        include_variable_dependency_operator_projection: bool,
        update: str,
        dropout: float,
        layer_norm: bool,
        embed_confs: Mapping[str, Any],
        embedder: Optional[nn.Module],
        embedder_cache_key: Optional[str],
        emb_modulation_mode: str,
        fac_mode: str,
    ) -> None:
        super().__init__()
        self.in_zooms = [int(zoom) for zoom in in_zooms]
        self.target_zooms = [int(zoom) for zoom in target_zooms]
        self.token_zoom = int(token_zoom)
        self.token_len_depth = int(token_len_depth)
        self.token_len_time = {int(k): int(v) for k, v in token_len_time.items()}
        self.token_overlap_time = bool(token_overlap_time)
        self.token_overlap_depth = bool(token_overlap_depth)
        self.operator_dim = int(operator_dim)
        self.scale_shift = update == "shift_scale"
        self.update_multiplier = 2 if self.scale_shift else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.embedder = embedder
        self.embedding_keys = set(
            embedder.embedders.keys() if embedder is not None else ()
        )

        self.tokenizers = nn.ModuleDict()
        self.update_tokenizers = nn.ModuleDict()
        self.pre_layers = nn.ModuleDict()
        self.value_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        self.gammas = nn.ParameterDict()
        self.residual_gammas = nn.ParameterDict()
        self.update_shapes: Dict[int, List[int]] = {}

        input_zoom_field = int(embed_confs.get("input_zoom", min(self.in_zooms)))
        processing_zooms = list(dict.fromkeys([*self.in_zooms, *self.target_zooms]))
        for zoom in processing_zooms:
            key = str(zoom)
            tokenizer = Tokenizer(
                [zoom],
                token_zoom,
                overlap_thickness=int(token_overlap_space),
                grid_layers=grid_layers,
                token_len_time=self.token_len_time[zoom],
                token_len_depth=self.token_len_depth,
            )
            update_tokenizer = Tokenizer(
                [zoom],
                token_zoom,
                overlap_thickness=0,
                grid_layers=grid_layers,
                token_len_time=self.token_len_time[zoom],
                token_len_depth=self.token_len_depth,
            )
            self.tokenizers[key] = tokenizer
            self.update_tokenizers[key] = update_tokenizer
            n_space = sum(tokenizer.get_features()[0].values())
            n_space_update = sum(update_tokenizer.get_features()[1].values())
            token_shape = [
                self.token_len_time[zoom],
                n_space,
                self.token_len_depth,
                int(in_features[zoom]),
            ]
            update_shape = [
                self.token_len_time[zoom],
                n_space_update,
                self.token_len_depth,
                int(in_features[zoom]),
            ]
            self.update_shapes[zoom] = update_shape

            emb_tokenizer = Tokenizer(
                [input_zoom_field]
                if embedder is not None and embedder.has_space()
                else [],
                token_zoom,
                token_len_time=1,
                token_len_depth=(
                    self.token_len_depth
                    if embedder is not None and embedder.has_depth()
                    else 1
                ),
                overlap_thickness=int(embed_confs.get("token_overlap_space", False)),
                grid_layers=grid_layers,
            )
            emb_shape = list(token_shape)
            emb_shape[1] = (
                token_shape[1]
                if embedder is not None and embedder.has_space()
                else 1
            )
            operator_projection_rank_variables = (
                operator_projection_rank_variables_by_zoom[zoom]
            )
            projection_n_variables = (
                n_variables
                if include_variable_dependency_operator_projection
                else 1
            )
            self.pre_layers[key] = LinEmbLayer(
                emb_shape,
                emb_shape,
                ranks=list(operator_projection_ranks_by_zoom[zoom]),
                n_variables=1,
                fac_mode=fac_mode,
                identity_if_equal=True,
                embedder=embedder,
                field_tokenizer=emb_tokenizer,
                output_zoom=zoom,
                layer_norm=layer_norm,
                emb_modulation_mode=emb_modulation_mode,
                embedder_cache_key=embedder_cache_key,
            )
            if zoom in self.in_zooms:
                input_shape = [
                    token_shape[0] + 2 * int(self.token_overlap_time),
                    token_shape[1],
                    token_shape[2] + 2 * int(self.token_overlap_depth),
                    token_shape[3],
                ]
                self.value_layers[key] = get_layer(
                    input_shape,
                    [1, 1, 1, self.operator_dim],
                    ranks=list(operator_projection_ranks_by_zoom[zoom]),
                    n_variables=projection_n_variables,
                    rank_variables=operator_projection_rank_variables,
                    fac_mode=fac_mode,
                    bias=False,
                )
            if zoom in self.target_zooms:
                output_shape = [
                    *update_shape[:-1],
                    update_shape[-1] * self.update_multiplier,
                ]
                self.output_layers[key] = get_layer(
                    [1, 1, 1, self.operator_dim],
                    output_shape,
                    ranks=list(operator_projection_ranks_by_zoom[zoom]),
                    n_variables=projection_n_variables,
                    rank_variables=operator_projection_rank_variables,
                    fac_mode=fac_mode,
                    bias=False,
                )
                self.gammas[key] = nn.Parameter(torch.ones(update_shape) * 1e-12)
                self.residual_gammas[key] = nn.Parameter(
                    torch.ones(update_shape) * 1e-12
                )

        self.operator = FieldSpaceOperator(
            operator=operator,
            operator_dim=self.operator_dim,
            num_heads=num_heads,
            n_variables=n_variables,
            n_times=n_times,
            token_zoom=token_zoom,
            sequence_zoom=sequence_zoom,
            include_neighbors=include_neighbors,
            include_dependencies=include_dependencies,
            dependency_ranks=dependency_ranks,
            rank_in=rank_in,
            rank_out=rank_out,
            constraint=constraint,
            initialization=initialization,
            share_factors_across_heads=share_factors_across_heads,
            grid_layers=grid_layers,
        )

    def _aligned_embedding(
        self,
        emb: Optional[Dict[str, Any]],
        zoom: int,
        tokens: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        return align_time_embeddings_to_tokens(
            emb,
            zoom=zoom,
            token_len_time=self.token_len_time[zoom],
            field_time_steps=int(tokens.shape[2] * self.token_len_time[zoom]),
            embedding_keys=self.embedding_keys,
        )

    def _project_values(
        self,
        x_zooms: Mapping[int, torch.Tensor],
        emb: Optional[Dict[str, Any]],
        sample_configs: Mapping[int, Mapping[str, Any]],
    ) -> torch.Tensor:
        projections: List[torch.Tensor] = []
        for zoom in self.in_zooms:
            key = str(zoom)
            tokens = self.tokenizers[key]({zoom: x_zooms[zoom]}, sample_configs)
            aligned_emb = self._aligned_embedding(emb, zoom, tokens)
            tokens = self.pre_layers[key](
                tokens, emb=aligned_emb, sample_configs=sample_configs
            )
            if self.token_overlap_time:
                tokens = add_time_overlap_from_neighbor_patches(
                    tokens, overlap=1, pad_mode="edge"
                )
            if self.token_overlap_depth:
                tokens = add_depth_overlap_from_neighbor_patches(
                    tokens, overlap=1, pad_mode="edge"
                )
            projected = self.value_layers[key](
                tokens, emb=aligned_emb, sample_configs=sample_configs
            )
            projections.append(
                projected.reshape(*projected.shape[:5], self.operator_dim)
            )
        expected = projections[0].shape
        if any(projection.shape != expected for projection in projections[1:]):
            shapes = [tuple(projection.shape) for projection in projections]
            raise ValueError(
                f"Operator value projections must have identical shapes, got {shapes}"
            )
        return torch.stack(projections).sum(dim=0)

    @staticmethod
    def _detokenize(tokens: torch.Tensor) -> torch.Tensor:
        return rearrange(
            tokens,
            "b v T N D t n d f -> b v (T t) (N n) (D d) f",
        )

    def forward(
        self,
        x_zooms: Dict[int, torch.Tensor],
        *,
        emb: Optional[Dict[str, Any]],
        sample_configs: Mapping[int, Mapping[str, Any]],
    ) -> Dict[int, torch.Tensor]:
        values = self._project_values(x_zooms, emb, sample_configs)
        sample_config = sample_configs.get(self.token_zoom, {})
        values = self.operator(
            values, emb=emb, sample_config=sample_config
        )
        operator_tokens = values.view(*values.shape[:5], 1, 1, 1, self.operator_dim)
        for zoom in self.target_zooms:
            key = str(zoom)
            base = self.update_tokenizers[key](
                {zoom: x_zooms[zoom]}, sample_configs
            )
            aligned_emb = self._aligned_embedding(emb, zoom, base)
            projected = self.dropout(
                self.output_layers[key](
                    operator_tokens,
                    emb=aligned_emb,
                    sample_configs=sample_configs,
                )
            )
            gamma = self.gammas[key]
            gamma_res = self.residual_gammas[key]
            if self.scale_shift:
                scale, shift = projected.chunk(2, dim=-1)
                if scale.shape != base.shape:
                    raise ValueError(
                        f"Operator update for zoom {zoom} has shape "
                        f"{tuple(scale.shape)}, expected {tuple(base.shape)}"
                    )
                updated = base * (1 + gamma_res * scale) + gamma * shift
            else:
                if projected.shape != base.shape:
                    raise ValueError(
                        f"Operator update for zoom {zoom} has shape "
                        f"{tuple(projected.shape)}, expected {tuple(base.shape)}"
                    )
                updated = (1 + gamma_res) * base + gamma * projected
            x_zooms[zoom] = self._detokenize(updated)
        return x_zooms


class FieldSpaceOperatorBlock(nn.Module):
    """Run an ordered operator sequence followed by the existing FST MLP pattern."""

    def __init__(
        self,
        *,
        grid_layers: Mapping[str, GridLayer],
        in_zooms: Sequence[int],
        target_zooms: Sequence[int],
        in_features: Union[int, Sequence[int], Mapping[int, int]],
        token_zoom: int,
        operators: Sequence[OperatorName],
        num_heads: Sequence[int],
        sequence_zooms: Sequence[Optional[int]],
        include_neighbors: Sequence[bool],
        ranks_in: Sequence[Optional[int]],
        ranks_out: Sequence[Optional[int]],
        include_variable_dependency: Sequence[bool],
        include_time_dependency: Sequence[bool],
        include_space_dependency: Sequence[bool],
        ranks_variable: Sequence[Optional[int]],
        ranks_time: Sequence[Optional[int]],
        ranks_space: Sequence[Optional[int]],
        constraints: Sequence[ConstraintName],
        initializations: Sequence[str],
        share_factors_across_heads: Optional[Sequence[bool]] = None,
        operator_dim: int,
        n_variables: int,
        n_times: int,
        token_len_time: Any = 1,
        token_len_depth: int = 1,
        token_overlap_space: bool = False,
        token_overlap_time: bool = False,
        token_overlap_depth: bool = False,
        token_overlap_mlp_time: bool = False,
        token_overlap_mlp_depth: bool = False,
        operator_projection_rank_time: Any = None,
        operator_projection_rank_space: Any = None,
        operator_projection_rank_depth: Any = None,
        operator_projection_rank_features: Any = None,
        operator_projection_rank_variables: Any = None,
        include_variable_dependency_operator_projection: bool = False,
        mlp_projection_rank_time: Any = None,
        mlp_projection_rank_space: Any = None,
        mlp_projection_rank_depth: Any = None,
        mlp_projection_rank_features: Any = None,
        mlp_projection_rank_variables: Any = None,
        include_variable_dependency_mlp: bool = False,
        update: str = "shift",
        dropout: float = 0.0,
        layer_norm: bool = True,
        separate_mlp_norm: bool = True,
        mlp_residual_from_operators: bool = False,
        embed_confs: Optional[Mapping[str, Any]] = None,
        embedder: Optional[nn.Module] = None,
        embedder_cache_key: Optional[str] = None,
        emb_modulation_mode: str = "shift_scale",
        fac_mode: str = "Tucker",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected FieldSpaceOperatorBlock settings: {names}")
        self.in_zooms = [int(zoom) for zoom in in_zooms]
        self.target_zooms = [int(zoom) for zoom in target_zooms]
        self.token_zoom = int(token_zoom)
        if min([*self.in_zooms, *self.target_zooms]) < self.token_zoom:
            raise ValueError("All input and target zooms must be >= token_zoom")
        self.in_features = {
            int(key): int(value)
            for key, value in _axis_values(
                in_features, self.in_zooms, "in_features"
            ).items()
        }
        missing_target_features = [
            zoom for zoom in self.target_zooms if zoom not in self.in_features
        ]
        if missing_target_features:
            raise ValueError(
                f"Target zooms {missing_target_features} are not present in in_zooms"
            )
        all_zooms = list(dict.fromkeys([*self.in_zooms, *self.target_zooms]))
        self.token_len_time = {
            int(key): int(value)
            for key, value in _axis_values(
                token_len_time, all_zooms, "token_len_time"
            ).items()
        }
        if any(value <= 0 for value in self.token_len_time.values()):
            raise ValueError("token_len_time values must be positive")
        self.token_len_depth = int(token_len_depth)
        if self.token_len_depth <= 0:
            raise ValueError("token_len_depth must be positive")
        self.operator_dim = int(operator_dim)
        self.scale_shift = update == "shift_scale"
        self.update_multiplier = 2 if self.scale_shift else 1
        self.separate_mlp_norm = bool(separate_mlp_norm)
        self.mlp_residual_from_operators = bool(mlp_residual_from_operators)
        self.token_overlap_mlp_time = bool(token_overlap_mlp_time)
        self.token_overlap_mlp_depth = bool(token_overlap_mlp_depth)
        self.dropout_mlp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.mlp_activation = nn.SiLU()
        if share_factors_across_heads is None:
            share_factors_across_heads = [False] * len(operators)
        elif len(share_factors_across_heads) != len(operators):
            raise ValueError(
                "share_factors_across_heads must align with operators"
            )
        embed_confs = {} if embed_confs is None else dict(embed_confs)
        self.embedder = embedder
        self.embedding_keys = set(
            embedder.embedders.keys() if embedder is not None else ()
        )

        operator_projection_rank_time_by_zoom = _axis_values(
            operator_projection_rank_time,
            all_zooms,
            "operator_projection_rank_time",
        )
        operator_projection_rank_space_by_zoom = _axis_values(
            operator_projection_rank_space,
            all_zooms,
            "operator_projection_rank_space",
        )
        operator_projection_rank_depth_by_zoom = _axis_values(
            operator_projection_rank_depth,
            all_zooms,
            "operator_projection_rank_depth",
        )
        operator_projection_rank_features_by_zoom = _axis_values(
            operator_projection_rank_features,
            all_zooms,
            "operator_projection_rank_features",
        )
        operator_projection_rank_variables_by_zoom = _axis_values(
            operator_projection_rank_variables,
            all_zooms,
            "operator_projection_rank_variables",
        )
        if (
            not include_variable_dependency_operator_projection
            and any(
                rank is not None
                for rank in operator_projection_rank_variables_by_zoom.values()
            )
        ):
            raise ValueError(
                "operator_projection_rank_variables requires "
                "include_variable_dependency_operator_projection=True"
            )
        operator_projection_ranks_by_zoom = {
            zoom: [
                operator_projection_rank_time_by_zoom[zoom],
                operator_projection_rank_space_by_zoom[zoom],
                operator_projection_rank_depth_by_zoom[zoom],
                operator_projection_rank_features_by_zoom[zoom],
            ]
            for zoom in all_zooms
        }

        mlp_projection_rank_time_by_zoom = _axis_values(
            mlp_projection_rank_time,
            all_zooms,
            "mlp_projection_rank_time",
        )
        mlp_projection_rank_space_by_zoom = _axis_values(
            mlp_projection_rank_space,
            all_zooms,
            "mlp_projection_rank_space",
        )
        mlp_projection_rank_depth_by_zoom = _axis_values(
            mlp_projection_rank_depth,
            all_zooms,
            "mlp_projection_rank_depth",
        )
        mlp_projection_rank_features_by_zoom = _axis_values(
            mlp_projection_rank_features,
            all_zooms,
            "mlp_projection_rank_features",
        )
        mlp_projection_rank_variables_by_zoom = _axis_values(
            mlp_projection_rank_variables,
            all_zooms,
            "mlp_projection_rank_variables",
        )
        if (
            not include_variable_dependency_mlp
            and any(
                rank is not None
                for rank in mlp_projection_rank_variables_by_zoom.values()
            )
        ):
            raise ValueError(
                "mlp_projection_rank_variables requires "
                "include_variable_dependency_mlp=True"
            )
        mlp_projection_ranks_by_zoom = {
            zoom: [
                mlp_projection_rank_time_by_zoom[zoom],
                mlp_projection_rank_space_by_zoom[zoom],
                mlp_projection_rank_depth_by_zoom[zoom],
                mlp_projection_rank_features_by_zoom[zoom],
            ]
            for zoom in all_zooms
        }

        self.branches = nn.ModuleList()
        for index, operator in enumerate(operators):
            self.branches.append(
                _FieldSpaceOperatorBranch(
                    grid_layers=grid_layers,
                    in_zooms=self.in_zooms,
                    target_zooms=self.target_zooms,
                    in_features=self.in_features,
                    token_zoom=self.token_zoom,
                    token_len_time=self.token_len_time,
                    token_len_depth=self.token_len_depth,
                    token_overlap_space=token_overlap_space,
                    token_overlap_time=token_overlap_time,
                    token_overlap_depth=token_overlap_depth,
                    operator_dim=self.operator_dim,
                    operator=operator,
                    num_heads=int(num_heads[index]),
                    n_variables=int(n_variables),
                    n_times=int(n_times),
                    sequence_zoom=sequence_zooms[index],
                    include_neighbors=bool(include_neighbors[index]),
                    include_dependencies={
                        "variable": bool(include_variable_dependency[index]),
                        "time": bool(include_time_dependency[index]),
                        "space": bool(include_space_dependency[index]),
                    },
                    dependency_ranks={
                        "variable": ranks_variable[index],
                        "time": ranks_time[index],
                        "space": ranks_space[index],
                    },
                    rank_in=ranks_in[index],
                    rank_out=ranks_out[index],
                    constraint=constraints[index],
                    initialization=initializations[index],
                    share_factors_across_heads=bool(
                        share_factors_across_heads[index]
                    ),
                    operator_projection_ranks_by_zoom=(
                        operator_projection_ranks_by_zoom
                    ),
                    operator_projection_rank_variables_by_zoom=(
                        operator_projection_rank_variables_by_zoom
                    ),
                    include_variable_dependency_operator_projection=bool(
                        include_variable_dependency_operator_projection
                    ),
                    update=update,
                    dropout=dropout,
                    layer_norm=layer_norm,
                    embed_confs=embed_confs,
                    embedder=embedder,
                    embedder_cache_key=embedder_cache_key,
                    emb_modulation_mode=emb_modulation_mode,
                    fac_mode=fac_mode,
                )
            )

        self.mlp_tokenizers = nn.ModuleDict()
        self.mlp_update_tokenizers = nn.ModuleDict()
        self.mlp_pre_layers = nn.ModuleDict()
        self.mlp_projection_layers = nn.ModuleDict()
        self.mlp_output_layers = nn.ModuleDict()
        self.mlp_gammas = nn.ParameterDict()
        self.mlp_residual_gammas = nn.ParameterDict()
        input_zoom_field = int(embed_confs.get("input_zoom", min(self.in_zooms)))
        for zoom in self.target_zooms:
            key = str(zoom)
            tokenizer = Tokenizer(
                [zoom],
                self.token_zoom,
                overlap_thickness=int(token_overlap_space),
                grid_layers=grid_layers,
                token_len_time=self.token_len_time[zoom],
                token_len_depth=self.token_len_depth,
            )
            update_tokenizer = Tokenizer(
                [zoom],
                self.token_zoom,
                overlap_thickness=0,
                grid_layers=grid_layers,
                token_len_time=self.token_len_time[zoom],
                token_len_depth=self.token_len_depth,
            )
            self.mlp_tokenizers[key] = tokenizer
            self.mlp_update_tokenizers[key] = update_tokenizer
            token_shape = [
                self.token_len_time[zoom],
                sum(tokenizer.get_features()[0].values()),
                self.token_len_depth,
                self.in_features[zoom],
            ]
            update_shape = [
                self.token_len_time[zoom],
                sum(update_tokenizer.get_features()[1].values()),
                self.token_len_depth,
                self.in_features[zoom],
            ]
            emb_tokenizer = Tokenizer(
                [input_zoom_field]
                if embedder is not None and embedder.has_space()
                else [],
                self.token_zoom,
                token_len_time=1,
                token_len_depth=(
                    self.token_len_depth
                    if embedder is not None and embedder.has_depth()
                    else 1
                ),
                overlap_thickness=int(embed_confs.get("token_overlap_space", False)),
                grid_layers=grid_layers,
            )
            emb_shape = list(token_shape)
            emb_shape[1] = (
                token_shape[1]
                if embedder is not None and embedder.has_space()
                else 1
            )
            mlp_projection_rank_variables_zoom = (
                mlp_projection_rank_variables_by_zoom[zoom]
            )
            projection_n_variables = (
                int(n_variables)
                if include_variable_dependency_mlp
                else 1
            )
            self.mlp_pre_layers[key] = LinEmbLayer(
                emb_shape,
                emb_shape,
                ranks=list(mlp_projection_ranks_by_zoom[zoom]),
                n_variables=1,
                fac_mode=fac_mode,
                identity_if_equal=True,
                embedder=embedder,
                field_tokenizer=emb_tokenizer,
                output_zoom=zoom,
                layer_norm=layer_norm if separate_mlp_norm else False,
                emb_modulation_mode=emb_modulation_mode,
                embedder_cache_key=embedder_cache_key,
            )
            mlp_input_shape = [
                token_shape[0] + 2 * int(token_overlap_mlp_time),
                token_shape[1],
                token_shape[2] + 2 * int(token_overlap_mlp_depth),
                token_shape[3],
            ]
            self.mlp_projection_layers[key] = get_layer(
                mlp_input_shape,
                [1, 1, 1, self.operator_dim],
                ranks=list(mlp_projection_ranks_by_zoom[zoom]),
                n_variables=projection_n_variables,
                rank_variables=mlp_projection_rank_variables_zoom,
                fac_mode=fac_mode,
                bias=False,
            )
            self.mlp_output_layers[key] = get_layer(
                [1, 1, 1, self.operator_dim],
                [*update_shape[:-1], update_shape[-1] * self.update_multiplier],
                ranks=list(mlp_projection_ranks_by_zoom[zoom]),
                n_variables=projection_n_variables,
                rank_variables=mlp_projection_rank_variables_zoom,
                fac_mode=fac_mode,
                bias=False,
            )
            self.mlp_gammas[key] = nn.Parameter(torch.ones(update_shape) * 1e-12)
            self.mlp_residual_gammas[key] = nn.Parameter(
                torch.ones(update_shape) * 1e-12
            )

        self.mlp_layer1 = get_layer(
            [1, 1, 1, self.operator_dim],
            [1, 1, 1, self.operator_dim],
            ranks=[None] * 4,
            n_variables=1,
            fac_mode=fac_mode,
            bias=False,
        )
        self.mlp_layer2 = get_layer(
            [1, 1, 1, self.operator_dim],
            [1, 1, 1, self.operator_dim],
            ranks=[None] * 4,
            n_variables=1,
            fac_mode=fac_mode,
            bias=False,
        )

    def _aligned_embedding(
        self,
        emb: Optional[Dict[str, Any]],
        zoom: int,
        tokens: torch.Tensor,
    ) -> Optional[Dict[str, Any]]:
        return align_time_embeddings_to_tokens(
            emb,
            zoom=zoom,
            token_len_time=self.token_len_time[zoom],
            field_time_steps=int(tokens.shape[2] * self.token_len_time[zoom]),
            embedding_keys=self.embedding_keys,
        )

    def _run_mlp(
        self,
        x_zooms: Dict[int, torch.Tensor],
        entry_fields: Mapping[int, torch.Tensor],
        emb: Optional[Dict[str, Any]],
        sample_configs: Mapping[int, Mapping[str, Any]],
    ) -> Dict[int, torch.Tensor]:
        projections: List[torch.Tensor] = []
        embeddings: Dict[int, Optional[Dict[str, Any]]] = {}
        for zoom in self.target_zooms:
            key = str(zoom)
            tokens = self.mlp_tokenizers[key]({zoom: x_zooms[zoom]}, sample_configs)
            aligned_emb = self._aligned_embedding(emb, zoom, tokens)
            embeddings[zoom] = aligned_emb
            tokens = self.mlp_pre_layers[key](
                tokens, emb=aligned_emb, sample_configs=sample_configs
            )
            if self.token_overlap_mlp_time:
                tokens = add_time_overlap_from_neighbor_patches(
                    tokens, overlap=1, pad_mode="edge"
                )
            if self.token_overlap_mlp_depth:
                tokens = add_depth_overlap_from_neighbor_patches(
                    tokens, overlap=1, pad_mode="edge"
                )
            projections.append(
                self.mlp_projection_layers[key](
                    tokens, emb=aligned_emb, sample_configs=sample_configs
                )
            )
        expected = projections[0].shape
        if any(projection.shape != expected for projection in projections[1:]):
            raise ValueError("MLP zoom projections must have identical shapes")
        hidden = torch.stack(projections).sum(dim=0)
        shared_emb = embeddings[self.target_zooms[0]]
        hidden = self.mlp_layer1(
            hidden, emb=shared_emb, sample_configs=sample_configs
        )
        hidden = self.mlp_activation(hidden)
        hidden = self.dropout_mlp(hidden)
        hidden = self.mlp_layer2(
            hidden, emb=shared_emb, sample_configs=sample_configs
        )

        for zoom in self.target_zooms:
            key = str(zoom)
            residual_field = (
                x_zooms[zoom]
                if self.mlp_residual_from_operators
                else entry_fields[zoom]
            )
            base = self.mlp_update_tokenizers[key](
                {zoom: residual_field}, sample_configs
            )
            projected = self.dropout_mlp(
                self.mlp_output_layers[key](
                    hidden,
                    emb=embeddings[zoom],
                    sample_configs=sample_configs,
                )
            )
            gamma = self.mlp_gammas[key]
            gamma_res = self.mlp_residual_gammas[key]
            if self.scale_shift:
                scale, shift = projected.chunk(2, dim=-1)
                updated = base * (1 + gamma_res * scale) + gamma * shift
            else:
                updated = (1 + gamma_res) * base + gamma * projected
            x_zooms[zoom] = _FieldSpaceOperatorBranch._detokenize(updated)
        return x_zooms

    def forward(
        self,
        x_zooms: Dict[int, torch.Tensor],
        *,
        emb: Optional[Dict[str, Any]] = None,
        sample_configs: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[int, torch.Tensor]:
        del kwargs
        sample_configs = {} if sample_configs is None else sample_configs
        entry_fields = {zoom: x_zooms[zoom] for zoom in self.target_zooms}
        for branch in self.branches:
            x_zooms = branch(
                x_zooms, emb=emb, sample_configs=sample_configs
            )
        return self._run_mlp(
            x_zooms, entry_fields, emb, sample_configs
        )


class FieldSpaceOperatorModule(nn.Module):
    """Apply independent field-space operator blocks to variable groups."""

    def __init__(
        self,
        *,
        grid_layers: Mapping[str, GridLayer],
        in_zooms: Sequence[int],
        out_zooms: Sequence[int],
        target_zooms: Optional[Sequence[int]],
        in_features: Union[int, Sequence[int], Mapping[int, int]],
        output_features: Optional[Sequence[int]] = None,
        n_groups_variables: Sequence[int],
        n_groups_depths: Optional[Sequence[int]] = None,
        groups: Union[Sequence[bool], int] = -1,
        token_len_depth: Any = 1,
        token_overlap_depth: Any = False,
        token_overlap_mlp_depth: Any = False,
        operator_projection_rank_time: Any = None,
        operator_projection_rank_space: Any = None,
        operator_projection_rank_depth: Any = None,
        operator_projection_rank_features: Any = None,
        operator_projection_rank_variables: Any = None,
        include_variable_dependency_operator_projection: bool = False,
        mlp_projection_rank_time: Any = None,
        mlp_projection_rank_space: Any = None,
        mlp_projection_rank_depth: Any = None,
        mlp_projection_rank_features: Any = None,
        mlp_projection_rank_variables: Any = None,
        include_variable_dependency_mlp: bool = False,
        embed_confs: Optional[Mapping[str, Any]] = None,
        global_embedders: Optional[nn.ModuleDict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if (
            not include_variable_dependency_operator_projection
            and _contains_configured_value(operator_projection_rank_variables)
        ):
            raise ValueError(
                "operator_projection_rank_variables requires "
                "include_variable_dependency_operator_projection=True"
            )
        if (
            not include_variable_dependency_mlp
            and _contains_configured_value(mlp_projection_rank_variables)
        ):
            raise ValueError(
                "mlp_projection_rank_variables requires "
                "include_variable_dependency_mlp=True"
            )
        self.in_zooms = [int(zoom) for zoom in in_zooms]
        self.target_zooms = [
            int(zoom) for zoom in (
                self.in_zooms if target_zooms is None else target_zooms
            )
        ]
        self.out_zooms = [int(zoom) for zoom in out_zooms]
        features_by_zoom = {
            int(zoom): int(value)
            for zoom, value in _axis_values(
                in_features, self.in_zooms, "in_features"
            ).items()
        }
        if output_features is None:
            missing_output_features = [
                zoom for zoom in self.out_zooms if zoom not in features_by_zoom
            ]
            if missing_output_features:
                raise ValueError(
                    "output_features are required when out_zooms include zooms "
                    f"outside operator in_zooms: {missing_output_features}"
                )
            self.out_features = [features_by_zoom[zoom] for zoom in self.out_zooms]
        else:
            if len(output_features) != len(self.out_zooms):
                raise ValueError(
                    "output_features must align with out_zooms; got "
                    f"{len(output_features)} values for {len(self.out_zooms)} zooms"
                )
            self.out_features = [int(value) for value in output_features]
        self.n_groups_variables = [int(value) for value in n_groups_variables]
        n_groups = len(self.n_groups_variables)
        if groups == -1:
            self.active_groups = [True] * n_groups
        elif _is_sequence(groups) and len(groups) == n_groups:
            self.active_groups = [bool(value) for value in groups]
        else:
            raise ValueError(
                f"groups must be -1 or a boolean list of length {n_groups}"
            )
        n_groups_depths = _group_values(
            1 if n_groups_depths is None else n_groups_depths,
            n_groups,
            "n_groups_depths",
        )
        self.n_groups_depths = [int(value) for value in n_groups_depths]
        if any(value <= 0 for value in self.n_groups_depths):
            raise ValueError("n_groups_depths must contain positive values")
        token_len_depth = _group_values(
            token_len_depth, n_groups, "token_len_depth"
        )
        token_overlap_depth = _group_values(
            token_overlap_depth, n_groups, "token_overlap_depth"
        )
        token_overlap_mlp_depth = _group_values(
            token_overlap_mlp_depth, n_groups, "token_overlap_mlp_depth"
        )
        operator_projection_rank_depth = _group_values(
            operator_projection_rank_depth,
            n_groups,
            "operator_projection_rank_depth",
        )
        mlp_projection_rank_depth = _group_values(
            mlp_projection_rank_depth,
            n_groups,
            "mlp_projection_rank_depth",
        )

        embed_confs = {} if embed_confs is None else dict(embed_confs)
        input_zoom_field = int(embed_confs.get("input_zoom", min(self.in_zooms)))
        zoom_key = str(input_zoom_field)
        embedder_cache_key = None
        if not embed_confs.get("embed_names"):
            shared_embedder = None
        elif global_embedders is not None and zoom_key in global_embedders:
            shared_embedder = global_embedders[zoom_key]
            embedder_cache_key = zoom_key
        else:
            shared_embedder = get_embedder(
                **embed_confs,
                grid_layers=grid_layers,
                zoom=input_zoom_field,
            )

        self.blocks = nn.ModuleList()
        self.block_group_indices: List[int] = []
        for group_index, active in enumerate(self.active_groups):
            if not active:
                continue
            block_kwargs = dict(kwargs)
            block_kwargs.update(
                token_len_depth=int(token_len_depth[group_index]),
                token_overlap_depth=bool(token_overlap_depth[group_index]),
                token_overlap_mlp_depth=bool(
                    token_overlap_mlp_depth[group_index]
                ),
                operator_projection_rank_time=operator_projection_rank_time,
                operator_projection_rank_space=operator_projection_rank_space,
                operator_projection_rank_depth=(
                    operator_projection_rank_depth[group_index]
                ),
                operator_projection_rank_features=(
                    operator_projection_rank_features
                ),
                operator_projection_rank_variables=(
                    operator_projection_rank_variables
                ),
                include_variable_dependency_operator_projection=(
                    include_variable_dependency_operator_projection
                ),
                mlp_projection_rank_time=mlp_projection_rank_time,
                mlp_projection_rank_space=mlp_projection_rank_space,
                mlp_projection_rank_depth=(
                    mlp_projection_rank_depth[group_index]
                ),
                mlp_projection_rank_features=mlp_projection_rank_features,
                mlp_projection_rank_variables=mlp_projection_rank_variables,
                include_variable_dependency_mlp=include_variable_dependency_mlp,
            )
            self.blocks.append(
                FieldSpaceOperatorBlock(
                    grid_layers=grid_layers,
                    in_zooms=self.in_zooms,
                    target_zooms=self.target_zooms,
                    in_features=in_features,
                    n_variables=self.n_groups_variables[group_index],
                    embed_confs=embed_confs,
                    embedder=shared_embedder,
                    embedder_cache_key=embedder_cache_key,
                    **block_kwargs,
                )
            )
            self.block_group_indices.append(group_index)

    def forward(
        self,
        x_zooms_groups: List[Dict[int, torch.Tensor]],
        emb_groups: Optional[List[Optional[Dict[str, Any]]]] = None,
        mask_groups: Optional[List[Dict[int, torch.Tensor]]] = None,
        sample_configs: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[Dict[int, torch.Tensor]]:
        del mask_groups, kwargs
        if emb_groups is None:
            emb_groups = [None] * len(x_zooms_groups)
        sample_configs = {} if sample_configs is None else sample_configs
        for block, group_index in zip(self.blocks, self.block_group_indices):
            x_zooms_groups[group_index] = block(
                x_zooms_groups[group_index],
                emb=emb_groups[group_index],
                sample_configs=sample_configs,
            )
        for group_index, zooms in enumerate(x_zooms_groups):
            x_zooms_groups[group_index] = {
                zoom: zooms[zoom] for zoom in self.out_zooms
            }
        return x_zooms_groups
