"""Indexed in-memory HealPix dataset loading."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import xarray as xr

from .datasets_healpix import HealPixLoader


@dataclass
class _IndexedDatasetCacheEntry:
    """One resident file subset addressed by a stable integer handle."""

    path: str
    dataset: xr.Dataset
    original_time_indices: np.ndarray
    time_positions: Dict[int, int]
    variables: Set[str]


_INDEXED_DATASET_CACHE: list[_IndexedDatasetCacheEntry] = []
_INDEXED_DATASET_CACHE_BY_PATH: Dict[str, int] = {}


def _canonical_path(path: str) -> str:
    # Resolve relative components and symlinks so source/target aliases for the
    # same physical file share one cache entry.
    return os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _clear_indexed_dataset_cache() -> None:
    """Clear the process-local cache. Intended for tests and controlled teardown."""
    for entry in _INDEXED_DATASET_CACHE:
        entry.dataset.close()
    _INDEXED_DATASET_CACHE.clear()
    _INDEXED_DATASET_CACHE_BY_PATH.clear()


def _resolve_distributed_context() -> Tuple[int, int]:
    """Resolve an externally launched process's global rank and world size."""
    environment_pairs = (
        ("RANK", "WORLD_SIZE"),
        ("SLURM_PROCID", "SLURM_NTASKS"),
    )
    for rank_name, world_size_name in environment_pairs:
        rank_value = os.environ.get(rank_name)
        world_size_value = os.environ.get(world_size_name)
        if rank_value is None and world_size_value is None:
            continue
        if rank_value is None or world_size_value is None:
            raise RuntimeError(
                "Rank-sharded in-memory loading requires both "
                f"`{rank_name}` and `{world_size_name}` when either is set."
            )

        try:
            rank = int(rank_value)
            world_size = int(world_size_value)
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid distributed environment: {rank_name}={rank_value!r}, "
                f"{world_size_name}={world_size_value!r}."
            ) from exc

        if world_size <= 1:
            raise RuntimeError(
                "`distributed_shard=True` requires an externally launched distributed "
                "job with world size greater than one. On Slurm, use `srun` with more "
                "than one task instead of letting Lightning launch local children."
            )
        if rank < 0 or rank >= world_size:
            raise RuntimeError(
                f"Distributed rank {rank} is outside the valid range [0, {world_size})."
            )
        return rank, world_size

    raise RuntimeError(
        "`distributed_shard=True` requires an externally launched DDP environment. "
        "Set `RANK`/`WORLD_SIZE` (for example with torchrun) or "
        "`SLURM_PROCID`/`SLURM_NTASKS` (with srun)."
    )


def _format_index_runs(indices: Sequence[int], max_runs: int = 8) -> str:
    """Format sorted integer indices as compact inclusive ranges for diagnostics."""
    values = np.unique(np.asarray(indices, dtype=np.int64))
    if values.size == 0:
        return "empty"

    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    runs = np.split(values, breaks)
    formatted = [
        str(int(run[0])) if run.size == 1 else f"{int(run[0])}-{int(run[-1])}"
        for run in runs[:max_runs]
    ]
    if len(runs) > max_runs:
        formatted.append(f"...(+{len(runs) - max_runs} runs)")
    return ",".join(formatted)


class InMemoryHealPixLoader(HealPixLoader):
    """HealPix loader backed by indexed, variable-filtered in-memory datasets.

    The loader preserves the sample interface of :class:`HealPixLoader`, but it
    always uses the indexed in-memory backend and always treats one center time
    as one dataset item. DataLoader multiprocessing therefore requires ``fork``
    so workers inherit their rank's process-global cache. Standard externally
    launched DDP can instead assign a contiguous cache subset to every rank with
    ``distributed_shard=True``.
    """

    requires_fork_workers = True

    def __init__(
        self,
        data_dict: Mapping[str, Any],
        sampling_zooms: Mapping[int, Mapping[str, Any]],
        sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = None,
        sampling_times_emb: Optional[Mapping[str, Any]] = None,
        sampling_zooms_target: Optional[Mapping[int, Mapping[str, Any]]] = None,
        load_into_memory: bool = True,
        load_n_samples_time: int = 1,
        distributed_shard: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the indexed in-memory loader.

        ``load_n_samples_time`` is accepted for configuration compatibility but
        intentionally forced to one. Use the legacy loader for grouped items.
        ``distributed_shard`` requires external DDP ranks (for example Slurm
        tasks launched with ``srun``) and preloads only the local contiguous
        section of the logical dataset.
        """
        if not load_into_memory:
            raise ValueError(
                "`InMemoryHealPixLoader` always loads into memory. Use "
                "`HealPixLoader(load_into_memory=False)` for lazy loading."
            )

        del load_n_samples_time
        super().__init__(
            data_dict=data_dict,
            sampling_zooms=sampling_zooms,
            sampling_zooms_collate=sampling_zooms_collate,
            sampling_times_emb=sampling_times_emb,
            sampling_zooms_target=sampling_zooms_target,
            load_into_memory=False,
            load_n_samples_time=1,
            **kwargs,
        )

        self.distributed_shard = bool(distributed_shard)
        self.distributed_rank = 0
        self.distributed_world_size = 1
        self.global_len_dataset = int(self.len_dataset)
        self.global_item_indices = np.arange(self.global_len_dataset, dtype=np.int64)
        if self.distributed_shard:
            self.distributed_rank, self.distributed_world_size = _resolve_distributed_context()
            self._apply_distributed_shard()

        # BaseDataset creates this for its legacy cache. Do not retain dataset
        # references on instances of the indexed loader.
        del self._in_memory_datasets
        self.load_into_memory = True
        self._in_memory_cache_indices: Dict[str, int] = {}
        self._register_required_datasets()

    @staticmethod
    def _contiguous_shard_indices(
        n_items: int,
        rank: int,
        world_size: int,
    ) -> np.ndarray:
        """Return an equal-length contiguous shard, padding at local boundaries."""
        if n_items <= 0:
            raise ValueError("Cannot distribute an empty dataset across DDP ranks.")
        if world_size <= 0:
            raise ValueError(f"`world_size` must be positive, got {world_size}.")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"Rank {rank} is outside the valid range [0, {world_size}).")

        base_size, remainder = divmod(n_items, world_size)
        local_size = base_size + (1 if rank < remainder else 0)
        start = rank * base_size + min(rank, remainder)
        target_size = (n_items + world_size - 1) // world_size

        if local_size:
            indices = np.arange(start, start + local_size, dtype=np.int64)
            padding_value = int(indices[-1])
        else:
            # When there are fewer items than ranks, otherwise-empty ranks repeat
            # the last global item so every DDP rank still executes one step.
            indices = np.empty(0, dtype=np.int64)
            padding_value = n_items - 1

        if indices.size < target_size:
            padding = np.full(target_size - indices.size, padding_value, dtype=np.int64)
            indices = np.concatenate((indices, padding))
        return indices

    def _apply_distributed_shard(self) -> None:
        """Restrict every zoom-aligned index map to this process's local shard."""
        index_map_lengths = {zoom: len(rows) for zoom, rows in self.index_map.items()}
        if len(set(index_map_lengths.values())) != 1:
            raise RuntimeError(
                "Rank sharding requires zoom-aligned index maps with equal lengths, "
                f"got {index_map_lengths}."
            )

        self.global_len_dataset = next(iter(index_map_lengths.values()), 0)
        self.global_item_indices = self._contiguous_shard_indices(
            self.global_len_dataset,
            self.distributed_rank,
            self.distributed_world_size,
        )
        for zoom, rows in self.index_map.items():
            self.index_map[zoom] = rows[self.global_item_indices]
        self.len_dataset = len(self.global_item_indices)

        reference_rows = self.index_map[self.zooms[0]]
        center_indices = reference_rows[:, 2:].reshape(-1)
        unique_global_items = np.unique(self.global_item_indices)
        padded_items = self.len_dataset - len(unique_global_items)
        print(
            f"[InMemoryHealPixLoader pid={os.getpid()} "
            f"rank={self.distributed_rank}/{self.distributed_world_size}] "
            f"shard global_items={self.global_len_dataset} local_items={self.len_dataset} "
            f"global_item_indices={_format_index_runs(unique_global_items)} "
            f"center_indices={_format_index_runs(center_indices)} padded_items={padded_items}",
            flush=True,
        )

    @staticmethod
    def _expanded_time_indices(
        center_times: np.ndarray,
        n_past_ts: int,
        n_future_ts: int,
        shift: int = 0,
    ) -> np.ndarray:
        centers = np.asarray(center_times, dtype=np.int64).reshape(-1) + int(shift)
        windows = [
            np.arange(
                center - int(n_past_ts),
                center + int(n_future_ts) + 1,
                dtype=np.int64,
            )
            for center in centers
        ]
        if not windows:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(windows)

    def _active_variables(self, zoom: int, target: bool) -> Set[str]:
        variables: Set[str] = set()
        for group, group_variables in self.variables_by_group.items():
            if zoom not in self.group_zooms[group]:
                continue
            if target and group in ("embedding", "embedding_1D"):
                continue
            variables.update(str(variable) for variable in group_variables)
        return variables

    def _required_file_subsets(self) -> Dict[str, Dict[str, Set[Any]]]:
        requirements: Dict[str, Dict[str, Set[Any]]] = {}

        def add(path: str, time_indices: np.ndarray, variables: Set[str]) -> None:
            canonical_path = _canonical_path(path)
            requirement = requirements.setdefault(
                canonical_path,
                {"times": set(), "variables": set()},
            )
            requirement["times"].update(int(index) for index in time_indices)
            requirement["variables"].update(variables)

        for zoom in self.zooms:
            rows = self.index_map[zoom]
            if rows.size == 0:
                continue

            # Patch rows repeat the same file/time entry. Collapse those repeats
            # before expanding temporal windows.
            file_and_times = np.unique(
                rows[:, np.r_[0, np.arange(2, rows.shape[1])]],
                axis=0,
            )
            for row in file_and_times:
                file_index = int(row[0])
                center_times = row[1:].astype(np.int64, copy=False)
                source_path = self._get_file_path("source", zoom, file_index)
                target_path = self._get_file_path("target", zoom, file_index)

                source_indices = self._expanded_time_indices(
                    center_times,
                    self.sampling_zooms[zoom]["n_past_ts"],
                    self.sampling_zooms[zoom]["n_future_ts"],
                )
                embedding_indices = self._expanded_time_indices(
                    center_times,
                    self.sample_configs_emb[zoom]["n_past_ts"],
                    self.sample_configs_emb[zoom]["n_future_ts"],
                )
                target_indices = self._expanded_time_indices(
                    center_times,
                    self.sampling_zooms_target[zoom]["n_past_ts"],
                    self.sampling_zooms_target[zoom]["n_future_ts"],
                    shift=self.target_time_shift,
                )

                add(
                    source_path,
                    np.concatenate((source_indices, embedding_indices)),
                    self._active_variables(zoom, target=False),
                )
                add(
                    target_path,
                    target_indices,
                    self._active_variables(zoom, target=True),
                )

        return requirements

    @staticmethod
    def _load_subset(path: str, time_indices: np.ndarray, variables: Set[str]) -> xr.Dataset:
        with xr.open_dataset(path, decode_times=False) as dataset:
            missing_variables = sorted(variables - set(dataset.data_vars))
            if missing_variables:
                raise KeyError(
                    f"Dataset file `{path}` is missing required variables: {missing_variables}."
                )
            return dataset[sorted(variables)].isel(time=time_indices).load()

    def _cache_log_prefix(self) -> str:
        return (
            f"[InMemoryHealPixLoader pid={os.getpid()} "
            f"rank={self.distributed_rank}/{self.distributed_world_size}]"
        )

    def _print_cache_event(
        self,
        action: str,
        cache_index: int,
        entry: _IndexedDatasetCacheEntry,
    ) -> None:
        size_mib = float(entry.dataset.nbytes) / (1024.0 ** 2)
        variables = ",".join(sorted(entry.variables))
        print(
            f"{self._cache_log_prefix()} {action} cache[{cache_index}] "
            f"path={entry.path} time_rows={len(entry.original_time_indices)} "
            f"time_ranges={_format_index_runs(entry.original_time_indices)} "
            f"variables={len(entry.variables)} [{variables}] size={size_mib:.2f} MiB",
            flush=True,
        )

    def _register_cache_entry(
        self,
        path: str,
        time_indices: Set[Any],
        variables: Set[Any],
    ) -> int:
        requested_times = np.asarray(sorted(int(index) for index in time_indices), dtype=np.int64)
        requested_variables = {str(variable) for variable in variables}

        cache_index = _INDEXED_DATASET_CACHE_BY_PATH.get(path)
        if cache_index is None:
            dataset = self._load_subset(path, requested_times, requested_variables)
            cache_index = len(_INDEXED_DATASET_CACHE)
            _INDEXED_DATASET_CACHE_BY_PATH[path] = cache_index
            entry = _IndexedDatasetCacheEntry(
                path=path,
                dataset=dataset,
                original_time_indices=requested_times,
                time_positions={int(value): idx for idx, value in enumerate(requested_times)},
                variables=requested_variables,
            )
            _INDEXED_DATASET_CACHE.append(entry)
            self._print_cache_event("loaded", cache_index, entry)
            return cache_index

        entry = _INDEXED_DATASET_CACHE[cache_index]
        union_times = np.union1d(entry.original_time_indices, requested_times).astype(np.int64)
        union_variables = entry.variables | requested_variables
        if (
            np.array_equal(union_times, entry.original_time_indices)
            and union_variables == entry.variables
        ):
            self._print_cache_event("reused", cache_index, entry)
            return cache_index

        dataset = self._load_subset(path, union_times, union_variables)
        previous_dataset = entry.dataset
        entry.dataset = dataset
        entry.original_time_indices = union_times
        entry.time_positions = {int(value): idx for idx, value in enumerate(union_times)}
        entry.variables = union_variables
        previous_dataset.close()
        self._print_cache_event("expanded", cache_index, entry)
        return cache_index

    def _register_required_datasets(self) -> None:
        for path, requirement in self._required_file_subsets().items():
            cache_index = self._register_cache_entry(
                path,
                requirement["times"],
                requirement["variables"],
            )
            self._in_memory_cache_indices[path] = cache_index

    def _cache_entry(self, file_path: str) -> _IndexedDatasetCacheEntry:
        canonical_path = _canonical_path(file_path)
        try:
            cache_index = self._in_memory_cache_indices[canonical_path]
            entry = _INDEXED_DATASET_CACHE[cache_index]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                "The indexed in-memory dataset cache is unavailable in this process. "
                "Construct the dataset in every external DDP rank; DataLoader workers "
                "must inherit it with the `fork` start method."
            ) from exc

        if entry.path != canonical_path:
            raise RuntimeError(
                f"In-memory cache handle {cache_index} no longer refers to `{canonical_path}`."
            )
        return entry

    def _translate_time_indices(
        self,
        file_path: str,
        time_indices: Sequence[int],
    ) -> np.ndarray:
        entry = self._cache_entry(file_path)
        try:
            return np.asarray(
                [entry.time_positions[int(index)] for index in time_indices],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise IndexError(
                f"Time index {int(exc.args[0])} was not preloaded for `{entry.path}`."
            ) from exc

    def get_files(
        self,
        file_path_source: str,
        file_path_target: Optional[str] = None,
        drop_source: bool = False,
    ) -> tuple[xr.Dataset, Optional[xr.Dataset]]:
        """Resolve source and target datasets through integer cache handles."""
        source = self._cache_entry(file_path_source).dataset
        if file_path_target is None:
            target = None
        elif _canonical_path(file_path_target) == _canonical_path(file_path_source) and not drop_source:
            target = None
        else:
            target = self._cache_entry(file_path_target).dataset
        return source, target
