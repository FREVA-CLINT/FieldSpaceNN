"""Indexed in-memory HealPix dataset loading."""

from dataclasses import dataclass
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Set

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


class InMemoryHealPixLoader(HealPixLoader):
    """HealPix loader backed by indexed, variable-filtered in-memory datasets.

    The loader preserves the sample interface of :class:`HealPixLoader`, but it
    always uses the indexed in-memory backend and always treats one center time
    as one dataset item. DataLoader multiprocessing therefore requires ``fork``
    so workers inherit the process-global cache.
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
        **kwargs: Any,
    ) -> None:
        """Initialize the indexed in-memory loader.

        ``load_n_samples_time`` is accepted for configuration compatibility but
        intentionally forced to one. Use the legacy loader for grouped items.
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

        # BaseDataset creates this for its legacy cache. Do not retain dataset
        # references on instances of the indexed loader.
        del self._in_memory_datasets
        self.load_into_memory = True
        self._in_memory_cache_indices: Dict[str, int] = {}
        self._register_required_datasets()

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

    @staticmethod
    def _print_cache_event(
        action: str,
        cache_index: int,
        entry: _IndexedDatasetCacheEntry,
    ) -> None:
        size_mib = float(entry.dataset.nbytes) / (1024.0 ** 2)
        variables = ",".join(sorted(entry.variables))
        print(
            f"[InMemoryHealPixLoader pid={os.getpid()}] {action} cache[{cache_index}] "
            f"path={entry.path} time_rows={len(entry.original_time_indices)} "
            f"variables={len(entry.variables)} [{variables}] size={size_mib:.2f} MiB",
            flush=True,
        )

    @classmethod
    def _register_cache_entry(
        cls,
        path: str,
        time_indices: Set[Any],
        variables: Set[Any],
    ) -> int:
        requested_times = np.asarray(sorted(int(index) for index in time_indices), dtype=np.int64)
        requested_variables = {str(variable) for variable in variables}

        cache_index = _INDEXED_DATASET_CACHE_BY_PATH.get(path)
        if cache_index is None:
            dataset = cls._load_subset(path, requested_times, requested_variables)
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
            cls._print_cache_event("loaded", cache_index, entry)
            return cache_index

        entry = _INDEXED_DATASET_CACHE[cache_index]
        union_times = np.union1d(entry.original_time_indices, requested_times).astype(np.int64)
        union_variables = entry.variables | requested_variables
        if (
            np.array_equal(union_times, entry.original_time_indices)
            and union_variables == entry.variables
        ):
            cls._print_cache_event("reused", cache_index, entry)
            return cache_index

        dataset = cls._load_subset(path, union_times, union_variables)
        previous_dataset = entry.dataset
        entry.dataset = dataset
        entry.original_time_indices = union_times
        entry.time_positions = {int(value): idx for idx, value in enumerate(union_times)}
        entry.variables = union_variables
        previous_dataset.close()
        cls._print_cache_event("expanded", cache_index, entry)
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
                "The indexed in-memory dataset cache is unavailable. Construct all "
                "datasets before starting workers and use the `fork` multiprocessing "
                "start method."
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
