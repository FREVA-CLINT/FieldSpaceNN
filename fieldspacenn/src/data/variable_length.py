"""Monthly multigrid windows with deterministic, distributed length sampling."""

import copy
import math

import numpy as np
import torch
from torch.utils.data import BatchSampler, SequentialSampler

from .datasets_base import BaseDataset
from .datasets_healpix import HealPixLoader


class VariableLengthHealPixLoader(HealPixLoader):
    """Reuse one loader's mappings/normalizers across aligned window lengths.

    The batch sampler supplies (length, row, seed) indices. Windows start at
    their first month; the conditioning prefix is independent of that anchor.
    Existing fixed-window loaders are unaffected.
    """

    def __init__(self, data_dict, sampling_zooms, sequence_lengths, conditioning=None, **kwargs):
        lengths = sorted(set(int(t) for t in sequence_lengths))
        if not lengths or min(lengths) < 1:
            raise ValueError("sequence_lengths must contain positive lengths")
        conditioning = dict(conditioning or {})
        if conditioning.get("mode", "prefix") != "prefix" or conditioning.get("positive_prefix_sampling", "uniform") != "uniform":
            raise ValueError("Only uniform prefix conditioning is supported")
        self.probability_no_history = float(conditioning.get("probability_no_history", 0.2))
        if not 0 <= self.probability_no_history <= 1:
            raise ValueError("probability_no_history must be between zero and one")
        for key in ("p_dropout", "p_dropout_all", "p_drop_groups", "target_time_shift"):
            if kwargs.get(key, 0) != 0:
                raise ValueError(f"{key} must be zero for prefix windows")
        if kwargs.get("load_n_samples_time", 1) != 1:
            raise ValueError("Variable windows require load_n_samples_time=1")
        for key in ("sampling_zooms_target", "sampling_times_emb", "sampling_zooms_collate"):
            if kwargs.get(key) is not None:
                raise ValueError(f"{key} must be omitted for aligned variable windows")
        configs = {int(z): dict(c) for z, c in sampling_zooms.items()}
        for config in configs.values():
            if config.get("p_drop", 0) or config.get("mask_n_last_ts", 0):
                raise ValueError("Disable whole-zoom and fixed-suffix masking for prefix windows")
            config.update(n_past_ts=0, n_future_ts=lengths[0] - 1)
        kwargs.setdefault("residualize_zooms", False)
        kwargs.setdefault("calendar_time", True)
        kwargs.setdefault("output_binary_mask", True)
        super().__init__(data_dict=data_dict, sampling_zooms=configs, **kwargs)
        self.sequence_lengths = lengths
        views, rows_by_file = {}, {}
        base_rows = self.index_map[self.zooms[0]]
        if not len(base_rows):
            raise ValueError("No simulations can supply the shortest requested window")
        file_lengths = np.asarray(self.time_steps_files)
        for length in lengths:
            valid = base_rows[:, 2] + length <= file_lengths[base_rows[:, 0]]
            if not valid.any():
                raise ValueError(f"No simulation can supply {length} contiguous timesteps")
            view = copy.copy(self)
            view.sampling_zooms = copy.deepcopy(configs)
            for config in view.sampling_zooms.values():
                config["n_future_ts"] = length - 1
            view.sampling_zooms_target = copy.deepcopy(view.sampling_zooms)
            view.sample_configs_emb = copy.deepcopy(view.sampling_zooms)
            view.max_time_step_future = length - 1
            for name in ("zoom_time_steps_future", "zoom_time_steps_future_target", "zoom_time_steps_future_emb"):
                setattr(view, name, [length - 1] * len(self.zooms))
            view.index_map = {z: rows[valid] for z, rows in self.index_map.items()}
            view.len_dataset = int(valid.sum())
            views[length] = view
            rows = view.index_map[self.zooms[0]]
            rows_by_file[length] = [np.flatnonzero(rows[:, 0] == f) for f in np.unique(rows[:, 0])]
        self.length_views = views
        self.rows_by_file = rows_by_file

    def __getitem__(self, index):
        if isinstance(index, tuple):
            length, row, seed = index
        else:
            length, row, seed = self.sequence_lengths[0], int(index), int(index)
        sample = BaseDataset.__getitem__(self.length_views[length], row)
        sources, targets, masks, embeddings, patches = sample
        rng = torch.Generator().manual_seed(seed)
        prefix = 0
        if length > 1 and torch.rand((), generator=rng).item() >= self.probability_no_history:
            prefix = int(torch.randint(1, length, (), generator=rng))
        reference_times = None
        for source, target, mask, emb in zip(sources, targets, masks, embeddings):
            for zoom, values in source.items():
                if values.shape[2] != length or target[zoom].shape != values.shape:
                    raise ValueError("Source and target latent windows must be aligned")
                times = emb["TimeEmbedder"][zoom]
                if reference_times is not None and not torch.equal(times, reference_times):
                    raise ValueError("Climate variables and zooms must have identical timestamps")
                reference_times = times
                generation = torch.ones_like(values, dtype=torch.bool)
                generation[:, :, :prefix] = False
                mask[zoom] = generation
                source[zoom] = torch.where(generation, torch.zeros_like(values), values)
        return sources, targets, masks, embeddings, patches


class _EpochSampler(SequentialSampler):
    """Expose set_epoch through the hook used by Lightning's batch samplers."""

    epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)


class VariableLengthBatchSampler(BatchSampler):
    """Uniform length, then simulation, then valid window; shard global batches.

    All ranks construct the same global batch and take disjoint slots. Sampling
    is with replacement, so even small length buckets have equal-length ranks.
    Validation uses a fixed schedule (shuffle=False), including prefix seeds.
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=42, num_replicas=None, rank=None, batches_per_epoch=None):
        super().__init__(_EpochSampler(dataset), batch_size, drop_last=False)
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.num_replicas = (torch.distributed.get_world_size() if distributed else 1) if num_replicas is None else num_replicas
        self.rank = (torch.distributed.get_rank() if distributed else 0) if rank is None else rank
        if self.num_replicas < 1 or not 0 <= self.rank < self.num_replicas:
            raise ValueError("Invalid distributed sampler rank/world size")
        self.dataset, self.shuffle, self.seed = dataset, shuffle, seed
        self.n_batches = batches_per_epoch if batches_per_epoch is not None else math.ceil(len(dataset) / (batch_size * self.num_replicas))
        if self.n_batches < 1:
            raise ValueError("batches_per_epoch must be positive")

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + (self.sampler.epoch if self.shuffle else 0))
        def draw(n):
            return int(torch.randint(n, (), generator=generator))
        for _ in range(self.n_batches):
            length = self.dataset.sequence_lengths[draw(len(self.dataset.sequence_lengths))]
            files = self.dataset.rows_by_file[length]
            batch = []
            for _ in range(self.batch_size * self.num_replicas):
                rows = files[draw(len(files))]
                batch.append((length, int(rows[draw(len(rows))]), draw(2**31 - 1)))
            start = self.rank * self.batch_size
            yield batch[start:start + self.batch_size]
