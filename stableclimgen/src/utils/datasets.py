import os
import random

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

def file_loadchecker(filename, data_type, lazy_load=False):
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.load_dataset(filename, decode_times=True)

    except Exception:
        raise ValueError('Impossible to read {}.'
                         '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(basename))

    ds1 = ds
    ds = ds.drop_vars(data_type)

    if lazy_load:
        data = ds1[data_type]
    else:
        data = ds1[data_type].values

    dims = ds1[data_type].dims
    coords = {key: ds1[data_type].coords[key] for key in ds1[data_type].coords if key != "time"}
    ds1 = ds1.drop_vars(ds1.keys())
    ds1 = ds1.drop_dims("time")

    return data, data.shape[0], [ds, ds1, dims, coords]


def load_data(data_paths, data_type):
    ndata = len(data_paths)
    data, lengths, dss = zip(*[file_loadchecker(data_paths[i], data_type) for i in range(ndata)])

    return data, lengths, dss


class ClimateDataset(Dataset):
    def __init__(self, data_dict, lazy_load=True, normalizer=None, seq_length=1):
        self.lazy_load = lazy_load
        self.seq_length = seq_length
        self.normalizer = normalizer
        self.data_seq_lengths = []

        self.climate_in_data, self.climate_in_dss = [], []
        self.climate_out_data, self.climate_out_dss = [], []

        for i, data_type in enumerate(data_dict["input"].keys()):
            climate_in_data, in_lengths, in_dss = load_data(data_dict["input"][data_type], data_type)
            self.climate_in_data.append(climate_in_data)
            if i == 0:
                self.data_seq_lengths = [l - self.seq_length + 1 for l in in_lengths]
            self.climate_in_dss.append(in_dss)

        for data_type in data_dict["input"].keys():
            climate_in_data, in_sizes, in_dss = load_data(data_dict["input"][data_type], data_type)
            self.climate_out_data.append(climate_in_data)
            self.climate_out_dss.append(in_dss)

    def __getitem__(self, index):
        dataset_index = 0
        current_index = 0
        for length in self.data_seq_lengths:
            if index != 0 and index >= current_index + length:
                current_index += length
                dataset_index += 1
        seq_index = index - current_index

        in_data = self.load_data(dataset_index, seq_index, self.climate_in_data)
        gt_data = self.load_data(dataset_index, seq_index, self.climate_out_data)

        return in_data, gt_data

    def load_data(self, dataset_index, seq_index, dataset):
        data_sample = []
        for i in range(len(dataset)):
            data = dataset[i][dataset_index][seq_index:seq_index+self.seq_length]
            data = torch.from_numpy(np.nan_to_num(data))

            if self.normalizer:
                data = self.normalizer.normalize(data, i)
            data_sample.append(data)
        return torch.stack(data_sample)

    def __len__(self):
        return sum(self.data_seq_lengths)


class MaskClimateDataset(ClimateDataset):
    def __init__(self, mask_mode="prob_1.0", **kwargs):
        super().__init__(**kwargs)
        self.mask_mode = mask_mode

    def __getitem__(self, index):
        in_data, gt_data = super().__getitem__(index)
        mask = torch.zeros_like(gt_data)
        if "prob" in self.mask_mode:
            num_indices = int((1.0 - float(self.mask_mode.split("_")[1])) * self.seq_length)
            indices_to_set = torch.randperm(self.seq_length)[:num_indices]
        elif self.mask_mode == "random":
            # Generate a random number of indices to set to 1
            num_indices = random.randint(1, self.seq_length)
            indices_to_set = torch.randperm(self.seq_length)[:num_indices]
        elif "last" in self.mask_mode:
            predict_steps = int(self.mask_mode.split('_')[1])
            indices_to_set = torch.tensor(range(self.seq_length - predict_steps))
        elif "first" in self.mask_mode:
            predict_steps = int(self.mask_mode.split('_')[1])
            indices_to_set = torch.tensor(range(self.seq_length - predict_steps))
        mask[:, indices_to_set] = 1
        return in_data, gt_data, mask