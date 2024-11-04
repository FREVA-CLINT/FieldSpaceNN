import os
import random

import numpy as np
import torch
import xarray as xr
from numba.core.ir_utils import init_copy_propagate_data
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

    return data, data.shape[0], coords


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

        self.climate_in_data, self.in_coords = [], []
        self.climate_out_data, self.out_coords = [], []

        for i, data_type in enumerate(data_dict["input"].keys()):
            climate_in_data, in_lengths, in_coords = load_data(data_dict["input"][data_type], data_type)
            self.climate_in_data.append(climate_in_data)
            if i == 0:
                self.data_seq_lengths = [l - self.seq_length + 1 for l in in_lengths]
            self.in_coords.append(in_coords)

        for data_type in data_dict["input"].keys():
            climate_in_data, in_sizes, out_coords = load_data(data_dict["input"][data_type], data_type)
            self.climate_out_data.append(climate_in_data)
            self.out_coords.append(out_coords)

    def __getitem__(self, index):
        dataset_index = 0
        current_index = 0
        for length in self.data_seq_lengths:
            if index != 0 and index >= current_index + length:
                current_index += length
                dataset_index += 1
        seq_index = index - current_index

        in_data, in_coords = self.load_data(dataset_index, seq_index, self.climate_in_data, self.in_coords)
        gt_data, gt_coords = self.load_data(dataset_index, seq_index, self.climate_out_data, self.out_coords)

        return in_data.unsqueeze(-1), in_coords, gt_data.unsqueeze(-1), gt_coords

    def load_data(self, dataset_index, seq_index, dataset, coords):
        data_sample = []
        data_coords = []
        for i in range(len(dataset)):
            data = dataset[i][dataset_index][seq_index:seq_index+self.seq_length]
            data = torch.from_numpy(np.nan_to_num(data))

            if self.normalizer:
                data = self.normalizer.normalize(data, i)
            data_sample.append(data)

            # get coords
            lats = torch.from_numpy(coords[i][dataset_index]["lat"].values)
            lons = torch.from_numpy(coords[i][dataset_index]["lon"].values)

            lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing='ij')
            data_coords.append(torch.stack((lat_grid, lon_grid)))
        return torch.stack(data_sample), torch.stack(data_coords)

    def __len__(self):
        return sum(self.data_seq_lengths)


class MaskClimateDataset(ClimateDataset):
    def __init__(self, mask_mode="prob_1.0", **kwargs):
        super().__init__(**kwargs)
        self.mask_mode = mask_mode

    def __getitem__(self, index):
        in_data, in_coords, gt_data, gt_coords = super().__getitem__(index)
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
        return in_data, in_coords, gt_data, gt_coords, mask