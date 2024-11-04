import os
import random
from typing import Tuple, List, Dict, Any, Union, Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


def file_loadchecker(filename: str, data_type: str, lazy_load: bool = False) -> Tuple[Union[np.ndarray, xr.DataArray], int, Dict[str, Any]]:
    """
    Check and load data from a file, handling errors and optionally applying lazy loading.

    :param filename: Path to the file to load.
    :param data_type: Type of data to retrieve from the file.
    :param lazy_load: If True, loads data lazily as an xarray DataArray. Default is False.
    :return: A tuple with the loaded data, length of data, and coordinate information.
    """
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print(f'File {basename} not found.')

    try:
        # Load dataset with error handling
        ds = xr.load_dataset(filename, decode_times=True)
    except Exception:
        raise ValueError(f'Cannot read {basename}. Check if it is a valid netCDF file and is not corrupted.')

    ds1 = ds
    ds = ds.drop_vars(data_type)

    data = ds1[data_type] if lazy_load else ds1[data_type].values
    dims = ds1[data_type].dims
    coords = {key: ds1[data_type].coords[key] for key in ds1[data_type].coords if key != "time"}
    ds1 = ds1.drop_vars(ds1.keys()).drop_dims("time")

    return data, data.shape[0], coords


def load_data(data_paths: List[str], data_type: str) -> Tuple[List[Union[np.ndarray, xr.DataArray]], List[int], List[Dict[str, Any]]]:
    """
    Load multiple datasets based on provided paths.

    :param data_paths: List of file paths to load data from.
    :param data_type: Type of data to retrieve from each file.
    :return: A tuple containing loaded data, lengths, and coordinates for each file.
    """
    ndata = len(data_paths)
    data, lengths, dss = zip(*[file_loadchecker(data_paths[i], data_type) for i in range(ndata)])

    return data, lengths, dss


class ClimateDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing climate data.

    :param data_dict: Dictionary containing input data paths and types.
    :param lazy_load: If True, enables lazy loading of data. Default is True.
    :param normalizer: Optional normalizer for data preprocessing.
    :param seq_length: Length of the data sequences to load.
    """

    def __init__(self, data_dict: Dict[str, Dict[str, List[str]]], lazy_load: bool = True, normalizer: Optional[Any] = None, seq_length: int = 1):
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor]]:
        """
        Retrieve a sequence from the dataset at a given index.

        :param index: Index of the sequence to retrieve.
        :return: A tuple containing input data, input coordinates, ground truth data, and ground truth coordinates.
        """
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

    def load_data(self, dataset_index: int, seq_index: int, dataset: List[Union[np.ndarray, xr.DataArray]], coords: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Load and process a data sequence from a given dataset index.

        :param dataset_index: Index of the dataset to retrieve data from.
        :param seq_index: Index of the sequence within the dataset.
        :param dataset: List of datasets to load data from.
        :param coords: List of coordinate dictionaries for each dataset.
        :return: Processed data sample and coordinates for the given sequence.
        """
        data_sample, data_coords = [], []
        for i in range(len(dataset)):
            data = dataset[i][dataset_index][seq_index:seq_index+self.seq_length]
            data = torch.from_numpy(np.nan_to_num(data))

            if self.normalizer:
                data = self.normalizer.normalize(data, i)
            data_sample.append(data)

            lats = torch.from_numpy(coords[i][dataset_index]["lat"].values)
            lons = torch.from_numpy(coords[i][dataset_index]["lon"].values)
            lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing='ij')
            data_coords.append(torch.stack((lat_grid, lon_grid)))
        return torch.stack(data_sample), torch.stack(data_coords)

    def __len__(self) -> int:
        """
        Return the total number of sequences in the dataset.

        :return: Total sequence count.
        """
        return sum(self.data_seq_lengths)


class MaskClimateDataset(ClimateDataset):
    """
    A subclass of ClimateDataset with an additional masking feature.

    :param mask_mode: Mode for applying masks to sequences. Options include "prob", "random", "last", and "first".
    """

    def __init__(self, mask_mode: str = "prob_1.0", **kwargs):
        super().__init__(**kwargs)
        self.mask_mode = mask_mode

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Retrieve a sequence from the dataset with a mask applied.

        :param index: Index of the sequence to retrieve.
        :return: A tuple containing input data, input coordinates, ground truth data, ground truth coordinates, and mask.
        """
        in_data, in_coords, gt_data, gt_coords = super().__getitem__(index)
        mask = torch.zeros_like(gt_data)
        if "prob" in self.mask_mode:
            num_indices = int((1.0 - float(self.mask_mode.split("_")[1])) * self.seq_length)
            indices_to_set = torch.randperm(self.seq_length)[:num_indices]
        elif self.mask_mode == "random":
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
