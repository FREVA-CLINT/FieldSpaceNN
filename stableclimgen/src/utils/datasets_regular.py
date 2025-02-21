import copy
import json
import os
import random
from typing import Tuple, List, Dict, Any, Union

import numpy as np
import omegaconf
import torch
import xarray as xr
from attr.validators import instance_of
from torch import Tensor
from torch.utils.data import Dataset

from . import normalizer as normalizers


def file_loadchecker(filename: str, data_type: str, lazy_load: bool = False) -> Tuple[
    Union[np.ndarray, xr.DataArray], int, Dict[str, Any]]:
    """
    Check and load data from a file, handling errors and optionally applying lazy loading.

    :param filename: Path to the file to load.
    :param data_type: Type of data to retrieve from the file.
    :param lazy_load: If True, loads data lazily as an xarray DataArray. Default is False.
    :return: A tuple with the loaded data, length of data, and coordinate information.
    """
    basename = os.path.basename(filename)

    if not os.path.isfile(filename):
        raise FileNotFoundError(f'File {basename} not found.')

    try:
        # Load dataset with error handling for corrupted or incompatible files
        ds = xr.load_dataset(filename, decode_times=True)
    except Exception as e:
        raise ValueError(f'Cannot read {basename}. Ensure it is a valid netCDF file and not corrupted. Error: {e}')

    # Retrieve data, optionally as a lazy-loaded xarray DataArray
    data = ds[data_type] if lazy_load else ds[data_type].values
    coords = {key: ds[data_type].coords[key] for key in ds[data_type].coords if key != "time"}

    return data, data.shape[0], coords


def load_data(data_paths: List[str], data_type: str) -> Tuple[
    List[Union[np.ndarray, xr.DataArray]], List[int], List[Dict[str, Any]]]:
    """
    Load multiple datasets based on provided paths.

    :param data_paths: List of file paths to load data from.
    :param data_type: Type of data to retrieve from each file.
    :return: A tuple containing loaded data, lengths, and coordinates for each file.
    """
    # Load data, lengths, and coordinate dictionaries for each file path
    data, lengths, coords = zip(*[file_loadchecker(path, data_type) for path in data_paths])
    return list(data), list(lengths), list(coords)


class ClimateDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing climate data.

    :param data_dict: Dictionary containing input data paths and types.
    :param norm_dict: Dictionary containing normalizer type and variable specific data stats.
    :param lazy_load: If True, enables lazy loading of data. Default is True.
    :param seq_length: Length of the data sequences to load.
    """

    def __init__(self, data_dict: Dict[str, Dict[str, List[str]]], norm_dict: Dict, lazy_load: bool = True,
                 seq_length: int = 1, n_sample_vars: int = -1, shared_files: bool = False):
        self.lazy_load = lazy_load
        self.seq_length = seq_length

        self.var_normalizers = {}
        self.variables_source = data_dict["source"]["variables"]
        self.variables_target = data_dict["target"]["variables"]
        self.climate_in_files = {}
        self.climate_out_files = {}
        self.n_sample_vars = n_sample_vars

        with open(norm_dict) as json_file:
            norm_dict = json.load(json_file)

        for var in self.variables_source:
            # create normalizers
            norm_class = norm_dict[var]['normalizer']['class']
            assert norm_class in normalizers.__dict__.keys(), f'normalizer class {norm_class} not defined'
            self.var_normalizers[var] = normalizers.__getattribute__(norm_class)(
                norm_dict[var]['stats'],
                norm_dict[var]['normalizer'])

        # Load source data and coordinates
        for i, file in enumerate(data_dict["source"]["files"]):
            if shared_files:
                for var in self.variables_source:
                    if var in self.climate_in_files.keys():
                        if isinstance(file, str):
                            self.climate_in_files[var].append(file)
                        else:
                            self.climate_in_files[var] += file
                    else:
                        self.climate_in_files[var] = [file] if isinstance(file, str) else file
            else:
                var = self.variables_source[i]
                if var in self.climate_in_files.keys():
                    if isinstance(file, str):
                        self.climate_in_files[var].append(file)
                    else:
                        self.climate_in_files[var] += file
                else:
                    self.climate_in_files[var] = [file] if isinstance(file, str) else file

        # Load target data and coordinates
        for i, file in enumerate(data_dict["target"]["files"]):
            if shared_files:
                for var in self.variables_target:
                    if var in self.climate_out_files.keys():
                        if isinstance(file, str):
                            self.climate_out_files[var].append(file)
                        else:
                            self.climate_out_files[var] += file
                    else:
                        self.climate_out_files[var] = [file] if isinstance(file, str) else file
            else:
                var = self.variables_target[i]
                if var in self.climate_out_files.keys():
                    if isinstance(file, str):
                        self.climate_out_files[var].append(file)
                    else:
                        self.climate_out_files[var] += file
                else:
                    self.climate_out_files[var] = [file] if isinstance(file, str) else file
        ds_source = xr.open_dataset(self.climate_in_files[self.variables_source[0]][0], decode_times=False)
        self.data_file_length = ds_source["time"].shape[0] - self.seq_length + 1
        self.len_dataset = self.data_file_length * len(self.climate_in_files[self.variables_source[0]])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, list[str] | Tensor]:
        """
        Retrieve a sequence from the dataset at a given index.

        :param index: Index of the sequence to retrieve.
        :return: A tuple containing input data, input coordinates, ground truth data, and ground truth coordinates.
        """
        file_index = index // self.data_file_length
        seq_index = index % self.data_file_length

        if self.n_sample_vars == -1:
            sample_vars = torch.arange(len(self.variables_source))
        else:
            sample_vars = torch.randperm(len(self.variables_source))[:self.n_sample_vars]

        # Load input and ground truth data for the selected sequence
        in_data, in_coords, out_data, out_coords = self.load_data(file_index, seq_index, sample_vars)

        return in_data.unsqueeze(-1), in_coords, out_data.unsqueeze(-1), out_coords, sample_vars

    def get_files(self, file_path_source, file_path_target=None):

        if self.lazy_load:
            ds_source = xr.open_dataset(file_path_source, decode_times=False)
        else:
            ds_source = xr.load_dataset(file_path_source, decode_times=False)

        if file_path_target is None:
            ds_target = copy.deepcopy(ds_source)

        elif file_path_target == file_path_source:
            ds_target = ds_source

        else:
            if self.lazy_load:
                ds_target = xr.open_dataset(file_path_target, decode_times=False)
            else:
                ds_target = xr.load_dataset(file_path_target, decode_times=False)

        return ds_source, ds_target

    def load_data(self, file_index: int, seq_index: int, sample_vars) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and process a data sequence from a given dataset index.

        :param sample_vars:
        :param file_index: Index of the file to retrieve data from.
        :param seq_index: Index of the sequence within the dataset.
        :param dataset: Dict of datasets to load data from.
        :param coords: Dict of coordinates for each dataset.
        :return: Processed data sample and coordinates for the given sequence.
        """

        data_in, coords_in, data_out, coords_out = [], [], [], []
        for i in sample_vars:
            var = self.variables_source[i.item()]
            # Extract data sequence and convert to torch tensor
            ds_source, ds_target = self.get_files(self.climate_in_files[var][file_index], self.climate_out_files[var][file_index])

            data_src = torch.from_numpy(np.nan_to_num(ds_source[var][seq_index:seq_index + self.seq_length]))
            data_tgt = torch.from_numpy(np.nan_to_num(ds_target[var][seq_index:seq_index + self.seq_length]))

            # Apply normalization if a normalizer is provided
            data_in.append(self.var_normalizers[var].normalize(data_src))
            data_out.append(self.var_normalizers[var].normalize(data_tgt))

            # Load and arrange latitude and longitude coordinates
            lats = torch.from_numpy(ds_source[var].coords["lat"].values)
            lons = torch.from_numpy(ds_source[var].coords["lon"].values)
            lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing='ij')
            coords_in.append(torch.stack((lat_grid, lon_grid), dim=-1).float())

            # Load and arrange latitude and longitude coordinates
            lats = torch.from_numpy(ds_target[var].coords["lat"].values)
            lons = torch.from_numpy(ds_target[var].coords["lon"].values)
            lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing='ij')
            coords_out.append(torch.stack((lat_grid, lon_grid), dim=-1).float())

        return torch.stack(data_in, dim=-1), torch.stack(coords_in, dim=-2), torch.stack(data_out, dim=-1), torch.stack(coords_out, dim=-2)

    def __len__(self) -> int:
        """
        Return the total number of sequences in the dataset.

        :return: Total sequence count.
        """
        return self.len_dataset


class MaskClimateDataset(ClimateDataset):
    """
    A subclass of ClimateDataset with an additional masking feature.

    :param mask_mode: Mode for applying masks to sequences. Options include "prob", "random", "last", and "first".
    """

    def __init__(self, mask_mode: str = "prob_1.0", **kwargs):
        super().__init__(**kwargs)
        self.mask_mode = mask_mode

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[str] | Tensor]:
        """
        Retrieve a sequence from the dataset with a mask applied.

        :param index: Index of the sequence to retrieve.
        :return: A tuple containing input data, input coordinates, ground truth data, ground truth coordinates, and mask.
        """
        in_data, in_coords, gt_data, gt_coords, sample_vars = super().__getitem__(index)
        mask = torch.zeros_like(gt_data, dtype=torch.bool)

        # Define indices to set in mask based on mask mode
        if "prob" in self.mask_mode:
            mask_prob = float(self.mask_mode.split("_")[1])
            num_indices = int((1.0 - mask_prob) * self.seq_length)
            indices_to_set = torch.randperm(self.seq_length)[:num_indices]
        elif self.mask_mode == "random":
            num_indices = random.randint(1, self.seq_length)
            indices_to_set = torch.randperm(self.seq_length)[:num_indices]
        elif "last" in self.mask_mode:
            predict_steps = int(self.mask_mode.split('_')[1])
            indices_to_set = torch.arange(self.seq_length - predict_steps)
        elif "first" in self.mask_mode:
            predict_steps = int(self.mask_mode.split('_')[1])
            indices_to_set = torch.arange(predict_steps)

        # Apply mask based on selected indices
        mask[:, indices_to_set] = True
        return in_data, in_coords, gt_data, gt_coords, mask, sample_vars
