import os
import random

import numpy as np
import torch
import xarray as xr
import zarr
from torch.utils.data import Dataset, Sampler
import healpy as hp

from .normalizer import DataNormalizer


def file_loadchecker(filename, data_type):
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
    data = ds1[data_type].values

    dims = ds1[data_type].dims
    coords = {key: ds1[data_type].coords[key] for key in ds1[data_type].coords if key != "time"}
    ds1 = ds1.drop_vars(ds1.keys())
    ds1 = ds1.drop_dims("time")

    return [ds, ds1, dims, coords], data, data.shape[1:]


def load_netcdf(data_paths, data_types, keep_dss=False):
    if data_paths is None:
        return None, None
    else:
        ndata = len(data_paths)
        assert ndata == len(data_types)
        dss, data, sizes = zip(*[file_loadchecker(data_paths[i], data_types[i]) for i in range(ndata)])

        if keep_dss:
            return dss[0], data, sizes
        else:
            return data, sizes


class ClimateDataset(Dataset):
    def __init__(self, data_in_dirs, data_out_dirs, data_stats=None, lazy_load=True, normalizer=None):
        if data_stats and isinstance(data_stats, str):
            data_stats = torch.load(data_stats)
        self.data_stats = data_stats if data_stats else {}
        self.lazy_load = lazy_load

        self.climate_in_data, self.in_sizes = load_netcdf(data_in_dirs)
        self.climate_out_data, self.out_sizes = load_netcdf(data_out_dirs)

        self.normalizer = normalizer(self.climate_in_data)

    def init_loader(self, data_stats, normalization, norm_quantile):
        # create data normalizers
        self.normalizer = DataNormalizer(normalization, len(self.climate_data), data_stats, norm_quantile)

    def __getitem__(self, index):
        in_data = self.load_data(index, self.climate_in_data)
        gt_data = self.load_data(index, self.climate_out_data)

        return in_data, gt_data

    def load_data(self, index, dataset):
        data_sample = []
        for i in range(len(dataset)):
            data = dataset[i][index]
            data = torch.from_numpy(np.nan_to_num(data))

            if self.normalizer:
                data = self.normalizer.normalize(data, i)
            data_sample.append(data)
        return torch.stack(data_sample)

    def __len__(self):
        return sum(self.img_lengths)


class NetCDFLoader(Dataloader):
    def __init__(self, data_dir_dict, prediction_mode="abs", seq_length=1, normalization=None, norm_quantile=1.0,
                 data_stats=None, timerange=None, mask_mode="random", conditioning=None, add_mask_to_input=False,
                 strict=False, lazy_load=True):
        super(NetCDFLoader, self).__init__(data_dir_dict, prediction_mode, seq_length, normalization, data_stats,
                                           timerange, mask_mode, conditioning, add_mask_to_input, strict, lazy_load)

        for d, datatype in enumerate(data_dir_dict[list(data_dir_dict.keys())[0]]["gt"].keys()):
            gt_data = []
            for dataset_name, dataset in data_dir_dict.items():
                self.n_ensembles = len(dataset["gt"][datatype])
                for i, datapath in enumerate(dataset["gt"][datatype]):
                    if not self.xr_dss:
                        xr_dss, data, self.img_sizes = load_netcdf([datapath], [datatype],
                                                                   keep_dss=True)
                        self.xr_dss.append(xr_dss)
                    else:
                        data, self.img_sizes = load_netcdf([datapath], [datatype])
                    data = np.concatenate(data)
                    if d == 0:
                        self.img_lengths.append((data.shape[0] if not self.timerange else self.timerange[1] - self.timerange[0]) - self.seq_length + 1)
                    gt_data.append(data)
            self.img_data.append(gt_data)
            self.data_types.append(datatype)

        if "cond" in dataset.keys() and self.conditioning is not None:
            for datatype in data_dir_dict[list(data_dir_dict.keys())[0]]["cond"].keys():
                cond_data = []
                for dataset_name, dataset in data_dir_dict.items():
                    for i, datapath in enumerate(dataset["cond"][datatype]):
                        data, _ = load_netcdf([datapath], [datatype])
                        data = np.concatenate(data)
                        cond_data.append(data)
                    self.cond_data_length = len(cond_data)
                self.cond_data.append(cond_data)

        if "mask" in dataset.keys():
            for datatype in data_dir_dict[list(data_dir_dict.keys())[0]]["mask"].keys():
                mask_data = []
                for dataset_name, dataset in data_dir_dict.items():
                    for i, datapath in enumerate(dataset["mask"][datatype]):
                        data, _ = load_netcdf([datapath], [datatype])
                        data = np.concatenate(data)
                        self.mask_lengths.append(len(data) - self.seq_length + 1)
                        mask_data.append(data)
                self.mask.append(mask_data)

        self.init_loader(self.data_stats, normalization, norm_quantile)

    def get_out_channels(self):
        return len(self.data_types)


class ZarrLoader(Dataloader):
    def __init__(self, data_dir_dict, prediction_mode="abs", seq_length=1, normalization=None, norm_quantile=1.0,
                 data_stats=None, timerange=None, mask_mode="random", conditioning=None, add_mask_to_input=False,
                 strict=False, lazy_load=True):
        super(ZarrLoader, self).__init__(data_dir_dict, prediction_mode, seq_length, normalization, data_stats,
                                           timerange, mask_mode, conditioning, add_mask_to_input, strict, lazy_load)
        self.nside = 0
        for d, datatype in enumerate(data_dir_dict[list(data_dir_dict.keys())[0]]["gt"].keys()):
            gt_data = []
            for dataset_name, dataset in data_dir_dict.items():
                self.n_ensembles = len(dataset["gt"][datatype])
                for i, datapath in enumerate(dataset["gt"][datatype]):
                    if self.lazy_load:
                        store = zarr.DirectoryStore(datapath)
                        cache_store = zarr.LRUStoreCache(store, max_size=128 * 2 ** 30) # 128 GB
                        data = zarr.open(cache_store, 'r')[datatype]
                    else:
                        data = zarr.open(datapath, 'r')[datatype][:]
                    self.nside = max(hp.npix2nside(data.shape[1]), self.nside)
                    if d == 0:
                        self.img_lengths.append((data.shape[0] if not self.timerange else self.timerange[1] - self.timerange[0]) - self.seq_length + 1)
                    gt_data.append(data)
            self.img_data.append(gt_data)
            self.data_types.append(datatype)
        if "cond" in dataset.keys() and self.conditioning is not None:
            for datatype in data_dir_dict[list(data_dir_dict.keys())[0]]["cond"].keys():
                cond_data = []
                for dataset_name, dataset in data_dir_dict.items():
                    for i, datapath in enumerate(dataset["cond"][datatype]):
                        if self.lazy_load:
                            data = zarr.open(datapath, 'r')[datatype]
                        else:
                            data = zarr.open(datapath, 'r')[datatype][:]
                        cond_data.append(data)
                    self.cond_data_length = len(cond_data)
                self.cond_data.append(cond_data)

        if "mask" in dataset.keys():
            for datatype in data_dir_dict[list(data_dir_dict.keys())[0]]["mask"].keys():
                mask_data = []
                for dataset_name, dataset in data_dir_dict.items():
                    for i, datapath in enumerate(dataset["mask"][datatype]):
                        data = zarr.open(datapath, 'r')[datatype]
                        self.mask_lengths.append(data.shape[0] - self.seq_length + 1)
                        mask_data.append(data)
                self.mask.append(mask_data)

        self.init_loader(self.data_stats, normalization, norm_quantile)

    def transform_data(self, data):
        data = np.array([hp.reorder(data[t], n2r=True) for t in range(data.shape[0])])
        if hp.npix2nside(data.shape[-1]) != self.nside:
            theta_out, phi_out = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
            data = np.array([hp.get_interp_val(data[t], theta_out, phi_out, nest=False) for t in range(data.shape[0])]).astype(np.float32)
        return data

    def get_out_channels(self):
        return len(self.data_types)


class TensorLoader(Dataloader):
    def __init__(self, data_dir_dict, prediction_mode="abs", seq_length=1, normalization=None, norm_quantile=1.0,
                 data_stats=None, timerange=None, mask_mode="random", conditioning=None, add_mask_to_input=False,
                 strict=False, lazy_load=True):
        super(TensorLoader, self).__init__(data_dir_dict, prediction_mode, seq_length, normalization, data_stats,
                                           timerange, mask_mode, conditioning, add_mask_to_input, strict, lazy_load)
        self.channels = None
        self.orig_img_sizes = None

        for dataset_name, dataset in data_dir_dict.items():
            self.n_ensembles = len(dataset["gt"])
            for i, datapath in enumerate(dataset["gt"]):
                data = torch.load(datapath)
                data = data.detach().numpy()
                self.channels = data.shape[0]
                self.img_sizes = (data.shape[2:],)
                self.img_lengths.append((data.shape[1] if not self.timerange else self.timerange[1] - self.timerange[0]) - self.seq_length + 1)
                self.img_data.append(data)
        self.img_data = np.array(self.img_data)
        self.img_data = np.moveaxis(self.img_data, 1, 0)
        if "cond" in dataset.keys() and self.conditioning is not None:
            self.cond_data_length = len(dataset["cond"])
            for dataset_name, dataset in data_dir_dict.items():
                for i, datapath in enumerate(dataset["cond"]):
                    data = torch.load(datapath)
                    data = data.detach().numpy()
                    self.cond_data.append(data)
            self.cond_data = np.array(self.cond_data)
            self.cond_data = np.moveaxis(self.cond_data, 1, 0)

        self.init_loader(self.data_stats, normalization, norm_quantile)

    def __len__(self):
        return sum(self.img_lengths)

    def get_out_channels(self):
        return self.channels
