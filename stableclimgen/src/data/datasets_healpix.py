import math
from typing import List

import copy
import numpy as np
import xarray as xr

from .datasets_base import BaseDataset
from ..modules.grids.grid_utils import hierarchical_zoom_distance_map
import healpy as hp
from omegaconf import DictConfig, ListConfig, OmegaConf

class HealPixLoader(BaseDataset):
    def __init__(self, 
                 data_dict,
                 sampling_zooms,
                 sampling_zooms_collate=None,
                 **kwargs
                 ):
        
        self.data_dict = data_dict
        if isinstance(sampling_zooms, (DictConfig, ListConfig)):
            sampling_zooms = OmegaConf.to_container(sampling_zooms, resolve=True)
        if isinstance(sampling_zooms_collate, (DictConfig, ListConfig)):
            sampling_zooms_collate = OmegaConf.to_container(sampling_zooms_collate, resolve=True)
        self.sampling_zooms = copy.deepcopy(sampling_zooms)
        self.sampling_zooms_collate = copy.deepcopy(sampling_zooms_collate)

        self.indices = {}
        for zoom, sampling in sampling_zooms.items():
            npix = hp.nside2npix(2**zoom)

            if sampling['zoom_patch_sample'] == -1:
                self.indices[zoom] = np.arange(npix).reshape(1,-1)
            else:
                n = 4**(zoom - sampling['zoom_patch_sample'])
                self.indices[zoom] = np.arange(npix).reshape(-1, n)


        super().__init__(mapping_fcn=hierarchical_zoom_distance_map, **kwargs)
    


    def get_indices_from_patch_idx(self, zoom, patch_idx):
        return self.indices[zoom][patch_idx].reshape(-1)
