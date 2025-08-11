import math
from typing import List

import numpy as np
import xarray as xr

from .datasets_base import BaseDataset
from ..modules.grids.grid_utils import hierarchical_zoom_distance_map
import healpy as hp

class HealPixLoader(BaseDataset):
    def __init__(self, 
                 data_dict,
                 sampling_zooms,
                 **kwargs
                 ):
        
        self.data_dict = data_dict
        self.sampling_zooms = sampling_zooms

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