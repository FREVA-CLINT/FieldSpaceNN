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
                 zoom_patch_sample,
                 out_zooms,
                 **kwargs
                 ):
        
        self.zoom_patch_sample = zoom_patch_sample

        npix = hp.nside2npix(2**max(out_zooms))

        self.zoom = int(math.log(npix // 12, 4)) 

        if zoom_patch_sample == -1:
            n_sample_patches = 1
            self.indices = np.arange(npix).reshape(1,-1)
        else:
            n_sample_patches = npix // 4**(self.zoom - zoom_patch_sample)
            self.indices = np.arange(npix).reshape(-1, 4**(self.zoom - zoom_patch_sample))

        super().__init__(n_sample_patches, data_dict, out_zooms=out_zooms, mapping_fcn=hierarchical_zoom_distance_map, **kwargs)
    

    def get_indices_from_patch_idx(self, patch_idx):
        return self.indices[patch_idx].reshape(-1)