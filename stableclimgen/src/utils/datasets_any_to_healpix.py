import math

import numpy as np
import xarray as xr

from .datasets_base import BaseDataset

class AnyToHealPixLoader(BaseDataset):
    def __init__(self, 
                 data_dict,
                 zoom_patch_sample,
                 **kwargs
                 ):
        
        self.zoom_patch_sample = zoom_patch_sample

        ds = xr.open_zarr(data_dict["source"]["files"][0])
        npix = len(ds.cell)

        self.zoom = int(math.log(npix // 12, 4)) 

        if zoom_patch_sample == -1:
            n_sample_patches = 1
            self.indices = np.arange(npix).reshape(1,-1)
        else:
            n_sample_patches = npix // 4**(self.zoom - zoom_patch_sample)
            self.indices = np.arange(npix).reshape(-1, 4**(self.zoom - zoom_patch_sample))

        super().__init__(n_sample_patches, data_dict, **kwargs)
    
    def get_indices_from_patch_idx(self, patch_idx):
        return self.indices[patch_idx].reshape(-1,1)