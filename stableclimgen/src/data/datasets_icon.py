import math
from typing import Any, Mapping

import numpy as np
import xarray as xr

from .datasets_base import BaseDataset

class ICONLoader(BaseDataset):
    def __init__(
        self,
        data_dict: Mapping[str, Any],
        zoom_patch_sample: int,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ICON dataset loader and compute patch indices.

        :param data_dict: Dataset configuration including source file paths.
        :param zoom_patch_sample: Target zoom level for patch sampling (-1 for full map).
        :param kwargs: Additional arguments forwarded to the base dataset initializer.
        :return: None.
        """
        self.zoom_patch_sample: int = zoom_patch_sample

        # Normalize source file paths into a list for consistent indexing.
        if isinstance(data_dict["source"]["files"], list):
            data_source = data_dict["source"]["files"]
        else:
            data_source = np.loadtxt(data_dict["source"]["files"], dtype='str', ndmin=1)

        # Load dataset metadata to infer the ICON resolution.
        ds = xr.open_dataset(data_source[0])
        npix = len(ds.cell)

        self.zoom: int = int(math.log(npix // 5, 4)) 

        if zoom_patch_sample == -1:
            n_sample_patches = 1
            self.indices: np.ndarray = np.arange(npix).reshape(1, -1)
        else:
            n_sample_patches = npix // 4**(self.zoom - zoom_patch_sample)
            self.indices: np.ndarray = np.arange(npix).reshape(-1, 4**(self.zoom - zoom_patch_sample))

        super().__init__(n_sample_patches, data_dict, **kwargs)
    
    def get_indices_from_patch_idx(self, patch_idx: int) -> np.ndarray:
        """
        Get the pixel indices corresponding to a patch index.

        :param patch_idx: Patch index within the sampled patch grid.
        :return: Column vector of pixel indices for the selected patch with shape ``(n, 1)``.
        """
        return self.indices[patch_idx].reshape(-1, 1)
