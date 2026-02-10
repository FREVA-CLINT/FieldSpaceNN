import math
from typing import Any, Mapping

import numpy as np
import xarray as xr

from .datasets_base import BaseDataset

class AnyToHealPixLoader(BaseDataset):
    def __init__(
        self,
        data_dict: Mapping[str, Any],
        zoom_patch_sample: int,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the HealPix dataset loader and compute patch indices.

        :param data_dict: Dataset configuration including source file paths.
        :param zoom_patch_sample: Target zoom level for patch sampling (-1 for full map).
        :param kwargs: Additional arguments forwarded to the base dataset initializer.
        :return: None.
        """
        self.zoom_patch_sample: int = zoom_patch_sample

        # Load dataset metadata to infer the HealPix resolution.
        ds: xr.Dataset = xr.open_zarr(data_dict["source"]["files"][0])
        npix: int = len(ds.cell)

        # Infer the HealPix zoom level from the number of pixels.
        self.zoom: int = int(math.log(npix // 12, 4))

        if zoom_patch_sample == -1:
            # Use a single patch that spans the full set of pixels.
            n_sample_patches: int = 1
            self.indices: np.ndarray = np.arange(npix).reshape(1, -1)
        else:
            # Split the full resolution into patches at the requested zoom level.
            n_sample_patches: int = npix // 4 ** (self.zoom - zoom_patch_sample)
            self.indices: np.ndarray = np.arange(npix).reshape(
                -1, 4 ** (self.zoom - zoom_patch_sample)
            )

        super().__init__(n_sample_patches, data_dict, **kwargs)

    def get_indices_from_patch_idx(self, patch_idx: int) -> np.ndarray:
        """
        Get the pixel indices corresponding to a patch index.

        :param patch_idx: Patch index within the sampled patch grid.
        :return: Column vector of pixel indices for the selected patch.
        """
        return self.indices[patch_idx].reshape(-1, 1)
