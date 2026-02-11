from typing import Any, Dict, Mapping, Optional

import copy
import numpy as np

from .datasets_base import BaseDataset
from ..modules.grids.grid_utils import hierarchical_zoom_distance_map
import healpy as hp
from omegaconf import DictConfig, ListConfig, OmegaConf

class HealPixLoader(BaseDataset):
    def __init__(
        self,
        data_dict: Mapping[str, Any],
        sampling_zooms: Mapping[int, Mapping[str, Any]],
        sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a HealPix dataset loader and build per-zoom patch indices.

        :param data_dict: Dataset configuration including source/target file paths.
        :param sampling_zooms: Sampling configuration keyed by zoom level.
        :param sampling_zooms_collate: Optional collate configuration keyed by zoom level.
        :param kwargs: Additional arguments forwarded to the base dataset initializer.
        :return: None.
        """
        self.data_dict: Mapping[str, Any] = data_dict
        if isinstance(sampling_zooms, (DictConfig, ListConfig)):
            sampling_zooms = OmegaConf.to_container(sampling_zooms, resolve=True)
        if isinstance(sampling_zooms_collate, (DictConfig, ListConfig)):
            sampling_zooms_collate = OmegaConf.to_container(sampling_zooms_collate, resolve=True)
        self.sampling_zooms: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms)
        self.sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = copy.deepcopy(
            sampling_zooms_collate
        )

        # Build patch indices per zoom from the HealPix pixelization scheme.
        self.indices: Dict[int, np.ndarray] = {}
        for zoom, sampling in sampling_zooms.items():
            npix = hp.nside2npix(2**zoom)

            if sampling['zoom_patch_sample'] == -1:
                self.indices[zoom] = np.arange(npix).reshape(1, -1)
            else:
                n = 4**(zoom - sampling['zoom_patch_sample'])
                self.indices[zoom] = np.arange(npix).reshape(-1, n)


        super().__init__(mapping_fcn=hierarchical_zoom_distance_map, **kwargs)
    

    def get_indices_from_patch_idx(self, zoom: int, patch_idx: int) -> np.ndarray:
        """
        Get the pixel indices corresponding to a patch index at a given zoom.

        :param zoom: Zoom level whose patch grid is being queried.
        :param patch_idx: Patch index within the sampled patch grid.
        :return: Column vector of pixel indices for the selected patch with shape ``(n,)``.
        """
        return self.indices[zoom][patch_idx].reshape(-1)
