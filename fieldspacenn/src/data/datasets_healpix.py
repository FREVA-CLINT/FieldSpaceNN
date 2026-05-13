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
        sampling_zooms: Optional[Mapping[int, Mapping[str, Any]]] = None,
        sampling_zooms_source: Optional[Mapping[int, Mapping[str, Any]]] = None,
        sampling_zooms_target: Optional[Mapping[int, Mapping[str, Any]]] = None,
        sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a HealPix dataset loader and build per-zoom patch indices.

        :param data_dict: Dataset configuration including source/target file paths.
        :param sampling_zooms: Shared sampling configuration keyed by zoom level.
        :param sampling_zooms_source: Optional source sampling configuration keyed by zoom.
        :param sampling_zooms_target: Optional target sampling configuration keyed by zoom.
        :param sampling_zooms_collate: Optional collate configuration keyed by zoom level.
        :param kwargs: Additional arguments forwarded to the base dataset initializer.
        :return: None.
        """
        self.data_dict: Mapping[str, Any] = data_dict
        if sampling_zooms is None:
            if sampling_zooms_source is None and sampling_zooms_target is None:
                raise ValueError(
                    "Expected either 'sampling_zooms' or at least one of "
                    "'sampling_zooms_source'/'sampling_zooms_target'."
                )
            if sampling_zooms_source is None:
                sampling_zooms_source = sampling_zooms_target
            if sampling_zooms_target is None:
                sampling_zooms_target = sampling_zooms_source
        else:
            if sampling_zooms_source is None:
                sampling_zooms_source = sampling_zooms
            if sampling_zooms_target is None:
                sampling_zooms_target = sampling_zooms

        if isinstance(sampling_zooms_source, (DictConfig, ListConfig)):
            sampling_zooms_source = OmegaConf.to_container(sampling_zooms_source, resolve=True)
        if isinstance(sampling_zooms_target, (DictConfig, ListConfig)):
            sampling_zooms_target = OmegaConf.to_container(sampling_zooms_target, resolve=True)
        if isinstance(sampling_zooms, (DictConfig, ListConfig)):
            sampling_zooms = OmegaConf.to_container(sampling_zooms, resolve=True)
        if isinstance(sampling_zooms_collate, (DictConfig, ListConfig)):
            sampling_zooms_collate = OmegaConf.to_container(sampling_zooms_collate, resolve=True)
        self.sampling_zooms_source: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms_source)
        self.sampling_zooms_target: Mapping[int, Mapping[str, Any]] = copy.deepcopy(sampling_zooms_target)
        overlapping_zooms = set(self.sampling_zooms_source.keys()).intersection(
            set(self.sampling_zooms_target.keys())
        )
        for zoom in overlapping_zooms:
            if self.sampling_zooms_source[zoom] != self.sampling_zooms_target[zoom]:
                raise ValueError(
                    f"Zoom {zoom} is defined in both source and target with different "
                    "sampling configs. Use distinct zoom keys or identical configs."
                )

        self.sampling_zooms: Mapping[int, Mapping[str, Any]] = copy.deepcopy(self.sampling_zooms_source)
        for zoom, sampling in self.sampling_zooms_target.items():
            if zoom not in self.sampling_zooms:
                self.sampling_zooms[zoom] = copy.deepcopy(sampling)
        self.sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = copy.deepcopy(
            sampling_zooms_collate
        )

        # Build patch indices per zoom from the HealPix pixelization scheme.
        self.indices: Dict[int, np.ndarray] = {}
        for zoom, sampling in self.sampling_zooms.items():
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
