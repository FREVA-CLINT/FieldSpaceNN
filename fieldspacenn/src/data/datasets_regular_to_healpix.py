import copy
import os
from typing import Any, Dict, Mapping, Optional

import healpy as hp
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from .datasets_base import BaseDataset
from ..modules.grids.grid_utils import regular_to_healpix_nearest_mapping


class RegularToHealPixLoader(BaseDataset):
    def __init__(
        self,
        data_dict: Mapping[str, Any],
        sampling_zooms: Mapping[int, Mapping[str, Any]],
        mapping_file: Optional[str] = None,
        snapshot_dir: Optional[str] = None,
        sampling_zooms_collate: Optional[Mapping[int, Mapping[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize a regular-to-HealPix dataset loader and build per-zoom patch indices.

        :param data_dict: Dataset configuration including source/target file paths.
        :param sampling_zooms: Sampling configuration keyed by zoom level.
        :param mapping_file: Optional path to a saved mapping file.
        :param snapshot_dir: Directory used to store an auto-generated mapping file.
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

        self.mapping_file: Optional[str] = mapping_file
        self.snapshot_dir: Optional[str] = snapshot_dir
        self._mapping_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        # Build patch indices per zoom from the HealPix pixelization scheme.
        self.indices: Dict[int, np.ndarray] = {}
        for zoom, sampling in self.sampling_zooms.items():
            npix = hp.nside2npix(2**zoom)

            if sampling["zoom_patch_sample"] == -1:
                self.indices[zoom] = np.arange(npix).reshape(1, -1)
            else:
                n = 4 ** (zoom - sampling["zoom_patch_sample"])
                self.indices[zoom] = np.arange(npix).reshape(-1, n)

        super().__init__(mapping_fcn=self._get_or_create_mapping, **kwargs)

    def _resolve_mapping_path(self, zoom: int) -> str:
        """
        Resolve the mapping file path for a given zoom level.

        :param zoom: HealPix zoom level used in the mapping.
        :return: Absolute path to the mapping file.
        """
        if self.mapping_file is not None:
            return os.path.abspath(self.mapping_file)

        base_dir = self.snapshot_dir if self.snapshot_dir is not None else os.getcwd()
        filename = f"regular_to_healpix_mapping_zoom_{zoom}.npz"
        return os.path.abspath(os.path.join(base_dir, filename))

    def _load_mapping_from_file(self, mapping_path: str) -> Dict[str, torch.Tensor]:
        """
        Load a regular-to-HealPix mapping dictionary from disk.

        :param mapping_path: Path to a saved mapping ``.npz`` file.
        :return: Mapping dictionary with tensor entries.
        """
        with np.load(mapping_path) as loaded:
            indices = loaded["indices"]
            distances = loaded["distances"]
            resolution = loaded["resolution"]

            mapping = {
                "indices": torch.from_numpy(indices.astype(np.int64)).view(-1, 1),
                "distances": torch.from_numpy(distances.astype(np.float32)).view(-1, 1),
                "resolution": torch.tensor(float(np.asarray(resolution).reshape(-1)[0]), dtype=torch.float32),
            }

            if "regular_to_healpix_indices" in loaded.files:
                mapping["regular_to_healpix_indices"] = torch.from_numpy(
                    loaded["regular_to_healpix_indices"].astype(np.int64)
                )

        return mapping

    def _save_mapping_to_file(
        self,
        mapping_path: str,
        mapping: Mapping[str, torch.Tensor],
        zoom: int,
    ) -> None:
        """
        Save a regular-to-HealPix mapping dictionary to disk.

        :param mapping_path: Output path for the mapping ``.npz`` file.
        :param mapping: Mapping dictionary with tensor entries.
        :param zoom: HealPix zoom level used in the mapping.
        :return: None.
        """
        os.makedirs(os.path.dirname(mapping_path) or ".", exist_ok=True)

        payload = {
            "zoom": np.int32(zoom),
            "indices": mapping["indices"].detach().cpu().numpy().astype(np.int64),
            "distances": mapping["distances"].detach().cpu().numpy().astype(np.float32),
            "resolution": np.float32(mapping["resolution"].detach().cpu().item()),
        }

        if "regular_to_healpix_indices" in mapping:
            payload["regular_to_healpix_indices"] = (
                mapping["regular_to_healpix_indices"].detach().cpu().numpy().astype(np.int64)
            )

        np.savez(mapping_path, **payload)

    def _get_or_create_mapping(self, coords: torch.Tensor, zoom: int) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Load an existing mapping or compute and persist a new one.

        :param coords: Regular-grid coordinates tensor of shape ``(n, 2)`` in radians.
        :param zoom: HealPix zoom level.
        :return: Dictionary keyed by zoom level containing the mapping dictionary.
        """
        if zoom in self._mapping_cache:
            return {zoom: self._mapping_cache[zoom]}

        mapping_path = self._resolve_mapping_path(zoom)

        if os.path.isfile(mapping_path):
            mapping = self._load_mapping_from_file(mapping_path)
        else:
            mapping = regular_to_healpix_nearest_mapping(coords[:, 0], coords[:, 1], zoom)
            self._save_mapping_to_file(mapping_path, mapping, zoom)

        self._mapping_cache[zoom] = mapping
        return {zoom: mapping}

    def get_indices_from_patch_idx(self, zoom: int, patch_idx: int) -> np.ndarray:
        """
        Get the pixel indices corresponding to a patch index at a given zoom.

        :param zoom: Zoom level whose patch grid is being queried.
        :param patch_idx: Patch index within the sampled patch grid.
        :return: Column vector of pixel indices for the selected patch with shape ``(n,)``.
        """
        return self.indices[zoom][patch_idx].reshape(-1)
