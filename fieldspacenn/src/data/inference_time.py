"""Carry the forcing calendar through latent and decoded inference archives."""

from pathlib import Path

import numpy as np
import xarray as xr
import zarr


def _monthly_coordinate(store_path):
    with xr.open_zarr(store_path, consolidated=None, chunks=None) as store:
        if "time" not in store.coords:
            raise ValueError(f"{store_path}: no time coordinate")
        time = store.time.load().copy(deep=True)
    # The source may be Zarr v2 while generated archives are Zarr v3. Copy
    # only the CF calendar coding, never source compressor/chunk encodings.
    time.encoding = {key: value for key, value in time.encoding.items()
                     if key in ("units", "calendar")}
    if time.ndim != 1 or time.size == 0:
        raise ValueError(f"{store_path}: expected a nonempty one-dimensional time coordinate")
    try:
        months = np.asarray(time.dt.year * 12 + time.dt.month, dtype=np.int64)
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"{store_path}: time must contain decoded calendar dates") from exc
    if np.any(np.diff(months) != 1):
        raise ValueError(f"{store_path}: time must contain consecutive, ordered monthly dates")
    return time


def inference_time(level_paths, forcing_path=None):
    """Read the full calendar from the inference inputs and check every zoom.

    ``forcing_path`` additionally verifies that the cached loader inputs preserve
    the forcing file's exact dates; this catches stale cached inputs.
    """
    paths = list(level_paths.values())
    if not paths:
        raise ValueError("No inference level stores supplied")
    time = _monthly_coordinate(paths[0])
    for path in paths[1:]:
        other = _monthly_coordinate(path)
        if not np.array_equal(time.values, other.values):
            raise ValueError(f"Inference time differs between {paths[0]} and {path}")
    if forcing_path is not None:
        forcing_time = _monthly_coordinate(forcing_path)
        if not np.array_equal(time.values, forcing_time.values):
            raise ValueError(f"Cached inference time differs from forcing file {forcing_path}")
    return time


def assign_inference_time(dataset, time):
    """Attach original calendar dates; reject truncated or extended predictions."""
    if dataset.sizes.get("time") != time.size:
        raise ValueError(f"Prediction has {dataset.sizes.get('time')} timesteps, "
                         f"but the forcing has {time.size}")
    return dataset.assign_coords(time=time)


def verify_stored_time(store_path, expected_time):
    """Check dates after reopening an encoded or decoded Zarr archive."""
    actual = _monthly_coordinate(store_path)
    if not np.array_equal(actual.values, expected_time.values):
        raise ValueError(f"{store_path}: stored dates differ from the forcing dates")


def restore_stored_time(store_path, expected_time):
    """Repair an existing archive's time coordinate without rewriting fields.

    Only an integer 0..N-1 coordinate is accepted for repair. This prevents
    accidental relabeling of an already dated, possibly different simulation.
    """
    store_path = Path(store_path)
    with xr.open_zarr(store_path, consolidated=None, decode_times=False, chunks=None) as store:
        if "time" not in store.dims:
            raise ValueError(f"{store_path}: no time dimension to repair")
        current = store.time.values.copy()
    if current.size != expected_time.size or not np.array_equal(current, np.arange(current.size)):
        raise ValueError(f"{store_path}: expected an undated 0..N-1 time coordinate "
                         "with the same length as the forcing")
    # Older latent stores have no stored time array at all; xarray exposes a
    # virtual 0..N-1 coordinate for their time dimension. Decoded stores have
    # an integer array that must be removed before replacing its dtype.
    group = zarr.open_group(str(store_path), mode="a")
    had_stored_time = "time" in group
    if had_stored_time:
        del group["time"]
    try:
        xr.Dataset(coords={"time": expected_time}).to_zarr(store_path, mode="a")
        verify_stored_time(store_path, expected_time)
    except Exception:
        if "time" in group:
            del group["time"]
        if had_stored_time:
            xr.Dataset(coords={"time": current}).to_zarr(store_path, mode="a")
        raise
