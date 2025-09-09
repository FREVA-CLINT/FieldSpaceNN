import xarray as xr

from stableclimgen.src.modules.grids.grid_utils import get_lon_lat_names, remap_healpix_to_any


def export_healpix_to_netcdf(test_dataset, max_zoom, output, mask, run_id, filename):
    output_remapped = {}
    # get mapping
    for grid_type, variables_grid_type in test_dataset.grid_types_vars.items():
        variables = set(variables_grid_type)
        indices = test_dataset.mapping[max_zoom][grid_type]['indices'][..., [0]]
        lat_dim, lon_dim = 36, 72  # TODO: get dimensions from nc dataset

        output_remapped = remap_healpix_to_any(output, variables, indices, lat_dim, lon_dim)
        mask_remapped = remap_healpix_to_any(mask, variables, indices, lat_dim, lon_dim)

    reference_nc_path = test_dataset.data_dict['target'][max_zoom]['files'][0]

    if test_dataset.sample_timesteps:
        timesteps = xr.open_dataset(reference_nc_path, decode_times=False)["time"][test_dataset.sample_timesteps]
    else:
        timesteps = xr.open_dataset(reference_nc_path, decode_times=False)["time"][:]

    if ".nc" in reference_nc_path:
        for grid_type, variables_grid_type in test_dataset.grid_types_vars.items():
            lon, lat = get_lon_lat_names(grid_type)
            dimensions = ('time', lat, lon)
            save_tensor_as_netcdf(
                {key: value for key, value in output_remapped.items() if key in variables_grid_type},
                timesteps,
                filename.replace(".pt", ".nc"),
                dimensions,
                reference_nc_path,
                run_id=run_id)
            save_tensor_as_netcdf(
                {key: value for key, value in mask_remapped.items() if key in variables_grid_type},
                timesteps,
                filename.replace(".pt", "_mask.nc"),
                dimensions,
                reference_nc_path,
                run_id=run_id)

def save_tensor_as_netcdf(data, timesteps, output_path, dims, reference_nc_path=None, run_id=None):
    with xr.open_dataset(reference_nc_path) as ref_ds:
        # Create a new dataset, copying only static variables (those without a 'time' dim)
        vars_to_copy = {
            name: var for name, var in ref_ds.variables.items()
            if 'time' not in var.dims and name not in data.keys()
        }
        output_ds = xr.Dataset(vars_to_copy)
        output_ds.attrs = ref_ds.attrs
        output_ds.attrs["run_id"] = run_id

        # Get coordinates from reference, but replace time with the new timesteps
        new_coords = {name: coord for name, coord in ref_ds.coords.items() if name != 'time'}
        new_coords['time'] = timesteps
        output_ds = output_ds.assign_coords(new_coords)

        # Add the new/updated variables as DataArrays
        for var_name, tensor in data.items():
            da = xr.DataArray(
                data=tensor.float().cpu().view(tuple(len(output_ds[dim]) for dim in dims)).numpy(),
                dims=dims,
                coords={dim: output_ds[dim] for dim in dims}
            )
            if var_name in ref_ds:
                da.attrs = ref_ds[var_name].attrs
            output_ds[var_name] = da
    output_ds.to_netcdf(output_path)