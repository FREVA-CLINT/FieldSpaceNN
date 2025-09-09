import xarray as xr

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