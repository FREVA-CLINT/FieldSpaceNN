import os
from typing import Any, Dict, Optional

import healpy as hp
import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
from scipy.interpolate import griddata

def healpix_plot_local(
    values: np.ndarray,
    zoom: int,
    ax=None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
    zoom_patch_sample: int = 1,
    patch_index: int = 0,
    **kwargs: Any
):
    """
    Plot a local HEALPix patch by interpolating onto a lat/lon grid.

    :param values: Values for the selected HEALPix patch of shape ``(n,)``.
    :param zoom: Zoom level for the HEALPix grid.
    :param ax: Optional Matplotlib axis for plotting.
    :param vmin: Optional minimum value for color scaling.
    :param vmax: Optional maximum value for color scaling.
    :param title: Plot title.
    :param zoom_patch_sample: Patch sampling zoom.
    :param patch_index: Patch index to select.
    :param kwargs: Additional keyword arguments (unused).
    :return: None.
    """
    # Select pixel indices for the requested patch.
    ipix = np.arange(hp.nside2npix(2**zoom)).reshape(-1,4**(zoom-zoom_patch_sample))[patch_index[0]]
    lon, lat = hp.pix2ang(2**zoom, ipix, nest=True, lonlat=True)

    n_p = max(100, int(np.sqrt(len(lon))))

    # Create grid for interpolation.
    lon_grid = np.linspace(np.min(lon), np.max(lon), n_p)
    lat_grid = np.linspace(np.min(lat), np.max(lat), n_p)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate scattered data onto a regular lat/lon grid.
    grid_values = griddata(
        points=np.stack((lon, lat), axis=-1),
        values=values,
        xi=(lon_mesh, lat_mesh),
        method='nearest',
        fill_value=np.nan
    )

    # Plot using contourf.
    ctf = ax.contourf(lon_mesh, lat_mesh, grid_values,
                        levels=100,
                        vmin=vmin,
                        vmax=vmax,
                        cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(ctf, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)

    tick_locs = np.linspace(ctf.get_clim()[0], ctf.get_clim()[1], 5)
    cbar.set_ticks(tick_locs)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_xticklabels([f"{x:.2f}" for x in tick_locs])

def plot_zooms(
    input_maps: Dict[int, np.ndarray],
    output_maps: Dict[int, np.ndarray],
    gt_maps: Dict[int, np.ndarray],
    mask_maps: Optional[Dict[int, np.ndarray]],
    save_path: str,
    sample_configs: Dict[int, Dict[str, Any]] = {}
):
    """
    Plot HEALPix maps per zoom level for a specific variable and save the result.

    :param input_maps: Input maps per zoom, each of shape ``(n,)``.
    :param output_maps: Output maps per zoom, each of shape ``(n,)``.
    :param gt_maps: Ground truth maps per zoom, each of shape ``(n,)``.
    :param mask_maps: Optional mask maps per zoom, each of shape ``(n,)``.
    :param save_path: Path to save the figure.
    :param sample_configs: Sampling configuration per zoom.
    :return: None.
    """
    zoom_levels = sorted(input_maps.keys())
    n_rows = len(zoom_levels)
    n_cols = 4  # input, output, gt, error
    n_plots = n_rows * n_cols

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    titles = ['Input', 'Output', 'Ground Truth', 'Error']

    for row_idx, zoom in enumerate(zoom_levels):
        sample_configs_zoom = sample_configs[zoom]
        inp_map = input_maps[zoom]
        out_map = output_maps[zoom]
        gt_map = gt_maps[zoom]
        mask_map = mask_maps[zoom] if mask_maps is not None else None

        error_map = (out_map - gt_map) #* mask_map

        maps = [inp_map, out_map, gt_map, error_map]
        gt_min, gt_max = np.quantile(gt_map, [0.001,0.999])
        error_map_min, error_map_max = np.quantile(error_map, [0.001,0.999])
        min_max = [(gt_min, gt_max), (gt_min, gt_max), (gt_min, gt_max), (error_map_min, error_map_max)]

        for col_idx in range(n_cols):
            plot_idx = row_idx * n_cols + col_idx + 1  # 1-based index for subplots

            if sample_configs_zoom['zoom_patch_sample']==-1:
                hp.mollview(maps[col_idx],
                            title=f"{titles[col_idx]} (zoom {zoom})",
                            sub=(n_rows, n_cols, plot_idx),
                            nest=True,
                            min=min_max[col_idx][0],
                            max=min_max[col_idx][1],
                            fig=fig)
            else:
                ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                healpix_plot_local(maps[col_idx], 
                                  zoom, 
                                  ax=ax, 
                                  vmin=min_max[col_idx][0], 
                                  vmax=min_max[col_idx][1], 
                                  title=f"{titles[col_idx]} (zoom {zoom})", 
                                  **sample_configs_zoom)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def healpix_plot_zooms_var(input_zooms: Dict[int, torch.Tensor], 
                           output_zooms: Dict[int, torch.Tensor],
                           gt_zooms: Dict[int, torch.Tensor],
                           save_dir: str, 
                           mask_zooms: Optional[Dict[int, torch.Tensor]] = None,
                           sample_configs: Dict[int, Dict[str, Any]] = {}, 
                           emb: Optional[Dict[str, Any]] = None,
                           plot_name: str = "healpix_plot",
                           sample: int = 0,
                           plot_n_vars: int = -1,
                           plot_n_ts: int = 1):
    """
    Create HEALPix plots for each variable across zoom levels and save them.

    :param input_zooms: Input tensor dict by zoom with shape ``(b, v, t, n, d, f)``.
    :param output_zooms: Output tensor dict by zoom with shape ``(b, v, t, n, d, f)``.
    :param gt_zooms: Ground truth tensor dict by zoom with shape ``(b, v, t, n, d, f)``.
    :param save_dir: Directory to save plots.
    :param mask_zooms: Optional mask tensor dict by zoom with shape ``(b, v, t, n, d, m)``.
    :param sample_configs: Sampling configuration per zoom.
    :param emb: Optional embedding dictionary for variable names.
    :param plot_name: Base name for plot files.
    :param sample: Sample index to plot.
    :param plot_n_vars: Number of variables to plot (-1 for all).
    :param plot_n_ts: Number of timesteps to plot.
    :return: List of saved file paths.
    """

    zoom_levels = sorted(output_zooms.keys())
    save_paths = []

    B, V, T, _, _, _ = input_zooms[zoom_levels[-1]].shape
    plot_ts = (T-1) - np.arange(plot_n_ts)

    for ts in plot_ts:
        # Assume all zoom levels have same number of variables
        B, V, T, _, _, _ = input_zooms[zoom_levels[0]].shape

        if plot_n_vars==-1:
            plot_n_vars=V

        for var in range(plot_n_vars):
            input_maps = {}
            output_maps = {}
            gt_maps = {}
            mask_maps = {}

            for zoom in zoom_levels:
                input_maps[zoom] = input_zooms[zoom][sample, var, ts, :, 0, 0].float().cpu().numpy()
                output_maps[zoom] = output_zooms[zoom][sample, var, ts, :, 0, 0].float().cpu().numpy()
                gt_maps[zoom] = gt_zooms[zoom][sample, var, ts, :, 0, 0].float().cpu().numpy()
                mask_maps[zoom] = mask_zooms[zoom][sample, var, ts, :, 0, 0].float().cpu().numpy() if mask_zooms is not None else None

            # Use embedding index for variable name if available
            if emb is not None and 'VariableEmbedder' in emb:
                var_idx = emb['VariableEmbedder'][sample, var].item()
            else:
                var_idx = var

            save_path = os.path.join(save_dir, f"{plot_name}_{ts}_{var_idx}.png")
            plot_zooms(input_maps, output_maps, gt_maps, mask_maps, save_path, sample_configs=sample_configs)
            save_paths.append(save_path)
    
    return save_paths


def regular_plot(
    gt_data: torch.Tensor,
    in_data: torch.Tensor,
    out_data: torch.Tensor,
    filename: str,
    directory: str,
    gt_coords: Optional[torch.Tensor] = None,
    in_coords: Optional[torch.Tensor] = None
):
    """
    Generate and save comparison plots between ground truth and generated images.

    :param gt_data: Ground truth tensor of shape ``(b, v, t, h, w)``.
    :param in_data: Input tensor of the same shape as ``gt_data``.
    :param out_data: Generated tensor of the same shape as ``gt_data``.
    :param filename: Base filename for saving the images.
    :param directory: Directory where the images will be saved.
    :param gt_coords: Optional coordinate tensor for ground truth data, used for geospatial plotting.
    :param in_coords: Optional coordinate tensor for input data, used for geospatial plotting.
    :return: List of saved file paths.
    """
    # Move data to CPU if necessary
    gt_data, in_data, out_data = gt_data.cpu().float(), in_data.cpu().float(), out_data.cpu().float()
    if gt_coords is not None and in_coords is not None:
        gt_coords, in_coords = gt_coords.cpu(), in_coords.cpu()

    # Set the projection for geospatial plots if coordinates are provided.
    subplot_kw = {"projection": ccrs.Robinson()} if gt_coords is not None and in_coords is not None else {}

    # Limit to a maximum of 12 timesteps and 16 samples for readability.
    gt_data, in_data, out_data = gt_data[:16, :12], in_data[:16, :12], out_data[:16, :12]

    # Define image size and calculate differences between ground truth and output.
    img_size = 3
    differences = gt_data - out_data

    save_paths = []

    # Iterate over each channel in the output data.
    for v in range(out_data.shape[1]):  # Loop over channels

        # Set up the figure layout.
        fig, axes = plt.subplots(
            nrows=gt_data.shape[0], ncols=gt_data.shape[1] * 4,
            figsize=(2 * img_size * gt_data.shape[1] * 4, img_size * gt_data.shape[0]),
            subplot_kw=subplot_kw
        )
        axes = np.atleast_2d(axes)

        # Plot each sample and timestep.
        for i in range(gt_data.shape[0]):  # Loop over samples
            for t in range(gt_data.shape[2]):  # Loop over timesteps
                gt_min = torch.min(gt_data[i, v, t, ..., :])
                gt_max = torch.max(gt_data[i, v, t, ..., :])
                for index, data, vmin, vmax, coords, title in [
                    (t, in_data, None, None, in_coords, "Input"),
                    (t + gt_data.shape[1], gt_data, gt_min, gt_max, gt_coords, "GT"),
                    (t + 2 * gt_data.shape[1], out_data, gt_min, gt_max, gt_coords, "Output"),
                    (t + 3 * gt_data.shape[1], differences, None, None, gt_coords, "Error")
                ]:
                    # Turn off axes for cleaner plots
                    axes[i, index].set_axis_off()
                    axes[i, index].set_title(title)
                    if coords is not None:  # Geospatial plotting with coordinates
                        # Add coastlines and borders.
                        axes[i, index].add_feature(cartopy.feature.COASTLINE, edgecolor="black", linewidth=0.6)
                        axes[i, index].add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.6)
                        # Create a pcolormesh with geospatial coordinates.
                        pcm = axes[i, index].pcolormesh(
                            coords[0, v, t, 0, :, 1], coords[0, v, t, :, 0, 0],
                            np.squeeze(data[i, v, t, ..., :].numpy()),
                            transform=ccrs.PlateCarree(), shading='auto',
                            cmap="RdBu_r", rasterized=True, vmin=vmin, vmax=vmax
                        )
                    else:  # Standard plot without coordinates
                        pcm = axes[i, index].pcolormesh(
                            np.squeeze(data[i, v, t, ..., :].numpy()),
                            vmin=vmin, vmax=vmax, shading='auto', cmap="RdBu_r"
                        )
                    # Add color bar to each difference plot.
                    cb = fig.colorbar(pcm, ax=axes[i, index])

        # Adjust layout and save the figure for the current channel.
        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
        os.makedirs(directory, exist_ok=True)
        save_path = os.path.join(directory, f'{filename}_{v}.png')
        plt.savefig(save_path, bbox_inches='tight')
        save_paths.append(save_path)
        plt.clf()  # Clear the figure for the next iteration.

    # Close all open figures to free memory.
    plt.close('all')
    return save_paths
