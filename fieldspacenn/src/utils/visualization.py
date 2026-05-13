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
    if isinstance(patch_index, (list, tuple, np.ndarray, torch.Tensor)):
        patch_idx = int(patch_index[0])
    else:
        patch_idx = int(patch_index)

    ipix = np.arange(hp.nside2npix(2**zoom)).reshape(-1,4**(zoom-zoom_patch_sample))[patch_idx]
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
    input_zoom_levels = sorted(input_maps.keys())
    output_zoom_levels = sorted(set(output_maps.keys()) & set(gt_maps.keys()))
    zoom_levels = sorted(set(input_zoom_levels) | set(output_zoom_levels))
    if len(zoom_levels) == 0:
        return

    n_rows = len(zoom_levels)
    n_cols = 4  # input, output, gt, error
    n_plots = n_rows * n_cols

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    titles = ['Input', 'Output', 'Ground Truth', 'Error']

    for row_idx, zoom in enumerate(zoom_levels):
        sample_configs_zoom = dict(sample_configs.get(zoom, {"zoom_patch_sample": -1, "patch_index": np.array([0])}))
        sample_configs_zoom.setdefault("zoom_patch_sample", -1)
        sample_configs_zoom.setdefault("patch_index", np.array([0]))

        out_map = output_maps.get(zoom)
        gt_map = gt_maps.get(zoom)

        inp_map = input_maps.get(zoom)

        mask_map = mask_maps.get(zoom) if mask_maps is not None else None

        error_map = (out_map - gt_map) if out_map is not None and gt_map is not None else None

        maps = [inp_map, out_map, gt_map, error_map]
        in_minmax = (
            np.quantile(inp_map, [0.001, 0.999]) if inp_map is not None else (None, None)
        )
        if gt_map is not None:
            gt_minmax = np.quantile(gt_map, [0.001, 0.999])
        elif out_map is not None:
            gt_minmax = np.quantile(out_map, [0.001, 0.999])
        else:
            gt_minmax = (None, None)
        err_minmax = (
            np.quantile(error_map, [0.001, 0.999]) if error_map is not None else (None, None)
        )
        min_max = [in_minmax, gt_minmax, gt_minmax, err_minmax]

        for col_idx in range(n_cols):
            plot_idx = row_idx * n_cols + col_idx + 1  # 1-based index for subplots
            map_i = maps[col_idx]

            if map_i is None:
                ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                ax.set_title(f"{titles[col_idx]} (zoom {zoom})")
                ax.set_axis_off()
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            if sample_configs_zoom['zoom_patch_sample']==-1:
                hp.mollview(map_i,
                            title=f"{titles[col_idx]} (zoom {zoom})",
                            sub=(n_rows, n_cols, plot_idx),
                            nest=True,
                            min=min_max[col_idx][0],
                            max=min_max[col_idx][1],
                            fig=fig)
            else:
                ax = fig.add_subplot(n_rows, n_cols, plot_idx)
                healpix_plot_local(map_i, 
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
    def _is_plot_tensor(x: Any) -> bool:
        return torch.is_tensor(x) and x.ndim >= 6

    if output_zooms is None or gt_zooms is None:
        return []

    zoom_levels_input = sorted(
        [zoom for zoom, tensor in input_zooms.items() if _is_plot_tensor(tensor)]
    )
    zoom_levels_output_gt = sorted(
        [
            zoom
            for zoom in (set(output_zooms.keys()) & set(gt_zooms.keys()))
            if _is_plot_tensor(output_zooms.get(zoom)) and _is_plot_tensor(gt_zooms.get(zoom))
        ]
    )
    if len(zoom_levels_output_gt) == 0:
        return []

    save_paths = []

    ref_zoom = zoom_levels_output_gt[-1]
    _, v_out, t_out, _, _, _ = output_zooms[ref_zoom].shape
    _, v_gt, t_gt, _, _, _ = gt_zooms[ref_zoom].shape
    max_vars = min(v_out, v_gt)
    max_ts = min(t_out, t_gt)

    if max_vars <= 0 or max_ts <= 0:
        return []

    if plot_n_vars == -1:
        n_plot_vars = max_vars
    else:
        n_plot_vars = min(plot_n_vars, max_vars)

    n_plot_ts = min(plot_n_ts, max_ts)
    plot_ts = (max_ts - 1) - np.arange(n_plot_ts)

    for ts in plot_ts:
        for var in range(n_plot_vars):
            input_maps = {}
            output_maps = {}
            gt_maps = {}
            mask_maps = {}

            for zoom in zoom_levels_input:
                input_zoom = input_zooms[zoom]
                if var < input_zoom.shape[1]:
                    sample_in = min(sample, input_zoom.shape[0] - 1)
                    var_in = var
                    ts_in = min(int(ts), input_zoom.shape[2] - 1)
                    input_maps[zoom] = input_zoom[sample_in, var_in, ts_in, :, 0, 0].float().cpu().numpy()

            for zoom in zoom_levels_output_gt:
                output_zoom = output_zooms[zoom]
                gt_zoom = gt_zooms[zoom]
                if var >= output_zoom.shape[1] or var >= gt_zoom.shape[1]:
                    continue

                sample_out = min(sample, output_zoom.shape[0] - 1)
                sample_gt = min(sample, gt_zoom.shape[0] - 1)
                var_out = var
                var_gt = var
                ts_out = min(int(ts), output_zoom.shape[2] - 1)
                ts_gt = min(int(ts), gt_zoom.shape[2] - 1)

                output_maps[zoom] = output_zoom[sample_out, var_out, ts_out, :, 0, 0].float().cpu().numpy()
                gt_maps[zoom] = gt_zoom[sample_gt, var_gt, ts_gt, :, 0, 0].float().cpu().numpy()

                if mask_zooms is not None and zoom in mask_zooms and _is_plot_tensor(mask_zooms.get(zoom)):
                    mask_zoom = mask_zooms[zoom]
                    sample_mask = min(sample, mask_zoom.shape[0] - 1)
                    if var < mask_zoom.shape[1]:
                        var_mask = var
                        ts_mask = min(int(ts), mask_zoom.shape[2] - 1)
                        mask_maps[zoom] = mask_zoom[sample_mask, var_mask, ts_mask, :, 0, 0].float().cpu().numpy()

            # Use embedding index for variable name if available
            if emb is not None and 'VariableEmbedder' in emb:
                var_embedder = emb['VariableEmbedder']
                if (
                    torch.is_tensor(var_embedder)
                    and var_embedder.ndim >= 2
                    and sample < var_embedder.shape[0]
                    and var < var_embedder.shape[1]
                ):
                    var_idx = var_embedder[sample, var].item()
                else:
                    var_idx = var
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
