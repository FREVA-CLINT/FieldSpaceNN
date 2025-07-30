import os
from typing import Optional,Dict

import healpy as hp
import cartopy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import numpy as np
import torch
from scipy.interpolate import griddata

def healpix_plot_local(values, zoom, ax=None, vmin=None, vmax=None, title="", zoom_patch_sample=1, patch_index=0):
    ipix = np.arange(hp.nside2npix(2**zoom)).reshape(-1,4**(zoom-zoom_patch_sample))[patch_index[0]]
    lon, lat = hp.pix2ang(2**zoom, ipix, nest=True, lonlat=True)

    n_p = max(100, int(np.sqrt(len(lon))))

    # Create grid for interpolation
    lon_grid = np.linspace(np.min(lon), np.max(lon), n_p)
    lat_grid = np.linspace(np.min(lat), np.max(lat), n_p)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate scattered data onto regular lat/lon grid
    grid_values = griddata(
        points=np.stack((lon, lat), axis=-1),
        values=values,
        xi=(lon_mesh, lat_mesh),
        method='nearest',
        fill_value=np.nan
    )

    # Plot using contourf
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

def plot_zooms(input_maps, output_maps, gt_maps, mask_maps, save_path, sample_dict={}):
    """
    Plot HEALPix maps per zoom level for a specific variable and save the result.
    """
    zoom_levels = sorted(input_maps.keys())
    n_rows = len(zoom_levels)
    n_cols = 4  # input, output, gt, error
    n_plots = n_rows * n_cols

    fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
    titles = ['Input', 'Output', 'Ground Truth', 'Error']

    for row_idx, zoom in enumerate(zoom_levels):
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

            if len(sample_dict)==0:
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
                                  **sample_dict)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def healpix_plot_zooms_var(input_zooms: Dict[int, torch.Tensor], 
                           output_zooms: Dict[int, torch.Tensor],
                           gt_zooms: Dict[int, torch.Tensor],
                           save_dir: str, 
                           mask_zooms: Dict[int, torch.Tensor] = None,
                           sample_dict={}, 
                           emb=None,
                           plot_name: str = "healpix_plot",
                           sample: int = 0,
                           plot_n_vars: int= -1,
                           plot_n_ts: int = 1):
    """
    Create HEALPix plots for each variable across zoom levels and save them.
    """

    zoom_levels = sorted(input_zooms.keys())
    save_paths = []

    for ts in range(plot_n_ts):
        # Assume all zoom levels have same number of variables
        B, V, T, _, _ = input_zooms[zoom_levels[0]].shape

        if plot_n_vars==-1:
            plot_n_vars=V

        for var in range(plot_n_vars):
            input_maps = {}
            output_maps = {}
            gt_maps = {}
            mask_maps = {}

            for zoom in zoom_levels:
                input_maps[zoom] = input_zooms[zoom][sample, var, ts, :, 0].float().cpu().numpy()
                output_maps[zoom] = output_zooms[zoom][sample, var, ts, :, 0].float().cpu().numpy()
                gt_maps[zoom] = gt_zooms[zoom][sample, var, ts, :, 0].float().cpu().numpy()
                mask_maps[zoom] = mask_zooms[zoom][sample, var, ts, :, 0].float().cpu().numpy() if mask_zooms is not None else None

            # Use embedding index for variable name if available
            if emb is not None and 'VariableEmbedder' in emb:
                var_idx = emb['VariableEmbedder'][sample, var].item()
            else:
                var_idx = var

            save_path = os.path.join(save_dir, f"{plot_name}_{ts}_{var_idx}.png")
            plot_zooms(input_maps, output_maps, gt_maps, mask_maps, save_path, sample_dict=sample_dict)
            save_paths.append(save_path)
    
    return save_paths