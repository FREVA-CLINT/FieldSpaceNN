import os
from typing import Optional

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata


def scatter_plot(input, output, gt, coords_input, coords_output, mask, input_inter=None, input_density=None, save_path=None):

    input = input.cpu().numpy()
    output = output.cpu().to(dtype=torch.float32).numpy()
    gt = gt.cpu().numpy()

    if torch.is_tensor(input_inter):
        input_inter = input_inter.cpu().to(dtype=torch.float32).numpy()
    
    if torch.is_tensor(input_density):
        input_density = input_density.cpu().to(dtype=torch.float32).numpy()

    coords_input = coords_input.rad2deg().cpu().numpy()
    coords_output = coords_output.rad2deg().cpu().numpy()
    plot_input_inter = input_inter is not None
    plot_input_density = input_density is not None

    if mask is not None:
        mask = mask.cpu().bool().numpy()
    else:
        mask = np.zeros_like(input, dtype=bool).squeeze(-1)

    coords_output = coords_output.reshape(gt.shape[0], -1, 2)

    # Define image size and calculate differences between ground truth and output
    img_size = 3

    plot_var = False
    if output.ndim>2 and output.shape[-1]>1:
        output_var = output[...,1]
        output = output[...,0]
        plot_var = True

    # Set up the figure layout
    fig, axes = plt.subplots(
        nrows=gt.shape[0], ncols=4 + plot_input_inter  + plot_input_density + plot_var,
        figsize=(2 * img_size * 4, img_size * gt.shape[0]),
        subplot_kw={"projection": ccrs.Mollweide()}
    )
    axes = np.atleast_2d(axes)
    
    # Plot each sample and timestep
    for i in range(gt.shape[0]):
        gt_min = np.min(gt[i])
        gt_max = np.max(gt[i])
        plot_samples = [
            (input[i][mask[i] == False], coords_input[i][mask[i] == False].reshape(-1, 2), "Input", None, None),
            (gt[i], coords_output[i], "Ground Truth", gt_min, gt_max),
            (output[i], coords_output[i], "Output", gt_min, gt_max),
            (gt[i].squeeze() - output[i].squeeze(), coords_output[i], "Error", None, None)
        ] 
        
        if plot_input_inter:
          plot_samples.insert(1, (input_inter[i], coords_output[i], "Input Interpolated", None, None))
          
        if plot_input_density:
          plot_samples.insert(1, (input_density[i], coords_output[i], "Input Density", None, None))

        if plot_var:
          plot_samples.insert(-2, (output_var[i], coords_output[i], "Output variance", None, None))

        # Loop over samples
        for index, plot_sample in enumerate(plot_samples):
            data, coords, title, vmin, vmax = plot_sample
            # Turn off axes for cleaner plots
            axes[i, index].set_axis_off()
            axes[i, index].set_title(title)
            cax = axes[i, index].scatter(coords[:, 0], coords[:, 1], c=data, transform=ccrs.PlateCarree(), s=6, vmin=vmin, vmax=vmax)

            # Add color bar to each difference plot
            cb = fig.colorbar(cax, ax=axes[i, index], orientation='horizontal', shrink=0.6)

    # Adjust layout and save the figure for the current channel
    plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

def plot_images(
    gt_data: torch.Tensor,
    in_data: torch.Tensor,
    out_data: torch.Tensor,
    filename: str,
    directory: str,
    gt_coords: Optional[torch.Tensor] = None,
    in_coords: Optional[torch.Tensor] = None
) -> None:
    """
    Generate and save comparison plots between ground truth and generated images.

    :param gt_data: Ground truth data tensor of shape (samples, channels, timesteps, height, width).
    :param in_data: Input data tensor of the same shape as `gt_data`.
    :param out_data: Generated data tensor of the same shape as `gt_data`.
    :param filename: Base filename for saving the images.
    :param directory: Directory where the images will be saved.
    :param gt_coords: Optional coordinate tensor for ground truth data, used for geospatial plotting.
    :param in_coords: Optional coordinate tensor for input data, used for geospatial plotting.
    """
    # Move data to CPU if necessary
    gt_data, in_data, out_data = gt_data.cpu(), in_data.cpu(), out_data.cpu()
    if gt_coords is not None and in_coords is not None:
        gt_coords, in_coords = gt_coords.cpu(), in_coords.cpu()

    # Set the projection for geospatial plots if coordinates are provided
    subplot_kw = {"projection": ccrs.Robinson()} if gt_coords is not None and in_coords is not None else {}

    # Limit to a maximum of 12 timesteps and 16 samples for readability
    gt_data, in_data, out_data = gt_data[:16, :12], in_data[:16, :12], out_data[:16, :12]

    # Define image size and calculate differences between ground truth and output
    img_size = 3
    differences = gt_data - out_data

    # Iterate over each channel in the output data
    for v in range(out_data.shape[-2]):  # Loop over channels

        # Set up the figure layout
        fig, axes = plt.subplots(
            nrows=gt_data.shape[0], ncols=gt_data.shape[1] * 4,
            figsize=(2 * img_size * gt_data.shape[1] * 4, img_size * gt_data.shape[0]),
            subplot_kw=subplot_kw
        )
        axes = np.atleast_2d(axes)

        # Plot each sample and timestep
        for i in range(gt_data.shape[0]):  # Loop over samples
            for j in range(gt_data.shape[1]):  # Loop over timesteps
                gt_min = torch.min(gt_data[i, j, ..., v, :])
                gt_max = torch.max(gt_data[i, j, ..., v, :])
                for index, data, vmin, vmax, coords, title in [
                    (j, in_data, None, None, in_coords, "Input"),
                    (j + gt_data.shape[1], gt_data, gt_min, gt_max, gt_coords, "GT"),
                    (j + 2 * gt_data.shape[1], out_data, gt_min, gt_max, gt_coords, "Output"),
                    (j + 3 * gt_data.shape[1], differences, None, None, gt_coords, "Error")
                ]:
                    # Turn off axes for cleaner plots
                    axes[i, index].set_axis_off()
                    axes[i, index].set_title(title)
                    if coords is not None:  # Geospatial plotting with coordinates
                        # Add coastlines and borders
                        axes[i, index].add_feature(cartopy.feature.COASTLINE, edgecolor="black", linewidth=0.6)
                        axes[i, index].add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.6)
                        # Create a pcolormesh with geospatial coordinates
                        pcm = axes[i, index].pcolormesh(
                            coords[0, j, 0, :, v, 1], coords[0, j, :, 0, v, 0],
                            np.squeeze(data[i, j, ..., v, :].numpy()),
                            transform=ccrs.PlateCarree(), shading='auto',
                            cmap="RdBu_r", rasterized=True, vmin=vmin, vmax=vmax
                        )
                    else:  # Standard plot without coordinates
                        pcm = axes[i, index].pcolormesh(
                            np.squeeze(data[i, j, ..., v, :].numpy()),
                            vmin=vmin, vmax=vmax, shading='auto', cmap="RdBu_r"
                        )
                    # Add color bar to each difference plot
                    cb = fig.colorbar(pcm, ax=axes[i, index])

        # Adjust layout and save the figure for the current channel
        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, f'{filename}_{v}.png'), bbox_inches='tight')
        plt.clf()  # Clear the figure for the next iteration

    # Close all open figures to free memory
    plt.close('all')


def griddata_plot(tensor, indices_dict, model, title):
    fig, axes = plt.subplots(
        nrows=1, ncols=1,
        figsize=(6, 6),
        subplot_kw={"projection": ccrs.Mollweide()}
    )

    if indices_dict is not None and isinstance(indices_dict, dict):
        indices = model.get_global_indices_local(indices_dict['sample'],
                                                 indices_dict['sample_level'],
                                                 0)
        coords = model.cell_coords_global[indices].unsqueeze(dim=-2)
    else:
        coords = model.cell_coords_global.unsqueeze(dim=0).unsqueeze(dim=-2)
    coords = coords.rad2deg().cpu().numpy().reshape(tensor.shape[0], -1, 2)

    data = tensor.cpu().to(dtype=torch.float32).squeeze(-1).squeeze(-1).numpy()

    grid_lon = np.arange(-180, 181, 1)
    grid_lat = np.arange(-90, 91, 1)
    target_lon, target_lat = np.meshgrid(grid_lon, grid_lat)

    values = data[0]
    points = coords[0]

    grid_data = griddata(points, values, (target_lon, target_lat), method='linear')

    mesh = axes.pcolormesh(target_lon, target_lat, grid_data,
                           transform=ccrs.PlateCarree(),
                           cmap='viridis',
                           shading='auto')  # Added shading for potentially better results with pcolormesh

    axes.coastlines()
    axes.gridlines(draw_labels=False, linestyle='--', color='gray', alpha=0.5)

    cbar = plt.colorbar(mesh, ax=axes, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Interpolated Data Value')

    axes.set_title(title)
    axes.set_global()

    plt.show()