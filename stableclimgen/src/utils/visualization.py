import os
from typing import Optional

import cartopy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import numpy as np
import torch

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
        mask = mask.squeeze(-1).cpu().bool().numpy()
    else:
        mask = np.zeros_like(input, dtype=bool).squeeze(-1)

    coords_output = coords_output.reshape(gt.shape[0], -1, 2)

    # Define image size and calculate differences between ground truth and output
    img_size = 3

    # Set up the figure layout
    fig, axes = plt.subplots(
        nrows=gt.shape[0], ncols=4 + plot_input_inter  + plot_input_density,
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
