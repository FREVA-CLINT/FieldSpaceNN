import os
from typing import Optional

import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch


def scatter_plot(input, output, gt, coords_input, coords_output, mask, save_path=None):

    coords_input = coords_input.rad2deg().cpu().numpy()
    coords_output = coords_output.rad2deg().cpu().numpy()

    input = input.cpu().numpy()
    output = output.cpu().numpy()
    gt = gt.cpu().numpy()
    
    if mask is not None:
        mask = mask.cpu().numpy()
        input = input[mask==False]
        coords_input = coords_input[:,mask.squeeze()==False]


    projection = ccrs.Mollweide()

    fig = plt.figure(figsize= (20,8))

    ax = plt.subplot(1,4,1,projection=projection)
    cax = ax.scatter(coords_input[0], coords_input[1], c=input, transform=ccrs.PlateCarree(),s=6)
    plt.colorbar(cax, ax=ax)


    ax = plt.subplot(1,4,2,projection=projection)
    cax = ax.scatter(coords_output[0], coords_output[1], c=output, transform=ccrs.PlateCarree(),s=5)
    plt.colorbar(cax, ax=ax)

    ax = plt.subplot(1,4,3,projection=projection)
    cax = ax.scatter(coords_output[0], coords_output[1], c=gt, transform=ccrs.PlateCarree(),s=6)
    plt.colorbar(cax, ax=ax)

    ax = plt.subplot(1,4,4,projection=projection)
    cax = ax.scatter(coords_output[0], coords_output[1], c=gt.squeeze()-output.squeeze(), transform=ccrs.PlateCarree(),s=6)
    plt.colorbar(cax, ax=ax)

    if save_path is not None:
        plt.savefig(save_path)

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

        # Determine color limits for data and difference plots
        vmin_data, vmax_data = gt_data[..., v, :].min().item(), gt_data[..., v, :].max().item()
        vmax_diff = torch.max(torch.abs(differences[..., v, :])).item()
        vmin_diff = -vmax_diff

        # Set background color for the figure
        fig.patch.set_facecolor('black')

        # Plot each sample and timestep
        for i in range(gt_data.shape[0]):  # Loop over samples
            for j in range(gt_data.shape[1]):  # Loop over timesteps
                for index, data, vmin, vmax, coords in [
                    (j, in_data, vmin_data, vmax_data, in_coords),
                    (j + gt_data.shape[1], gt_data, vmin_data, vmax_data, gt_coords),
                    (j + 2 * gt_data.shape[1], out_data, vmin_data, vmax_data, gt_coords),
                    (j + 3 * gt_data.shape[1], differences, vmin_diff, vmax_diff, gt_coords)
                ]:
                    # Turn off axes for cleaner plots
                    axes[i, index].set_axis_off()
                    if coords is not None:  # Geospatial plotting with coordinates
                        # Add coastlines and borders
                        axes[i, index].add_feature(cartopy.feature.COASTLINE, edgecolor="black", linewidth=0.6)
                        axes[i, index].add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.6)
                        # Create a pcolormesh with geospatial coordinates
                        pcm = axes[i, index].pcolormesh(
                            coords[0, 1, 0, :, v], coords[0, 0, :, 0, v],
                            np.squeeze(data[i, j, ..., v, :].numpy()),
                            vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), shading='auto',
                            cmap="RdBu_r", rasterized=True
                        )
                    else:  # Standard plot without coordinates
                        pcm = axes[i, index].pcolormesh(
                            np.squeeze(data[i, j, ..., v, :].numpy()),
                            vmin=vmin, vmax=vmax, shading='auto', cmap="RdBu_r"
                        )
                # Add color bar to each difference plot
                cb = fig.colorbar(pcm, ax=axes[i, j + 3 * out_data.shape[1]])
                cb.ax.yaxis.set_tick_params(color="white")
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")

        # Adjust layout and save the figure for the current channel
        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, f'{filename}_{v}.png'), bbox_inches='tight')
        plt.clf()  # Clear the figure for the next iteration

    # Close all open figures to free memory
    plt.close('all')
