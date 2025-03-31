import os
from typing import Optional

import cartopy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import numpy as np
import torch

def scatter_plot(input, output, gt, coords_input, coords_output, mask, input_inter=None, save_path=None):
    input = input.cpu().numpy()
    output = output.cpu().to(dtype=torch.float32).numpy()
    gt = gt.cpu().numpy()

    if torch.is_tensor(input_inter):
        input_inter = input_inter.cpu().to(dtype=torch.float32).numpy()

    coords_input = coords_input.rad2deg().cpu().numpy()
    coords_output = coords_output.rad2deg().cpu().numpy()
    plot_input_inter = input_inter is not None

    if mask is not None:
        mask = mask.squeeze(-1).cpu().bool().numpy()
    else:
        mask = np.zeros_like(input, dtype=bool).squeeze(-1)

    coords_output = coords_output.reshape(gt.shape[0], -1, 2)

    # Define image size and calculate differences between ground truth and output
    img_size = 3

    # Set up the figure layout
    fig, axes = plt.subplots(
        nrows=gt.shape[0], ncols=4 + torch.is_tensor(input_inter),
        figsize=(2 * img_size * 4, img_size * gt.shape[0]),
        subplot_kw={"projection": ccrs.Mollweide()}
    )
    axes = np.atleast_2d(axes)

    # Plot each sample and timestep
    for i in range(gt.shape[0]):
        plot_samples = [
            (0, input[i][mask[i] == False], coords_input[i][mask[i] == False].reshape(-1, 2), "Input"),
            (1, gt[i], coords_output[i], "Ground Truth"),
            (2, output[i], coords_output[i], "Output"),
            (3, gt[i].squeeze() - output[i].squeeze(), coords_output[i], "Error")
        ] if not torch.is_tensor(input_inter) else [
            (0, input[i], coords_input[i], "Input"),
            (1, plot_input_inter[i], coords_output[i], "Input Interpolated"),
            (2, gt[i], coords_output[i], "Ground Truth"),
            (3, output[i], coords_output[i], "Output"),
            (4, gt[i].squeeze() - output[i].squeeze(), coords_output[i], "Error")
        ]
        # Loop over samples
        for index, data, coords, title in plot_samples:
            # Turn off axes for cleaner plots
            axes[i, index].set_axis_off()
            axes[i, index].set_title(title)
            cax = axes[i, index].scatter(coords[:, 0], coords[:, 1], c=data, transform=ccrs.PlateCarree(), s=6)

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
                            coords[0, 0, :, v, 1], coords[0, :, 0, v, 0],
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
