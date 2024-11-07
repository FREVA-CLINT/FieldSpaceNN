import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
import healpy as hp
from typing import Optional


def plot_images(gt_images: torch.Tensor, gen_images: torch.Tensor, filename: str, directory: str, xr_dss: Optional[list] = None) -> None:
    """
    Generate and save comparison plots between ground truth and generated images.

    :param gt_images: Ground truth images tensor of shape (samples, channels, timesteps, height, width).
    :param gen_images: Generated images tensor of the same shape as `gt_images`.
    :param filename: Base filename for saving the images.
    :param directory: Directory where the images will be saved.
    :param xr_dss: Optional list of xarray datasets for geospatial plotting, if applicable.
    """
    gen_images = gen_images.to(torch.device('cpu'))
    gt_images = gt_images.to(torch.device('cpu'))

    # Limit to a maximum of 12 timesteps and 16 samples
    gt_images = gt_images[:16, :12,]
    gen_images = gen_images[:16, :12]

    img_size = 3
    font_size = 12
    differences = gt_images - gen_images

    for v in range(gen_images.shape[-2]):  # Loop over channels
        subplot_kw = {}
        if xr_dss:
            subplot_kw["projection"] = ccrs.Robinson()
        elif gt_images.dim() == 4:
            subplot_kw["projection"] = 'mollweide'

        fig, axes = plt.subplots(nrows=gt_images.shape[0], ncols=gt_images.shape[1] * 3,
                                 figsize=(2 * img_size * gt_images.shape[1] * 3, img_size * gt_images.shape[0]),
                                 subplot_kw=subplot_kw)
        axes = np.atleast_2d(axes)

        vmin_img = torch.min(gt_images[..., v, :]).item()
        vmax_img = torch.max(gt_images[..., v, :]).item()
        vmax_diff = torch.max(torch.abs(differences[..., v, :])).item()
        vmin_diff = -vmax_diff

        fig.patch.set_facecolor('black')  # Set background color to black

        # Plot each sample and timestep
        for i in range(gt_images.shape[0]):  # Loop over samples
            for j in range(gt_images.shape[1]):  # Loop over timesteps
                for index, image, vmin, vmax in [(j, gt_images, vmin_img, vmax_img),
                                                 (j + gt_images.shape[1], gen_images, vmin_img, vmax_img),
                                                 (j + 2*gt_images.shape[1], differences, vmin_diff, vmax_diff)]:
                    axes[i, index].set_axis_off()
                    if xr_dss:  # Geospatial plotting
                        axes[i, index].add_feature(cartopy.feature.COASTLINE, edgecolor="black", linewidth=0.6)
                        axes[i, index].add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--", linewidth=0.6)
                        pcm = axes[i, index].pcolormesh(
                            xr_dss[1].coords["lon"], xr_dss[1].coords["lat"],
                            np.squeeze(image[i, j, ..., v, :].detach().numpy()),
                            vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(), shading='auto',
                            cmap="RdBu_r", linewidth=0, rasterized=True
                        )
                    elif gen_images.dim() == 5:  # Spherical projection for 3D data
                        plt.sca(axes[i, index])
                        pcm = hp.mollview(
                            np.squeeze(image[i, j, ..., v, :].detach().numpy()), min=vmin, max=vmax,
                            cmap="RdBu_r", hold=True, nest=False, cbar=False
                        )
                    else:  # Simple pcolormesh plot
                        pcm = axes[i, index].pcolormesh(
                            np.squeeze(image[i, j, ..., v, :].detach().numpy()),
                            vmin=vmin, vmax=vmax, shading='auto', cmap="RdBu_r", linewidth=0
                        )

                # Add colorbar if not using spherical projection
                if gen_images.dim() != 4:
                    cb = fig.colorbar(pcm, ax=axes[i, j + 2*gen_images.shape[1]])
                    cb.ax.yaxis.set_tick_params(color="white")
                    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")

        # Adjust layout and save the figure
        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
        plt.savefig(f'{directory}/{filename}_{v}.png', bbox_inches='tight')
        plt.clf()  # Clear the figure for the next iteration

    plt.close('all')  # Close all open figures to free memory
