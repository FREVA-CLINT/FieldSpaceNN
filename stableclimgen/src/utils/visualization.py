import cartopy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import torch
import healpy as hp

def plot_images(gt_images, gen_images, filename, directory, xr_dss=None):
    gen_images = gen_images.to(torch.device('cpu'))
    gt_images = gt_images.to(torch.device('cpu'))

    # plot max 12 timesteps
    gt_images = gt_images[:16, :, :12]
    gen_images = gen_images[:16, :, :12]

    img_size = 3
    font_size = 12
    differences = gt_images - gen_images
    for c in range(gen_images.shape[1]):
        subplot_kw = {}
        if xr_dss:
            subplot_kw["projection"] = ccrs.Robinson()
        elif gt_images.dim() == 4:
            subplot_kw["projection"] = 'mollweide'

        fig, axes = plt.subplots(nrows=gt_images.shape[0], ncols=gt_images.shape[2] * 3,
                                 figsize=(2 * img_size * gt_images.shape[2] * 3,
                                          img_size * gt_images.shape[0]),
                                 subplot_kw=subplot_kw)
        axes = np.atleast_2d(axes)

        vmin_img = torch.min(gt_images[:, c])
        vmax_img = torch.max(gt_images[:, c])

        vmax_diff = torch.max(torch.abs(differences[:, c]))
        vmin_diff = -vmax_diff

        # plot and save data
        fig.patch.set_facecolor('black')
        for i in range(gt_images.shape[0]):
            for j in range(gt_images.shape[2]):
                for index, image, vmin, vmax in [(j, gt_images, vmin_img, vmax_img),
                                                 (j + gt_images.shape[2], gen_images, vmin_img, vmax_img),
                                                 (j + 2*gt_images.shape[2], differences, vmin_diff, vmax_diff)]:
                    axes[i, index].set_axis_off()
                    if xr_dss:
                        axes[i, index].add_feature(cartopy.feature.COASTLINE, edgecolor="black", linewidth=0.6)
                        axes[i, index].add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--",
                                                   linewidth=0.6)
                        pcm = axes[i, index].pcolormesh(xr_dss[1].coords["lon"], xr_dss[1].coords["lat"],
                                                        np.squeeze(image[i, c, j].detach().numpy()),
                                                        vmin=vmin, vmax=vmax,
                                                        transform=ccrs.PlateCarree(), shading='auto', cmap="RdBu_r",
                                                        linewidth=0, rasterized=True)
                    elif gen_images.dim() == 4:
                        plt.sca(axes[i, index])
                        pcm = hp.mollview(np.squeeze(image[i, c, j].detach().numpy()), min=vmin, max=vmax,
                                          cmap="RdBu_r", hold=True, nest=False, cbar=False)
                    else:
                        pcm = axes[i, index].pcolormesh(np.squeeze(image[i, c, j].detach().numpy()),
                                                        vmin=vmin, vmax=vmax, shading='auto', cmap="RdBu_r",
                                                        linewidth=0)
                if gen_images.dim() != 4:
                    cb = fig.colorbar(pcm, ax=axes[i, j + 2*gen_images.shape[2]])
                    cb.ax.yaxis.set_tick_params(color="white")
                    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="white")

        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0, right=1, bottom=0, top=1)
        plt.savefig('{}/{}_{}.png'.format(
            directory, filename, c),
            bbox_inches='tight')
        plt.clf()

    plt.close('all')
