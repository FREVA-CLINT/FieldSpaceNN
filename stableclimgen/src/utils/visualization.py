import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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