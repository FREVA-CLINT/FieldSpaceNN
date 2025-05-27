import torch
import xarray as xr
import numpy as np

radius_earth= 6371

def get_coords_as_tensor(ds: xr.Dataset, lon:str='vlon', lat:str='vlat', grid_type:str=None, target='torch'):
    """
    :param ds: Input Xarray Dataset to read data from
    :param lon: Input longitude variable to read
    :param lat: Input latitude variable to read
    :param grid_type: Either cell,edge or vertex to read different coordinates

    :returns: Dictionary of the longitude and latitude coordinates (pytorch tensors)
    """
    if grid_type == 'cell':
        lon, lat = 'clon', 'clat'

    elif grid_type == 'vertex':
        lon, lat = 'vlon', 'vlat'

    elif grid_type == 'edge':
        lon, lat = 'elon', 'elat'
    
    elif grid_type == 'lonlat':
        lon, lat = 'lon', 'lat'
    
    if target=='torch':
        lons = torch.tensor(ds[lon].values)
        lats = torch.tensor(ds[lat].values)

        if grid_type=='lonlat':
            lons, lats = torch.meshgrid((lons.deg2rad(),lats.deg2rad()), indexing='xy')
            lons = lons.reshape(-1)
            lats = lats.reshape(-1)


        coords = torch.stack((lons, lats), dim=-1).float()
    else:
        lons = np.array(ds[lon].values)
        lats = np.array(ds[lat].values)

        if grid_type=='lonlat':
            lons, lats = np.meshgrid(np.deg2rad(lons),np.deg2rad(lats),indexing='xy')
            lons = lons.reshape(-1)
            lats = lats.reshape(-1) 
        coords = np.stack((lons, lats), axis=-1).astype(np.float32)

    return coords

def get_grid_type_from_var(ds:xr.Dataset, variable:str) -> dict:
    dims = ds[variable].dims
    spatial_dim = dims[-1]

    if 'lon' in spatial_dim or 'lat' in spatial_dim:
        return 'lonlat'
    
    elif 'cell' in spatial_dim:
        return 'cell'

    elif 'vertex' in spatial_dim:
        return 'vertex'
    
    elif 'edge' in spatial_dim:
        return 'edge'
    


def get_coord_dict_from_var(ds:xr.Dataset, variable:str) -> dict:
    """
    :param ds: Input Xarray Dataset to read data from
    :param variable: Input variable to read
    :returns: Dictionary of the longitude and latitude coordinates (pytorch tensors)
    """
    dims = ds[variable].dims
    spatial_dim = dims[-1]
    
    if not 'lon' in spatial_dim and not 'lat' in spatial_dim:
        coords = list(ds[spatial_dim].coords.keys())
    else:
        coords = dims

    if len(coords)==0:
        coords = list(ds[variable].coords.keys())
        
    lon_c = [var for var in coords if 'lon' in var]
    lat_c = [var for var in coords if 'lat' in var]
    
    if len(lon_c)==0:
        len_points = len(ds.coords[spatial_dim].values)
        dims_match = [dim for dim in ds.coords if len(ds[dim].values)==len_points]

        lon_c = [var for var in dims_match if 'lon' in var]
        lat_c = [var for var in dims_match if 'lat' in var]

    assert len(lon_c)==1, "no longitude variable was found"
    assert len(lat_c)==1, "no latitude variable was found"

    return {'lon':lon_c[0],'lat':lat_c[0]}


def scale_coordinates(coords: torch.tensor, scale_factor:float)->torch.tensor:
    """
    Scale a set of coordinates around their centroid by a given scaling factor.

    Parameters:
    ----------
    coords : torch.Tensor
        Tensor containing coordinate points, with shape (dim, num_points), where `dim` is 
        the number of spatial dimensions (e.g., 2 for 2D coordinates) and `num_points` is 
        the number of coordinate points.
    scale_factor : float
        Factor by which to scale the coordinates around their mean.

    Returns:
    -------
    torch.Tensor
        The scaled coordinates, adjusted around their centroid by the specified scaling factor.
    """
    m = coords.mean(dim=1, keepdim=True)
    return (coords - m) * scale_factor + m


def distance_on_sphere(lon1: torch.tensor, lat1: torch.tensor, lon2: torch.tensor, lat2: torch.tensor) -> torch.tensor:
    """
    :param lon1: target longitude 
    :param lat1: target latitude 
    :param lon2: source longitude 
    :param lat2: source latitude 

    :returns: distances on the sphere between lon1,lat1 and lon2,lat2
    """
    d_lat = torch.abs(lat1-lat2)
    d_lon = torch.abs(lon1-lon2)
    asin = torch.sin(d_lat/2)**2

    d_rad = 2*torch.arcsin(torch.sqrt(asin + (1-asin-torch.sin((lat1+lat2)/2)**2)*torch.sin(d_lon/2)**2))
    return d_rad


def rotate_coord_system(lons: torch.tensor, lats: torch.tensor, rotation_lon: torch.tensor, rotation_lat: torch.tensor):
    """
    :param lons: input longitudes
    :param lats: input latitudes 
    :param rotation_lon: longitude of rotation angles 
    :param rotation_lat: latitudes of rotation angles 

    :returns: rotated coordinates as tuple(longitudes, latitudes)
    """

    theta = rotation_lat
    phi = rotation_lon

    theta = theta.view(-1,1)
    phi = phi.view(-1,1)

    x = (torch.cos(lons) * torch.cos(lats)).view(1,-1)
    y = (torch.sin(lons) * torch.cos(lats)).view(1,-1)
    z = (torch.sin(lats)).view(1,-1)

    rotated_x =  torch.cos(theta)*torch.cos(phi) * x + torch.cos(theta)*torch.sin(phi)*y + torch.sin(theta)*z
    rotated_y = -torch.sin(phi)*x + torch.cos(phi)*y
    rotated_z = -torch.sin(theta)*torch.cos(phi)*x - torch.sin(theta)*torch.sin(phi)*y + torch.cos(theta)*z

    rot_lon = torch.atan2(rotated_y, rotated_x)
    rot_lat = torch.arcsin(rotated_z)

    #lat = arcsin(cos(ϑ) sin(lat') - cos(lon') sin(ϑ) cos(lat'))
    #lon = atan2(sin(lon'), tan(lat') sin(ϑ) + cos(lon') cos(ϑ)) - φ
    
    return rot_lon, rot_lat




def get_distance_angle(lon1: torch.tensor, lat1: torch.tensor, lon2: torch.tensor, lat2: torch.tensor, base:str='polar', periodic_fov:list=None, rotate_coords=True) -> torch.tensor:
    """
    :param lon1: target longitude 
    :param lat1: target latitude 
    :param lon2: source longitude 
    :param lat2: source latitude 
    :param base: Optional: Returns relative coordinates in either polat coordinates (distance, angle) or cartesian coordiantes (distance longitude, distance latitude)
    :param periodic_fov: Optinal if data is defined on a local patch with periodic boundary conditions

    :returns: distances on the sphere between lon1,lat1 and lon2,lat2
    """
    # does not produce proper results for now
    #lat2 = torch.arcsin(torch.cos(theta)*torch.sin(lat2) - torch.cos(lon2)*torch.sin(theta)*torch.cos(lat2))
    #lon2 = torch.atan2(torch.sin(lon2), torch.tan(lat2)*torch.sin(theta) + torch.cos(lon2)*torch.cos(theta)) - phi

    if rotate_coords:
        theta = lat1
        phi = lon1

        x = (torch.cos(lon2) * torch.cos(lat2))
        y = (torch.sin(lon2) * torch.cos(lat2))
        z = (torch.sin(lat2))

        rotated_x =  torch.cos(theta)*torch.cos(phi) * x + torch.cos(theta)*torch.sin(phi)*y + torch.sin(theta)*z
        rotated_y = -torch.sin(phi)*x + torch.cos(phi)*y
        rotated_z = -torch.sin(theta)*torch.cos(phi)*x - torch.sin(theta)*torch.sin(phi)*y + torch.cos(theta)*z

        lon2 = torch.atan2(rotated_y, rotated_x)
        lat2 = torch.arcsin(rotated_z)

        lat1=lon1=0

        d_lons = (lon2).abs()
        d_lats = (lat2).abs() 

        sgn_lat = torch.sign(lat2)
        sgn_lon = torch.sign(lon2)

    else:
        d_lons =  2*torch.arcsin(torch.cos(lat1)*torch.sin(torch.abs(lon2-lon1)/2))
        d_lats = (lat2-lat1).abs() 
        sgn_lat = torch.sign(lat1-lat2)
        sgn_lon = torch.sign(lon1-lon2)
     
    sgn_lat[(d_lats).abs()/torch.pi>1] = sgn_lat[(d_lats).abs()/torch.pi>1]*-1
    d_lats = d_lats*sgn_lat

    sgn_lon[(d_lons).abs()/torch.pi>1] = sgn_lon[(d_lons).abs()/torch.pi>1]*-1
    d_lons = d_lons*sgn_lon

    if periodic_fov is not None:
        rng_lon = (periodic_fov[1] - periodic_fov[0])
        d_lons[d_lons > rng_lon] = d_lons[d_lons > rng_lon] - rng_lon
        d_lons[d_lons < -rng_lon] = d_lons[d_lons < -rng_lon] + rng_lon

    if base == "polar":
        distance = torch.sqrt(d_lats**2 + d_lons**2)
        phi = torch.atan2(d_lats, d_lons)

        return distance.float(), phi.float()

    else:
        return d_lons.float(), d_lats.float()