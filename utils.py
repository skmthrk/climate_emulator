import os
import re
import logging
import __main__

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

colors = [
    '#348ABD', # blue
    '#188487', # breen
    '#E24A33', # orenge
    '#7A68A6', # purple
    '#A60628', # red
    '#467821', # green
    '#CF4457', # pink
]

script_name, _ = os.path.splitext(os.path.basename(__main__.__file__))

def make_logger(name=script_name):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s %(name)s] %(levelname)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    st_handler = logging.StreamHandler()
    fl_handler = logging.FileHandler(filename="log.txt")
    for handler in [st_handler, fl_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def load_japanese_font(font_path, font_size):
    """
    Load a Japanese font and update the matplotlib font parameters.

    Args:
        font_path (str): Path to the Japanese font file
        font_size (int): Font size
    """
    matplotlib.font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({'font.family':'Noto Sans JP', 'font.size': font_size})

def deg2rad(deg):
    """ Convert degrees to radians.

    Args:
        deg (float or array-like): Angle in degrees.

    Returns:
        float or array-like: Angle in radians.
    """
    return deg * np.pi / 180

def surface_area(lat, dlat, dlon):
    '''
    Compute the surface area in square kilometers
    represented by a grid cell at a particular latitude

    Args:
        lat (float): Latitude measured in radian representing the center of a grid cell
        dlat (float): Latitude grid cell size measured in radian
        dlon (float): Longitude grid cell size measured in radian

    Returns:
        float: Grid cell area in square kilometers
    '''
    r0 = 6378.137 # equatorial radius (kilometers)
    r1 = 6356.752 # polar radius (kilometers)

    v = (r0**2)*dlon*(np.sin(lat+dlat/2) - np.sin(lat-dlat/2)) # integration
    v /= ((r0/r1)**2 + (1-(r0/r1)**2)*(np.cos(lat))**2)  # adjustment for ellipse

    return v

def area(da):
    '''
    Compute and return a data array of cell area (square km)
    based on the lat/lon info of the data array
    Only used if area data is not externally supplied

    Args:
        da (xarray.DataArray): Input data array with latitude and longitude coordinates

    Returns:
        xarray.DataArray: Data array of grid cell areas in square kilometers
    '''

    # load latitude/longitude data (measured in degrees)
    lat_data = da.lat.data
    lon_data = da.lon.data

    # convert degrees to radians
    lat_vals = deg2rad(lat_data)
    lon_vals = deg2rad(lon_data)

    # compute grid cell size
    dlat_vals = np.gradient(lat_vals)
    dlon_vals = np.gradient(lon_vals)

    # number of latitude/longitude points
    lat_num = len(lat_vals)
    lon_num = len(lon_vals)

    # compute area matrix (in square kilometer)
    A = np.empty((lat_num, lon_num), dtype=float)
    for i, dlon in enumerate(dlon_vals):
        A[:,i] = surface_area(lat_vals, dlat_vals, dlon)

    # sanity check
    A_true = 510065621 # Earth's surface area in square kilometer
    A_sum = A.sum() # must be close to Earth's surface area
    logger.info(f' - A_tru: {A_true}')
    logger.info(f' - A_sum: {A_sum}')

    # convert the area matrix to data array with lat/lon info and units
    lat_da = xr.DataArray(lat_data, dims="lat", attrs={"units": da.lat.units})
    lon_da = xr.DataArray(lon_data, dims="lon", attrs={"units": da.lon.units})
    dims = ("lat", "lon")
    coords = {"lat": lat_da,"lon": lon_da}
    attrs = {"units": "km2", "long_name": "Grid-Cell Area"}
    array = xr.DataArray(A, dims=dims, coords=coords, attrs=attrs)

    return array

def list_files(dir_path, pattern):
    """
    List files in a directory that match a given pattern.

    Args:
        dir_path (str): Directory path to search for files.
        pattern (str): Regular expression pattern to match file names.

    Returns:
        list: Sorted list of file names that match the pattern.
    """
    matched_files = []
    for filename in os.listdir(dir_path):
        if re.search(pattern, filename):
            matched_files.append(filename)
    matched_files.sort()
    return matched_files

def plot_contourf(da, fig, ax, cmap, norm=None, levels=9, cbar_kwargs={}, fontsize=12):
    """
    Plot filled contours on a map.

    Args:
        da (xr.DataArray): Data to plot.
        fig (plt.Figure): Figure object.
        ax (plt.Axes): Axes object.
        cmap (str): Colormap name.
        norm (matplotlib.colors.Normalize, optional): Normalization for the colormap.
        levels (int, optional): Number of contour levels.
        cbar_kwargs (dict, optional): Keyword arguments for colorbar.
        fontsize (int, optional): Font size for labels.

    Returns:
        matplotlib.contour.QuadContourSet: The contour plot object.
    """
    try:
        lat = da.coords['latitude'][:]
        lon = da.coords['longitude'][:]
    except KeyError:
        lat = da.coords['lat'][:]
        lon = da.coords['lon'][:]

    data_values = da.values
    if norm is None:
        norm = matplotlib.colors.Normalize(vmin=data_values.min(), vmax=data_values.max())
    boundaries = np.linspace(norm.vmin, norm.vmax, levels+1)
    values = [(boundaries[idx] + boundaries[idx-1])/2 for idx in range(1, len(boundaries))]

    data_values, lon = add_cyclic_point(data_values, coord=lon)

    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
    cont = ax.contourf(lon_grid, lat_grid, data_values, cmap=cmap, norm=norm, levels=levels, transform=ccrs.PlateCarree())
    ax.coastlines(color='k', linestyle='-', linewidth=0.5)

    if cbar_kwargs:
        cbar_kwargs['boundaries'] = boundaries
        cbar_kwargs['values'] = values
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, **cbar_kwargs)

    return cont
