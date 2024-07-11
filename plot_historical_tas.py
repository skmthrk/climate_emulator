import os
import re
import csv

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs

from utils import list_files, make_logger, load_japanese_font, plot_contourf

logger = make_logger()

def normalize(years, values, base):
    """
    Normalize values relative to base period.

    Args:
        years (List[int]): List of years.
        values (List[float]): List of values corresponding to years.
        base (Tuple[int, int]): Start and end year of the base period.

    Returns:
        normalized_values (np.ndarray): Normalized values.
        mean_base_value (float): Base value used for normalization
    """
    start_year, end_year = base
    base_values = [value for year, value in zip(years, values) if start_year <= year <= end_year]
    mean_base_value = sum(base_values) / len(base_values)
    normalized_values = np.array(values) - mean_base_value

    return normalized_values, mean_base_value

def load_obs_da():
    """
    Load observational data from HadCRUT5.

    Returns:
        xr.DataArray: Observational data.
    """
    data_dir = "./data_raw/HadCRUT5"
    file_name = 'HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc'
    file_path = os.path.join(data_dir, file_name)
    logger.info(f"Processing {file_path}")
    try:
        with xr.open_dataset(file_path) as dataset:
            return dataset['tas_mean']
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading observational data: {str(e)}")
        raise

def load_model_da(model_id, variant_label):
    """
    Load model data from CMIP6.

    Args:
        model_id (str): ID of the model.
        variant_label (str): Variant label of the model.

    Returns:
        xr.DataArray: Model data.
    """
    data_dir = "./data_raw/CMIP6"
    var_id = 'tas'
    experiment_id = 'historical'
    pattern = rf'{var_id}_.*{model_id}_{experiment_id}_{variant_label}.*\.nc$'
    file_names = list_files(data_dir, pattern)
    datasets = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        logger.info(f"Processing {file_path}")
        try:
            with xr.open_dataset(file_path) as dataset:
                datasets.append(dataset)
        except Exception as e:
            logger.error(f"Error loading model data from {file_path}: {str(e)}")
            raise
    dataset = xr.concat(datasets, dim="time")
    return dataset[dataset.variable_id]


def load_obs_timeseries(base):
    """
    Load observational time series data.

    Args:
        base (Tuple[int, int]): Base period for normalization.

    Returns:
        Tuple[List[int], np.ndarray]: Years and normalized values.
    """
    data_dir = "./data_raw/HadCRUT5"
    file_name = "HadCRUT5.0Analysis_gl.txt"
    file_path = os.path.join(data_dir, file_name)
    logger.info(f"Loading {file_path}")
    years, values = [], []
    try:
        with open(file_path) as f:
            for line in f:
                items = [item.strip() for item in line.split(' ') if item.strip()]
                year, value = int(items[0]), float(items[-1])
                if year != years[-1] if years else True:
                    years.append(year)
                    values.append(value)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading observational time series: {str(e)}")
        raise
    
    values, base_value = normalize(years, values, base)
    return years, values

def load_model_timeseries(model_id, variant_label, base):
    """
    Load model time series data.

    Args:
        model_id (str): ID of the model.
        variant_label (str): Variant label of the model.
        base (Tuple[int, int]): Base period for normalization.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Years and normalized values.
    """
    data_dir = "./data_processed"
    var_id = 'tas'
    experiment_id = 'historical'
    file_name = rf'{var_id}_{model_id}_{experiment_id}_{variant_label}.csv'
    file_path = os.path.join(data_dir, file_name)
    logger.info(f"Loading {file_path}")
    years, values = [], []
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)  # Skip header
            for row in reader:
                year, value = int(row[0]), float(row[1])
                years.append(year)
                values.append(value)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model time series from {file_path}: {str(e)}")
        raise
    
    years = np.array(years) - years[0] + 1850
    values, _ = normalize(years, values, base)
    return years, values

def plot_historical(model_id, variant_label):
    """
    Create a historical plot comparing observational and model data.

    Args:
        model_id (str): ID of the model.
        variant_label (str): Variant label of the model.

    Returns:
        plt.Figure: The resulting figure.
    """
    font_path = './font/NotoSansJP-Regular.ttf'
    font_size = 12
    load_japanese_font(font_path, font_size)

    bgc = 'none'
    path_effects = [pe.withStroke(linewidth=3, foreground="w")]
    frameon = True
    figsize = np.array([230,110])/21
    projection = ccrs.Robinson()
    
    delta = 5
    fig = plt.figure(frameon=False, figsize=figsize, constrained_layout=None)
    gs = fig.add_gridspec(nrows=100, ncols=100, left=0.0, right=0.95, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0:46-delta,0:50], facecolor=bgc, frameon=frameon, projection=projection)
    ax2 = fig.add_subplot(gs[44+delta:100,0:50], facecolor=bgc, frameon=frameon, projection=projection)
    ax3 = fig.add_subplot(gs[0:91,50:100], facecolor=bgc, frameon=frameon)

    da_obs = load_obs_da()
    da_model = load_model_da(model_id, variant_label)
    models = [('HadCRUT5', da_obs, ax1), (model_id, da_model, ax2)]
    skipna = True
    cmap = plt.cm.RdBu_r
    norm = matplotlib.colors.Normalize(vmin=-1.2, vmax=1.2)
    levels = 8
    cbar_kwargs = dict(orientation="horizontal", aspect=35, shrink=0.55, pad=0.04, extend='both')
    
    for model_name, da_raw, ax in models:
        da = da_raw.resample(time="YE").mean(skipna=skipna)
        da_base = da.isel(time=slice(0,51)) # pre-industrial 1850-1900 mean 
        da_plot = da.isel(time=slice(51,165)) # 1900-2014 mean
    
        start_year = int(da_plot['time'][0].dt.year.values)
        end_year = int(da_plot['time'][-1].dt.year.values)
        da_base_mean = da_base.mean(dim="time", skipna=skipna)
        da_plot_mean = da_plot.mean(dim="time", skipna=skipna) - da_base_mean
    
        if model_name == 'HadCRUT5':
            plot_contourf(da_plot_mean, fig, ax, cmap=cmap, norm=norm, levels=levels, fontsize=font_size)
        else:
            plot_contourf(da_plot_mean, fig, ax, cmap=cmap, norm=norm, levels=levels, cbar_kwargs=cbar_kwargs, fontsize=font_size)
        
        title = f"{model_name}（{start_year}-{end_year}年平均）"
        ax.set_title(title)
        ax.coastlines(color='k', linestyle='-', linewidth=0.5)

    base = (1851,1900) # for data normalization
    years_obs, values_obs = load_obs_timeseries(base)
    years_model, values_model = load_model_timeseries(model_id, variant_label, base)
    ax = ax3
    c = 'k'
    lw = 1.5
    idx = years_obs.index(years_model[-1])
    ax.hlines(xmin=years_model[0], xmax=years_model[-1], y=0, color=c, lw=0.9, ls=':')
    ax.plot(years_obs[:idx], values_obs[:idx], label="HadCRUT5", c=c, alpha=0.8, lw=lw*0.4, path_effects=path_effects)
    ax.plot(years_model, values_model, label=f"{model_id} historical", c=c, lw=lw, path_effects=path_effects)
    ax.set_ylabel(f"地表面気温偏差の全球平均（K）")
    ax.set_xlabel('年')
    ax.legend(loc='best')
    for posi in ['top', 'right']:
        ax.spines[posi].set_visible(False)

    return fig

def main():

    model_id = 'MIROC6'
    variant_label = 'r1i1p1f1'

    fig = plot_historical(model_id, variant_label)

    # Save the figure
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    fig.savefig(f'./output/fig_{script_name}.svg')

if __name__ == '__main__':
    main()
