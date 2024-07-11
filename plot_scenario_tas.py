import os
import re
import csv

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs

from utils import list_files, make_logger, load_japanese_font, plot_contourf, colors

logger = make_logger()

def normalize(years, values, base):
    """
    Normalize values relative to base period.

    Args:
        years (list): List of years.
        values (list): List of values corresponding to years.
        base (tuple): Start and end year of the base period.

    Returns:
        tuple: Normalized values and mean base value.
    """
    logger.info(f"Normalizing values with base period: {base}")
    start_year, end_year = base
    base_values = [value for year, value in zip(years, values) if start_year <= year <= end_year]
    mean_base_value = np.mean(base_values)
    normalized_values = np.array(values) - mean_base_value
    logger.debug(f"Mean base value: {mean_base_value}")

    return normalized_values, mean_base_value

def load_da(var_id, model_id, experiment_id, variant_label, data_dir):
    """
    Load and process DataArray from netCDF files.

    Args:
        var_id (str): Variable identifier.
        model_id (str): Model identifier.
        experiment_id (str): Experiment identifier.
        variant_label (str): Variant label.
        data_dir (str): Directory containing the data files.

    Returns:
        xarray.DataArray: Processed DataArray containing annual mean grid data.
    """
    logger.info(f"Loading DataArray for {var_id}, {model_id}, {experiment_id}, {variant_label}")
    pattern = rf'{var_id}_.*{model_id}_{experiment_id}_{variant_label}.*\.nc$'
    file_names = list_files(data_dir, pattern)
    logger.debug(f"Found {len(file_names)} matching files")

    datasets = []
    for file_name in file_names:
        try:
            datasets.append(xr.open_dataset(os.path.join(data_dir, file_name)))
        except Exception as e:
            logger.error(f"Error opening file {file_name}: {str(e)}")

    dataset = xr.concat(datasets, dim="time")
    da_raw = dataset[dataset.variable_id]
    da = da_raw.resample(time="YE").mean(skipna=True)  # annual mean grid data
    logger.info(f"DataArray loaded and processed successfully")
    return da

def plot_da(var_id, model_id, variant_label, fig, axes, font_size):
    """
    Plot DataArray for different scenarios.

    Args:
        var_id (str): Variable identifier.
        model_id (str): Model identifier.
        variant_label (str): Variant label.
        fig (matplotlib.figure.Figure): Figure object to plot on.
        axes (list): List of axes to plot on.
        font_size (int): Font size for plot labels.
    """
    logger.info(f"Plotting DataArray for {var_id}, {model_id}, {variant_label}")
    data_dir = "./data_raw/CMIP6"
    
    da_historical = load_da(var_id, model_id, 'historical', variant_label, data_dir)
    da_base_mean = da_historical.isel(time=slice(0, 51)).mean(dim="time", skipna=True)  # 1850-1900 mean
    
    experiment_ids = ['ssp585', 'ssp119']
    das_ssp = {experiment_id: [] for experiment_id in experiment_ids}

    for experiment_id in experiment_ids:
        logger.info(f"Processing experiment: {experiment_id}")
        da_ssp = load_da(var_id, model_id, experiment_id, variant_label, data_dir)
        for time_slice in [slice(0, 36), slice(36, 86)]:  # 2015-2050 and 2051-2100
            da_plot = da_ssp.isel(time=time_slice)
            start_year, end_year = int(da_plot['time'][0].dt.year), int(da_plot['time'][-1].dt.year)
            da_plot_mean = da_plot.mean(dim="time", skipna=True) - da_base_mean
            das_ssp[experiment_id].append((da_plot_mean, start_year, end_year))
    
    cmap = plt.cm.RdBu_r
    norm = matplotlib.colors.Normalize(vmin=-8, vmax=8)
    levels = 16

    for i, experiment_id in enumerate(experiment_ids):
        for j, (da_plot_mean, start_year, end_year) in enumerate(das_ssp[experiment_id]):
            logger.debug(f"Plotting {experiment_id} for years {start_year}-{end_year}")
            ax = axes[i][j]
            cbar_kwargs = dict(orientation="horizontal", aspect=30, shrink=0.60, pad=0.04, extend='both') if i == 1 else {}
            plot_contourf(da_plot_mean, fig, ax, cmap, norm=norm, levels=levels, cbar_kwargs=cbar_kwargs, fontsize=font_size)
            ax.set_title(f"{experiment_id}（{start_year}-{end_year}年平均）", fontsize=font_size)
            ax.coastlines(color='k', linestyle='-', linewidth=0.5)
    
    logger.info("DataArray plotting completed")

def load_timeseries(var_id, model_id, variant_label, experiment_id, data_dir):
    """
    Load time series data from CSV files.

    Args:
        var_id (str): Variable identifier.
        model_id (str): Model identifier.
        variant_label (str): Variant label.
        experiment_id (str): Experiment identifier.
        data_dir (str): Directory containing the data files.

    Returns:
        tuple: Arrays of years and corresponding values.
    """
    logger.info(f"Loading time series for {var_id}, {model_id}, {experiment_id}, {variant_label}")
    file_name = fr'{var_id}_{model_id}_{experiment_id}_{variant_label}.csv'
    file_path = os.path.join(data_dir, file_name)
    years, values = [], []
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)  # Skip header
            for row in reader:
                year, value = int(row[0]), float(row[1])
                years.append(year)
                values.append(value)
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")

    if experiment_id == 'historical':
        years = np.array(years) - years[0] + 1850
    
    logger.info(f"Time series loaded successfully. Years range: {min(years)}-{max(years)}")
    return np.array(years), np.array(values)

def plot_scenario(var_id, model_id, variant_label):
    """
    Create a comprehensive plot of climate scenarios.

    Args:
        var_id (str): Variable identifier.
        model_id (str): Model identifier.
        variant_label (str): Variant label.

    Returns:
        matplotlib.figure.Figure: The complete figure with all plots.
    """
    logger.info(f"Creating scenario plot for {var_id}, {model_id}, {variant_label}")
    font_path = './font/NotoSansJP-Regular.ttf'
    font_size = 12
    load_japanese_font(font_path, font_size)
    matplotlib.rc('axes', lw=0.25, edgecolor='k')

    figsize = np.array([280, 210]) / 30
    projection = ccrs.Robinson()
    path_effects = [pe.withStroke(linewidth=3, foreground="w")]
    
    fig = plt.figure(frameon=False, figsize=figsize)
    gs = fig.add_gridspec(nrows=1000, ncols=1000, left=0.08, right=0.95, wspace=0.0, hspace=0.0)
    ax0 = fig.add_subplot(gs[0:1000, 0:1000], facecolor='none', frameon=True)
    axes = [
        [fig.add_subplot(gs[50:280-25, 0:400], facecolor='none', frameon=True, projection=projection),
         fig.add_subplot(gs[50:280-25, 300:700], facecolor='none', frameon=True, projection=projection)],
        [fig.add_subplot(gs[300+25:578, 0:400], facecolor='none', frameon=True, projection=projection),
         fig.add_subplot(gs[300+25:578, 300:700], facecolor='none', frameon=True, projection=projection)]
    ]
    
    plot_da(var_id, model_id, variant_label, fig, axes, font_size)
    
    data_dir = "./data_processed"
    years_hist, values_hist = load_timeseries(var_id, model_id, variant_label, 'historical', data_dir)
    normalized_values, base_value = normalize(years_hist, values_hist, (1851, 1900))
    
    ax0.plot(years_hist, normalized_values, label=f"{model_id} historical", path_effects=path_effects, c='gray', lw=2)
    ax0.axhline(y=0, color='k', lw=0.9, ls=':')
    
    experiment_ids = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
    for color, experiment_id in zip(colors, experiment_ids):
        logger.info(f"Plotting scenario: {experiment_id}")
        years_ssp, values_ssp = load_timeseries(var_id, model_id, variant_label, experiment_id, data_dir)
        normalized_ssp_values = values_ssp - base_value
        years = np.concatenate([[years_hist[-1]], years_ssp])
        values = np.concatenate([[normalized_values[-1]], normalized_ssp_values])
        ax0.plot(years, values, label=f"{model_id} {experiment_id}", path_effects=path_effects, c=color, lw=2)
        ax0.text(years[-1]+2, values[-1], f"{experiment_id}", path_effects=path_effects, va='center', ha='left', c=color, fontsize=font_size*1.1)
    
    ax0.set_ylabel("1850-1900年平均と比較した地表面気温偏差の全球平均（K）")
    ax0.set_xlabel('年')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    return fig

def main():

    var_id = 'tas'
    model_id = 'MIROC6'
    variant_label = 'r1i1p1f1'
    
    fig = plot_scenario(var_id, model_id, variant_label)
    
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    output_file = f'fig_{script_name}.svg'
    fig.savefig(output_file)

if __name__ == '__main__':
    main()
