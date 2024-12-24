import os 
import re
import csv
import sys
import argparse

import numpy as np
import xarray as xr
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from utils import list_files, make_logger, load_japanese_font, plot_contourf, colors

logger = make_logger()

unit_conversion = {
    'co2': (7.82116, 'GtCO2'), # ppm to GtCO2
    'ch4': (2.85094, 'MtCH4'), # ppb to MtCH4
    'n2o': (7.82187, 'MtN2O'), # ppb to MtN2O
}

baseline_concentration = {
    'co2': 278 * unit_conversion['co2'][0],
    'ch4': 720 * unit_conversion['ch4'][0],
    'n2o': 270 * unit_conversion['n2o'][0],
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="MIROC6", help="model ID")
    return parser.parse_args()

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

    Raises:
        FileNotFoundError: If the specified CSV file is not found.
        Exception: If there's an error reading the CSV file.
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
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

    if experiment_id == 'historical':
        years = np.array(years) - years[0] + 1850
    
    logger.info(f"Time series loaded successfully. Years range: {min(years)}-{max(years)}")
    return np.array(years), np.array(values)

def load_cmip_dataset(var_id, model_id, variant_label, data_dir):
    """
    Load CMIP dataset for various experiments.

    Args:
        var_id (str): Variable identifier.
        model_id (str): Model identifier.
        variant_label (str): Variant label.
        data_dir (str): Directory containing the data files.

    Returns:
        dict: A dictionary containing experiment data, with keys as experiment IDs
              and values as tuples of (years, values).
    """
    logger.info(f"Loading CMIP dataset for {var_id}, {model_id}, {variant_label}")
    cmip_dataset = {}
    
    # historical
    experiment_id = 'historical'
    years, values = load_timeseries(var_id, model_id, variant_label, experiment_id, data_dir)
    hist_value = values[0]
    cmip_dataset[experiment_id] = (years, values - hist_value)
    
    # SSP
    experiment_ids = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
    for experiment_id in experiment_ids:
        years, values = load_timeseries(var_id, model_id, variant_label, experiment_id, data_dir)
        cmip_dataset[experiment_id] = (years, values - hist_value)

    logger.info(f"CMIP dataset loaded successfully for {len(cmip_dataset)} experiments")
    return cmip_dataset

def load_concentration_dataset(data_dir, unit_conversion, baseline_concentration):
    """
    Load concentration dataset for various scenarios.

    Args:
        data_dir (str): Directory containing the data files.
        unit_conversion (dict): Dictionary of unit conversion factors.
        baseline_concentration (dict): Dictionary of baseline concentrations.

    Returns:
        dict: A dictionary containing concentration data for various scenarios.
    """
    years = np.arange(1750, 2501, 1)
    scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
    var_ids = ['co2', 'ch4', 'n2o']
    
    file_name = "rcmip-concentrations-annual-means-v5-1-0.csv"
    file_path = os.path.join(data_dir, file_name)
    logger.info(f"Reading concentration data from {file_path}")
    df_ssp = pd.read_csv(file_path)

    concentration_dataset = {}
    for scenario in scenarios:
        data = {}
        for var_id in var_ids:
            data[var_id] = df_ssp.loc[
                (df_ssp["Region"] == "World")
                & (df_ssp["Scenario"] == scenario)
                & (df_ssp["Variable"] == "Atmospheric Concentrations|{}".format(var_id.upper())),
                "1750":"2500",
            ].interpolate(axis=1).values.squeeze()*unit_conversion[var_id][0]
        concentration_dataset[scenario] = (years, data)

    logger.info(f"Concentration dataset loaded successfully for {len(scenarios)} scenarios")
    return concentration_dataset

def load_forcing_dataset(data_dir):
    """
    Load forcing dataset for various scenarios.

    Args:
        data_dir (str): Directory containing the data files.

    Returns:
        dict: A dictionary containing forcing data for various scenarios.
    """
    years = np.arange(1750, 2501, 1)
    var_ids = ['co2', 'ch4', 'n2o']
    scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']

    file_name = "rcmip-radiative-forcing-annual-means-v5-1-0.csv"
    file_path = os.path.join(data_dir, file_name)
    logger.info(f"Reading forcing data from {file_path}")
    df_ssp = pd.read_csv(file_path)

    forcing_dataset = {}
    for scenario in scenarios:
        data = {}
        for var_id in var_ids:
            data[var_id] = df_ssp.loc[
                (df_ssp["Region"] == "World")
                & (df_ssp["Scenario"] == scenario)
                & (df_ssp["Variable"] == "Effective Radiative Forcing|Anthropogenic|{}".format(var_id.upper())),
                "1750":"2500",
            ].interpolate(axis=1).values.squeeze()
        for var_id in ['anthropogenic', 'natural']:
            data[var_id] = df_ssp.loc[
                (df_ssp["Region"] == "World")
                & (df_ssp["Scenario"] == scenario)
                & (df_ssp["Variable"] == "Effective Radiative Forcing|{}".format(var_id.capitalize())),
                "1750":"2500",
            ].interpolate(axis=1).values.squeeze()
        data['total'] = df_ssp.loc[
            (df_ssp["Region"] == "World")
            & (df_ssp["Scenario"] == scenario)
            & (df_ssp["Variable"] == "Effective Radiative Forcing"),
            "1750":"2500",
        ].interpolate(axis=1).values.squeeze()
        data['non-co2'] = data['total'] - data['co2']
        forcing_dataset[scenario] = (years, data)

    logger.info(f"Forcing dataset loaded successfully for {len(scenarios)} scenarios")
    return forcing_dataset

def load_calibration_dataset(model_id, variant_label, data_dir):
    """
    Load calibration dataset for a specific model and variant.

    Args:
        model_id (str): Model identifier.
        variant_label (str): Variant label.
        data_dir (str): Directory containing the data files.

    Returns:
        dict: A dictionary containing calibration data.
    """
    logger.info(f"Loading calibration dataset for {model_id}, {variant_label}")
    experiment_id = 'abrupt-4xCO2'
    file_name = f"df_{model_id}_{experiment_id}_{variant_label}.csv"
    file_path = os.path.join(data_dir, file_name)
    data = {}
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, t in enumerate(reader):
                if i == 0:
                    keys = t[1:]
                    continue
                for key, value in zip(keys, t[1:]):
                    data.setdefault(key, []).append(float(value))
    except FileNotFoundError:
        logger.error(f"Calibration file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading calibration file {file_path}: {str(e)}")
        raise

    T1_values = np.array(data['T1'])
    R_values = np.array(data['R'])
    year_min = 1850
    years = year_min + np.arange(1, len(T1_values)+1, 1)
    calibration_dataset = {experiment_id:dict(T1=(years, T1_values), R=(years, R_values))}

    logger.info("Calibration dataset loaded successfully")
    return calibration_dataset

def load_calibrated_parameters(model_id, variant_label, data_dir):
    """
    Load calibrated parameters for a specific model and variant.

    Args:
        model_id (str): Model identifier.
        variant_label (str): Variant label.
        data_dir (str): Directory containing the data files.

    Returns:
        dict: A dictionary containing calibrated parameters.
    """
    logger.info(f"Loading calibrated parameters for {model_id}, {variant_label}")
    experiment_id = 'abrupt-4xCO2'
    file_name = f"parameter_{model_id}_{experiment_id}_{variant_label}.csv"
    file_path = os.path.join(data_dir, file_name)
    parameters = {}
    try:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for i, t in enumerate(reader):
                key, val = t
                parameters[key] = float(val)
    except FileNotFoundError:
        logger.error(f"Calibrated parameters file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading calibrated parameters file {file_path}: {str(e)}")
        raise

    return parameters

class Emulator:
    """
    A class representing an emulator for climate models.

    Attributes:
        parameters (dict): Model parameters.
        calibration_dataset (dict): Calibration dataset.
        concentration_dataset (dict): Concentration dataset.
        forcing_dataset (dict): Forcing dataset.
        baseline_concentration (dict): Baseline concentration values.
        x_hist_end (numpy.ndarray): End state of historical simulation.
    """

    def __init__(self, parameters, calibration_dataset=None, concentration_dataset=None, forcing_dataset=None, baseline_concentration=None):
        """
        Initialize the Emulator.

        Args:
            parameters (dict): Model parameters.
            calibration_dataset (dict, optional): Calibration dataset.
            concentration_dataset (dict, optional): Concentration dataset.
            forcing_dataset (dict, optional): Forcing dataset.
            baseline_concentration (dict, optional): Baseline concentration values.
        """
        self.parameters = parameters
        self.calibration_dataset = calibration_dataset
        self.concentration_dataset = concentration_dataset
        self.forcing_dataset = forcing_dataset
        self.baseline_concentration = baseline_concentration
        self.x_hist_end = None
        logger.info("Emulator initialized")

    def dxdt(self, x, t, u):
        """
        Compute the time derivative of the state vector.

        Args:
            x (numpy.ndarray): Current state vector.
            t (float): Current time.
            u (callable): Forcing function.

        Returns:
            list: Time derivatives of the state vector components.
        """
        A = self.build_A()
        gamma = self.parameters["gamma"]
        F, T1, T2 = x
        dFdt = A[0,0]*F + A[0,1]*T1 + A[0,2]*T2 + gamma*u(t)
        dT1dt = A[1,0]*F + A[1,1]*T1 + A[1,2]*T2
        dT2dt = A[2,0]*F + A[2,1]*T1 + A[2,2]*T2

        return [dFdt, dT1dt, dT2dt]

    def build_A(self):
        """
        Build the state transition matrix A.

        Returns:
            numpy.ndarray: The state transition matrix A.
        """
        parameters = self.parameters
        gamma = parameters['gamma']
        chi1 = parameters['chi1']
        chi2 = parameters['chi2']
        kappa1 = parameters['kappa1']
        kappa2 = parameters['kappa2']
        epsilon = parameters['epsilon']
    
        A = np.zeros((3,3))
        A[1-1,1-1] = -gamma
        A[2-1,1-1] = 1/chi1
        A[2-1,2-1] = -(kappa1 + epsilon + kappa2)/chi1
        A[2-1,3-1] = (kappa2 + epsilon)/chi1
        A[3-1,2-1] = kappa2/chi2
        A[3-1,3-1] = -kappa2/chi2

        return A

    def simulate(self, experiment_id):
        """
        Simulate the model for a given experiment.

        Args:
            experiment_id (str): Identifier for the experiment to simulate.

        Returns:
            tuple: Time domain and state values over time.
        """
        logger.info(f"Starting simulation for experiment: {experiment_id}")
        if experiment_id == 'abrupt-4xCO2':
            u = lambda t: self.parameters['Fbar']
            years = self.calibration_dataset[experiment_id]['T1'][0]
            x_init = [self.parameters["Fbar"], 0, 0]
            t_start, t_end = years[0], years[-1]
        elif experiment_id == 'historical':
            years, forcing = self.generate_forcing(experiment_id)
            u = CubicSpline(years, forcing)
            x_init = [u(years[0]), 0, 0]
            t_start, t_end = years[0], years[-1]
        else:
            years, forcing = self.generate_forcing(experiment_id)
            u = CubicSpline(years, forcing)
            if self.x_hist_end is None:
                logger.info("Historical simulation not yet run, simulating historical first")
                self.simulate('historical')
            x_init = self.x_hist_end
            t_start, t_end = years[0], years[-1]
            
        t_domain = np.linspace(t_start, t_end, 2000)
        logger.debug(f"Solving ODE for {experiment_id} from {t_start} to {t_end}")
        x_values = odeint(self.dxdt, x_init, t_domain, args=(u,))

        if experiment_id == 'historical':
            self.x_hist_end = x_values[-1,:]
            logger.debug("Updated historical end state")

        logger.info(f"Simulation completed for {experiment_id}")
        return t_domain, x_values

    def generate_forcing(self, experiment_id):
        """
        Generate forcing data for a given experiment.

        Args:
            experiment_id (str): Identifier for the experiment.

        Returns:
            tuple: Years and corresponding forcing values.
        """
        if experiment_id == 'historical':
            years, concentration_data = self.concentration_dataset['ssp245']
            forcing_co2 = (self.parameters['Fbar']/np.log(4))*np.log(concentration_data['co2']/self.baseline_concentration['co2'])
            forcing_nonco2 = self.forcing_dataset['ssp245'][1]['non-co2']
            forcing = forcing_co2 + forcing_nonco2
            idx_end = np.where(years == 2015)[0][0]
            years, forcing = years[:idx_end], forcing[:idx_end]
        else:
            years, concentration_data = self.concentration_dataset[experiment_id]
            forcing_co2 = (self.parameters['Fbar']/np.log(4))*np.log(concentration_data['co2']/self.baseline_concentration['co2'])
            forcing_nonco2 = self.forcing_dataset[experiment_id][1]['non-co2']
            forcing = forcing_co2 + forcing_nonco2
            idx_start, idx_end = np.where(years == 2015)[0][0], np.where(years == 2100)[0][0]
            years, forcing = years[idx_start-1:idx_end+1], forcing[idx_start-1:idx_end+1]

        logger.debug(f"Forcing generated for {experiment_id} from {years[0]} to {years[-1]}")
        return years, forcing

def plot_figure(model_id, calibration_dataset, cmip_dataset, model_output):
    """
    Plot the figure comparing model output with calibration and CMIP datasets.

    Args:
        model_id (str): Identifier for the model.
        calibration_dataset (dict): Calibration dataset.
        cmip_dataset (dict): CMIP dataset.
        model_output (dict): Model output data.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # plot
    font_path = './font/NotoSansJP-Regular.ttf'
    font_size = 12
    load_japanese_font(font_path, font_size)
    matplotlib.rc('axes', lw=0.25, edgecolor='k')
    path_effects = [pe.withStroke(linewidth=3, foreground="w")]

    figsize = np.array([297,150])/24
    fig = plt.figure(frameon=False, figsize=figsize)
    dw = 2
    gs = fig.add_gridspec(nrows=100, ncols=100, wspace=0.0, left=0.05, right=0.98, hspace=0.0)
    ax1 = fig.add_subplot(gs[0:100,0:50-dw], facecolor='none')
    ax2 = fig.add_subplot(gs[0:100,50+dw:100], facecolor='none')
    letters = ['A', 'B']
    
    ax = ax1
    experiment_id = 'abrupt-4xCO2'
    color = '#1e384d'
    ax.plot(*model_output[experiment_id], label=f"{experiment_id} (簡易モデル)", c=color, path_effects=path_effects, lw=2.5)
    ax.scatter(*calibration_dataset[experiment_id]['T1'], label=f"{experiment_id} ({model_id})", c=color, zorder=0, marker='o', s=15, edgecolors='w', linewidth=0.9)

    ax = ax2
    experiment_id = 'historical'
    color = 'k'
    ax.plot(*model_output[experiment_id], label=f"{experiment_id} (簡易モデル)", c=color, path_effects=path_effects, lw=2.5)
    ax.plot(*cmip_dataset[experiment_id], label=f"{experiment_id} ({model_id})", c=color, zorder=0, alpha=0.8, lw=1)
    scenario_ids = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
    for scenario_idx, scenario_id in enumerate(scenario_ids):
        color = colors[scenario_idx]
        ax.plot(*model_output[scenario_id], label=f"{scenario_id} (簡易モデル)", c=color, path_effects=path_effects, lw=2.5)
        ax.plot(*cmip_dataset[scenario_id], label=f"{scenario_id} ({model_id})", c=color, zorder=0, alpha=0.8, lw=1)
    
    for idx, ax in enumerate([ax1, ax2]):
        if idx == 0:
            ax.set_ylabel('地表面気温偏差の全球年平均（K）')
        ax.set_xlabel('年')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title=None, loc='best')
        ax.set_xlim([1840, 2110])
        ax.set_ylim([-0.4, 4.8])
        set_panel_id(ax, letters[idx])
        for posi in ['top', 'right']:
            ax.spines[posi].set_visible(False)

    return fig

def set_panel_id(ax, letter):
    """
    Set the panel identifier on the plot.

    Args:
        ax (matplotlib.axes.Axes): The axes to add the panel identifier to.
        letter (str): The letter identifier for the panel.
    """
    xlim = list(ax.get_xlim())
    ylim = list(ax.get_ylim())
    dx = (xlim[1]-xlim[0])/100
    dy = (ylim[1]-ylim[0])/100
    ax.text(xlim[0]-3*dx, ylim[1]+4*dy, letter, fontsize=16, ha='center', va='center')

def main():

    args = parse_args()
    model_id = args.model_id

    var_id = 'tas'
    variant_label = 'r1i1p1f1'
    
    data_dir = './output'
    calibration_dataset = load_calibration_dataset(model_id, variant_label, data_dir)
    parameters = load_calibrated_parameters(model_id, variant_label, data_dir)
    
    data_dir = "./data_processed"
    cmip_dataset = load_cmip_dataset(var_id, model_id, variant_label, data_dir)
    
    data_dir = './data_raw/RCMIP'
    concentration_dataset = load_concentration_dataset(data_dir, unit_conversion, baseline_concentration)
    forcing_dataset = load_forcing_dataset(data_dir)

    # emulator instance
    emulator = Emulator(parameters)
    emulator.calibration_dataset = calibration_dataset
    emulator.concentration_dataset = concentration_dataset
    emulator.forcing_dataset = forcing_dataset
    emulator.baseline_concentration = baseline_concentration
    
    # model output for plotting
    model_output = {}
    experiment_ids = ['abrupt-4xCO2', 'historical', 'ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
    for experiment_id in experiment_ids:
        t_domain, x_values = emulator.simulate(experiment_id)
        model_output[experiment_id] = (t_domain, x_values[:,1])

    # plot
    fig = plot_figure(model_id, calibration_dataset, cmip_dataset, model_output)
    script_name, _ = os.path.splitext(os.path.basename(__file__))
    output_file = f'./output/fig_{script_name}.svg'
    fig.savefig(output_file)

if __name__ == '__main__':

    main()
