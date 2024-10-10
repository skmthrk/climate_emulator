import os
import sys
import csv
from glob import glob

from tqdm.auto import tqdm
import multiprocessing as mp

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

colors = [
    '#348ABD', # 0: blue
    '#7A68A6', # 1: purple
    '#A60628', # 2: red
    '#467821', # 3: green
    '#188487', # 4: breen
    '#CF4457', # 5: pink
    '#E24A33', # 6: orange
]

var_ids = ['co2', 'ch4', 'n2o']

def load_forcing_dataset():
    """
    """
        
    # forcing
    data_file = "rcmip-radiative-forcing-annual-means-v5-1-0.csv"
    data_path = f"./data_raw/RCMIP/{data_file}"
    df_ssp = pd.read_csv(data_path)
    
    forcing_dataset = {}
    year0, year1 = 1750, 2500
    years = np.arange(year0, year1+1)
    scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
    for scenario in scenarios:
        data = {}
        for var_id in var_ids:
            data[var_id] = (
                years,
                df_ssp.loc[
                    (df_ssp["Region"] == "World")
                    & (df_ssp["Scenario"] == scenario)
                    & (df_ssp["Variable"] == "Effective Radiative Forcing|Anthropogenic|{}".format(var_id.upper())),
                    f"{year0}":f"{year1}",
                ].interpolate(axis=1).values.squeeze()
            )
        for var_id in ['anthropogenic', 'natural']:
            data[var_id] = (
                years,
                df_ssp.loc[
                    (df_ssp["Region"] == "World")
                    & (df_ssp["Scenario"] == scenario)
                    & (df_ssp["Variable"] == "Effective Radiative Forcing|{}".format(var_id.capitalize())),
                    f"{year0}":f"{year1}",
                ].interpolate(axis=1).values.squeeze()
            )
        data['total'] = (
            years,
            df_ssp.loc[
                (df_ssp["Region"] == "World")
                & (df_ssp["Scenario"] == scenario)
                & (df_ssp["Variable"] == "Effective Radiative Forcing"),
                f"{year0}":f"{year1}",
            ].interpolate(axis=1).values.squeeze()
        )
        forcing_dataset[scenario] = data

    var_ids_all = var_ids + ['anthropogenic', 'natural', 'total']
    for var_id in var_ids_all:
        units = 'W m-2'
        fig = plt.figure(frameon=False)
        figname = f"fig_{var_id}_forcing.svg"
        ax = fig.add_subplot(1,1,1, facecolor='none')
        for scenario in scenarios:
            years, values = forcing_dataset[scenario][var_id]
            ax.plot(years, values, label=scenario)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{var_id.upper()} effective radiative forcing ({units})")
        ax.legend()
        fig.set_tight_layout(True)
        fig.savefig(f'./output/{figname}')
    
    units = 'W m-2'
    fig = plt.figure(frameon=False)
    figname = "fig_anthropogenic_forcing_non-wmghg.svg"
    ax = fig.add_subplot(1,1,1, facecolor='none')
    for scenario in scenarios:
        years = forcing_dataset[scenario]["anthropogenic"][0]
        values = (
            forcing_dataset[scenario]["anthropogenic"][1]
            - forcing_dataset[scenario]["co2"][1]
            - forcing_dataset[scenario]["ch4"][1]
            - forcing_dataset[scenario]["n2o"][1]
            )
        ax.plot(years, values, label=scenario)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Anthropogenic non-wmghg effective radiative forcing ({units})")
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig(f'./output/{figname}')

    return forcing_dataset

def build_A(parameters):
    """ convert parameter list into matrix A
    """
    gamma = parameters['gamma']
    chi1 = parameters['chi1']
    chi2 = parameters['chi2']
    kappa1 = parameters['kappa1']
    kappa2 = parameters['kappa2']
    epsilon = parameters['epsilon']

    A = np.zeros((3,3))

    # energy balance model
    A[1-1,1-1] = -gamma
    A[2-1,1-1] = 1/chi1
    A[2-1,2-1] = -(kappa1 + epsilon + kappa2)/chi1
    A[2-1,3-1] = (kappa2 + epsilon)/chi1
    A[3-1,2-1] = kappa2/chi2
    A[3-1,3-1] = -kappa2/chi2
    return A

def dxdt(x, t, u, parameters):
    A = build_A(parameters)
    gamma = parameters["gamma"]
    F, T1, T2 = x
    dFdt = A[0,0]*F + A[0,1]*T1 + A[0,2]*T2 + gamma*u(t)
    dT1dt = A[1,0]*F + A[1,1]*T1 + A[1,2]*T2
    dT2dt = A[2,0]*F + A[2,1]*T1 + A[2,2]*T2
    return [dFdt, dT1dt, dT2dt]

def load_cmip_data(cmip_data_dir_path, file_name_pattern):
    years = []
    vals = []
    file_paths = sorted(glob(os.path.join(cmip_data_dir_path, file_name_pattern)))
    for file_num, file_path in enumerate(file_paths):
        print("===> Processing", file_path)
        with open(file_path, 'r') as csvfile:
            f = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, t in enumerate(f):
                if i == 0:
                    continue
                years.append(int(t[0]))
                vals.append(float(t[1]))
    return np.array(years), np.array(vals)

def build_temperature_scenarios(forcing_dataset, pulse_size=0, pulse_year=2020, pulse_var=None, model_id="MIROC6", variant_label='r1i1p1f1'):
#    print('model_id:', model_id)
#    print('variant_label:', variant_label)

    if pulse_var is None or pulse_size == 0:
        pulse_size = 0
        pulse_var = None
        pulse_dir = "nopulse"
    else:
        if pulse_var not in ['co2', 'ch4', 'n2o']:
            print("ERROR: pulse_var must be either 'co2', 'ch4', or 'n2o'")
            sys.exit()
        pulse_dir = f"pulse_{pulse_size}_{pulse_year}"

    cmip_data_dir_path = "./data_processed/processed"
    var_id= 'tas'

    # historical experiment data
    experiment_id = 'historical'
    file_name_pattern = f"{var_id}_{model_id}_{experiment_id}_{variant_label}*.csv"
    hist_years, hist_vals = load_cmip_data(cmip_data_dir_path, file_name_pattern)
    #hist_val0 = hist_vals[0]
    #hist_values = hist_vals - hist_val0

    # rff forcing
    scenario = 'ssp245'
    ssp_years = forcing_dataset[scenario]["anthropogenic"][0]
    ssp_values = (
        forcing_dataset[scenario]["total"][1]
        - forcing_dataset[scenario]["co2"][1]
        - forcing_dataset[scenario]["ch4"][1]
        - forcing_dataset[scenario]["n2o"][1]
    )

    idx1 = np.where(ssp_years == 2300)[0][0]
    years = ssp_years[:idx1+1]
    forcing_values_non_wmghg = ssp_values[:idx1+1]

    # load parameter values estimated
    experiment_id = 'abrupt-4xCO2'
    file_name = f"parameter_{model_id}_{experiment_id}_{variant_label}.csv"
    file_path = os.path.join('./output', file_name)
    parameters = {}
    with open(file_path, 'r') as csvfile:
        f = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, t in enumerate(f):
            if i == 0:
                continue
            key, val = t
            parameters[key] = float(val)

    if pulse_var:
        tas_dir = f"tas_under_{pulse_var}_pulse_{model_id}"
    else:
        tas_dir = f"tas_{model_id}"

    out_dir0 = './output'
    out_dir = os.path.join(out_dir0, pulse_dir, tas_dir)
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        #print("Directory {} already exists".format(out_dir))
        pass
        #return None

    num_samples = 10000
    samples = range(1, num_samples+1)
    for sample in tqdm(samples, desc=f"temperature under {pulse_var} pulse"):
        forcing_values = forcing_values_non_wmghg.copy()
        for var_id in var_ids:
            if var_id == pulse_var:
                data_path = os.path.join(out_dir0, pulse_dir, var_id)
            else:
                data_path = os.path.join(out_dir0, 'nopulse', var_id)
            with open(os.path.join(data_path, f"{var_id}_forcing_sample_{sample}.csv"), 'r') as f:
                reader = csv.reader(f, delimiter=',')
                years = np.array(reader.__next__()).astype(int)
                forcing_values += np.array(reader.__next__()).astype(float)

        u_forcing = CubicSpline(years, forcing_values) # error
        t_list = np.arange(years[0], years[-1]+1, 1)
        x_init = [u_forcing(t_list[0]), 0, 0]
        x = odeint(dxdt, x_init, t_list, args=(u_forcing, parameters))
    
        with open(os.path.join(out_dir, f'total_forcing_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in forcing_values))
        with open(os.path.join(out_dir, f'tas_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in t_list))
            f.write('\n')
            f.write(','.join(str(value) for value in x[:,1]))

def process_func(args):
    '''
    function for multiprocessing
    '''
    build_temperature_scenarios(*args)

def main():
    model_ids = [
        'MIROC6',
#        'CanESM5',
#        "ACCESS-CM2",
#        "BCC-CSM2-MR",
#        "CESM2",
#        "CMCC-CM2-SR5",
#        "CNRM-CM6-1",
#        "FGOALS-f3-L",
#        "GISS-E2-1-G",
#        "HadGEM3-GC31-LL",
#        "INM-CM5-0",
#        "IPSL-CM6A-LR",
#        "KACE-1-0-G",
#        "MPI-ESM1-2-LR",
#        "MRI-ESM2-0",
#        "NorESM2-LM",
    ]
    variant_label = 'r1i1p1f1'

    forcing_dataset = load_forcing_dataset()
    pulse_vars = [None, 'co2', 'ch4', 'n2o']
    args_list = [(forcing_dataset, 1, 2020, pulse_var, model_id, variant_label) for pulse_var in pulse_vars for model_id in model_ids]

    # parallel processing
    with mp.Pool() as pool:
        pool.map(process_func, args_list)

if __name__ == '__main__':
    main()
