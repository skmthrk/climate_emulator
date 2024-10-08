import os
import csv
from tqdm.auto import tqdm
from tqdm.contrib import tenumerate

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

from utils import make_logger, colors
logger = make_logger()

emission_unit_conversion_rates = {
    'co2': (1e-3 * 1, 'GtCO2 yr-1'), # MtCO2/yr to GtCO2/yr
    'ch4': (1, 'MtCH4 yr-1'), # MtCH4/yr
    'n2o': (1e-3 * 1, 'MtN2O yr-1'), # KtN2O/yr to MtN2O/yr
}
    
concentration_unit_conversion_rates = {
    'co2': (7.82116, 'GtCO2'), # ppm to GtCO2
    'ch4': (2.85094, 'MtCH4'), # ppb to MtCH4
    'n2o': (7.82187, 'MtN2O'), # ppb to MtN2O
}
    
rff_unit_conversion_rates = {
    'co2': (44/12, 'GtCO2 yr-1'), # GtC to GtCO2
    'ch4': (1, 'MtCH4 yr-1'), # MtCH4 to MtCH4
    'n2o': (44/28, 'MtN2O yr-1'), # MtN2 to MtN2O
}
    
var_ids = ['co2', 'ch4', 'n2o']

def build_emissions_dataset():
    '''
    build emissions dataset (SSP scenarios, based on RCMIP data)

    NOTE: only use ssp245 up until 2020
    '''

    # emission
    data_file = "rcmip-emissions-annual-means-v5-1-0.csv"
    data_path = f"./data_raw/RCMIP/{data_file}"
    df_ssp = pd.read_csv(data_path)
    
    emissions_dataset = {}
    year0, year1 = 1750, 2500
    years = np.arange(year0, year1+1)
    scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
    for scenario in scenarios:
        emissions_data = {}
        for var_id in var_ids:
            conversion_rate, _ = emission_unit_conversion_rates[var_id]
            emissions_data[var_id] = (
                years,
                df_ssp.loc[
                    (df_ssp["Region"] == "World")
                    & (df_ssp["Scenario"] == scenario)
                    & (df_ssp["Variable"] == "Emissions|{}".format(var_id.upper())),
                    f"{year0}":f"{year1}",
                ].interpolate(axis=1).values.squeeze()*conversion_rate
            )
        emissions_dataset[scenario] = emissions_data

    for var_id in var_ids:
        _, units = emission_unit_conversion_rates[var_id]
        fig = plt.figure(frameon=False)
        figname = f"./output/fig_rcmip_{var_id}_emission.svg"
        ax = fig.add_subplot(1,1,1, facecolor='none')
        for j, scenario in enumerate(scenarios):
            years, values = emissions_dataset[scenario][var_id]
            ax.plot(years, values, label=scenario, c=colors[j])
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{var_id.upper()} emission ({units})")
        ax.legend()
        fig.set_tight_layout(True)
        fig.savefig(figname)

    return emissions_dataset

def build_concentrations_dataset():
    '''
    build concentrations dataset (SSP scenarios, based on RCMIP data)

    NOTE: only use this dataset for the initial value in building concentratio paths
    '''
    
    # concentration
    data_file = "rcmip-concentrations-annual-means-v5-1-0.csv"
    data_path = f"./data_raw/RCMIP/{data_file}"
    df_ssp = pd.read_csv(data_path)
    
    concentrations_dataset = {}
    year0, year1 = 1750, 2500
    years = np.arange(year0, year1+1)
    scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
    for scenario in scenarios:
        data = {}
        for var_id in var_ids:
            conversion_rate = concentration_unit_conversion_rates[var_id][0]
            data[var_id] = (
                years,
                df_ssp.loc[
                    (df_ssp["Region"] == "World")
                    & (df_ssp["Scenario"] == scenario)
                    & (df_ssp["Variable"] == "Atmospheric Concentrations|{}".format(var_id.upper())),
                    f"{year0}":f"{year1}",
                ].interpolate(axis=1).values.squeeze()*conversion_rate
            )
        concentrations_dataset[scenario] = data
    
    for var_id in var_ids:
        units = concentration_unit_conversion_rates[var_id][1]
        fig = plt.figure(frameon=False)
        figname = f"./output/fig_rcmip_{var_id}_concentration.svg"
        ax = fig.add_subplot(1,1,1, facecolor='none')
        for j, scenario in enumerate(scenarios):
            years, values = concentrations_dataset[scenario][var_id]
            ax.plot(years, values, label=scenario, c=colors[j])
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{var_id.upper()} concentration ({units})")
        ax.legend()
        fig.set_tight_layout(True)
        fig.savefig(figname)

    return concentrations_dataset

def build_forcing_dataset():
    '''
    build forcing dataset (SSP scenarios, based on RCMIP data)

    NOTE: this dataset is not necessary for building concentrations scenarios
    '''
        
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
        figname = f"./output/fig_rcmip_{var_id}_forcing.svg"
        ax = fig.add_subplot(1,1,1, facecolor='none')
        for scenario in scenarios:
            years, values = forcing_dataset[scenario][var_id]
            ax.plot(years, values, label=scenario)
        ax.set_xlabel("Year")
        ax.set_ylabel(f"{var_id.upper()} effective radiative forcing ({units})")
        ax.legend()
        fig.set_tight_layout(True)
        fig.savefig(figname)
    
    units = 'W m-2'
    fig = plt.figure(frameon=False)
    figname = "./output/fig_rcmip_anthropogenic_forcing_non-wmghg.svg"
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
    fig.savefig(figname)

    return forcing_dataset

def build_rff_dataset(sample_max=10000):
    '''
    build future emission dataset (RFFSP)

    NOTE: set sample_max = 1000 or smaller for experiment (faster)
    '''

    # RFF emission scenario
    dir_name = 'data_raw/RFF/emissions'
    rff_dataset = {}
    
    for var_id in var_ids:
    
        file_name = f"rffsp_{var_id}_emissions.csv"
        sample2values = {}
        year2values = {}
        unit_conversion_rate = rff_unit_conversion_rates[var_id][0]
        with open(os.path.join(dir_name, file_name), 'r') as f:
            '''
            sample, year, value
            '''
            reader = csv.reader(f, delimiter=',')
            for i, lst in tenumerate(reader, desc=f'loading {var_id} rffsp data'):
                if i == 0:
                    continue
                sample, year, value = lst
                sample = int(sample)
                year = int(year)
                value = float(value) * unit_conversion_rate
                sample2values.setdefault(sample, []).append(value)
                year2values.setdefault(year, []).append(value)
                if sample >= sample_max:
                    break
        
        years = np.array(sorted(list(year2values.keys())))
        rff_dataset[var_id] = (years, sample2values)
    
    return rff_dataset

def build_concentration_scenarios(emissions_dataset, concentrations_dataset, rff_dataset, pulse_year=2020, pulse_size=0):
    """
    """

    if pulse_size == 0:
        pulse_dir = "nopulse"
    else:
        pulse_dir = f"pulse_{pulse_size}_{pulse_year}"

    def pulse(t, var_id):
        if t >= pulse_year and t < pulse_year+1:
            return pulse_size
        return 0

    # equilibrium concentration values
    xbars = {
        'co2': 278 * concentration_unit_conversion_rates['co2'][0], # GtCO2
        'ch4': 720 * concentration_unit_conversion_rates['ch4'][0], # MtCH4
        'n2o': 270 * concentration_unit_conversion_rates['n2o'][0], # MtN2O
    }
    
    out_dir0 = 'output'
    try:
        os.mkdir(out_dir0)
    except FileExistsError:
        print("Directory {} already exists".format(out_dir0))
    try:
        os.mkdir(os.path.join(out_dir0, pulse_dir))
    except FileExistsError:
        print("Directory {} already exists".format(os.path.join(out_dir0, pulse_dir)))

    out_dirs = {var_id: os.path.join(out_dir0, pulse_dir, var_id) for var_id in var_ids}
    for var_id in var_ids:
        out_dir = out_dirs[var_id]
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            print("Directory {} already exists".format(out_dir))

    # construct emissions/concentrations scenario
    
    i, var_id = 0, 'co2'
    out_dir = out_dirs[var_id]
    
    joos_scenario_id = 'PI5000'
    with open(f'./output/parameter_co2_cycle_nonlinear_{joos_scenario_id}.csv', 'r') as f:
         gammas = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])
    
    with open(f'./output/parameter_co2_cycle_linear_{joos_scenario_id}.csv', 'r') as f:
         deltas = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])

    def build_carbon_cycle_matrix(deltas=deltas):
        """ convert parameter list into carbon cycle matrix
        """
        [delta21, delta31, delta12, delta32, delta13, delta43, delta34] = deltas
        D = np.zeros((4,4))
        D[1-1,1-1] = -delta21 - delta31
        D[2-1,1-1] = delta21
        D[3-1,1-1] = delta31
        D[1-1,2-1] = delta12
        D[2-1,2-1] = -delta12 - delta32
        D[3-1,2-1] = delta32
        D[1-1,3-1] = delta13
        D[3-1,3-1] = -delta13 - delta43
        D[4-1,3-1] = delta43
        D[3-1,4-1] = delta34
        D[4-1,4-1] = -delta34
        return D
    
    def dMdt(x, t, deltas, gammas, u):
        M1, M2, M3, M4 = x
        M = sum(x)
        gamma0, gamma1 = gammas
        D = build_carbon_cycle_matrix(deltas)/np.exp(gamma0 + gamma1*M)
        dM1dt = D[0,0]*M1 + D[0,1]*M2 + D[0,2]*M3 + D[0,3]*M4 + u(t)
        dM2dt = D[1,0]*M1 + D[1,1]*M2 + D[1,2]*M3 + D[1,3]*M4
        dM3dt = D[2,0]*M1 + D[2,1]*M2 + D[2,2]*M3 + D[2,3]*M4
        dM4dt = D[3,0]*M1 + D[3,1]*M2 + D[3,2]*M3 + D[3,3]*M4
        return [dM1dt, dM2dt, dM3dt, dM4dt]
    
    def forcing(u, phi, zeta):
        if zeta == 0:
            return phi * np.log(u)
        return phi * (1/zeta)*(u**zeta - 1)
    
    rff_years, sample2values = rff_dataset[var_id]
    unit = rff_unit_conversion_rates[var_id][1]
    var_name = var_id.upper()
    samples = sorted(list(sample2values.keys()))
    
    scenario = 'ssp245'
    ssp_years, ssp_emission_values = emissions_dataset[scenario][var_id]
    _, ssp_concentration_values = concentrations_dataset[scenario][var_id]
    #_, ssp_forcing_values = forcing_dataset[scenario][var_id]
    
    idx = np.where(ssp_years == rff_years[0])[0][0]
    hist_years = ssp_years[:idx]
    years = np.append(hist_years, rff_years)
    
    hist_emission_values = ssp_emission_values[:idx]
    hist_concentration_values = ssp_concentration_values[:idx]
    #hist_forcing_values = ssp_forcing_values[:idx]

    # initial concentration of carbon reserviors (deviation from equilibrium point)
    x_init = [hist_concentration_values[0] - xbars[var_id], 0, 0, 0]
    
    for sample in tqdm(samples, desc=f"building {var_id} concentration"):

        # baseline emission (w/o pulse)
        emission_values = np.append(hist_emission_values, sample2values[sample])

        # add emission pulse if any
        emission_pulse_values = np.zeros(len(years))
        for i, year in enumerate(years):
            if year == 2020:
                emission_pulse_values[i] = pulse_size

        with open(os.path.join(out_dir, f'{var_id}_emission_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in emission_values+emission_pulse_values))

        # generate concentration path based on emission path and the estimated carbon cycle model (dMdt)
        u_emission_raw = CubicSpline(years, emission_values, extrapolate=True)
        def u_emission(t): return u_emission_raw(t) + pulse(t, var_id)
        x = odeint(dMdt, x_init, years, args=(deltas, gammas, u_emission))
        concentration_values = x[:,0] + xbars[var_id]
    
        with open(os.path.join(out_dir, f'{var_id}_concentration_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in concentration_values))

        # convert concentration into forcing based on the estimated forcing model (phi, zeta)
        with open(f"./output/parameter_{var_id}_forcing.csv", 'r') as f:
            phi, zeta = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])
        forcing_values = forcing(concentration_values, phi, zeta) - forcing(xbars[var_id], phi, zeta)
        
        with open(os.path.join(out_dir, f'{var_id}_forcing_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in forcing_values))

    # CH4 and N2O
    
    def build_gas_cycle_matrix(deltas):
        """ convert parameter list into gas cycle matrix Delta
        """
        [delta11, delta21, delta31, delta12, delta22, delta32, delta13, delta23, delta33] = deltas
        D = np.zeros((4,4))
        D[1-1,1-1] = delta11
        D[2-1,1-1] = delta21
        D[3-1,1-1] = delta31
        D[1-1,2-1] = delta12
        D[2-1,2-1] = delta22
        D[3-1,2-1] = delta32
        D[1-1,3-1] = delta13
        D[2-1,3-1] = delta23
        D[3-1,3-1] = delta33
        return D
    
    def dxdt(x, t, deltas, u):
        x1, x2, x3 = x
        D = build_gas_cycle_matrix(deltas)
        dx1dt = D[0,0]*x1 + D[0,1]*x2 + D[0,2]*x3 + u(t)
        dx2dt = D[1,0]*x1 + D[1,1]*x2 + D[1,2]*x3
        dx3dt = D[2,0]*x1 + D[2,1]*x2 + D[2,2]*x3
        return [dx1dt, dx2dt, dx3dt]
    
    for j, var_id in enumerate(['ch4', 'n2o']):
        i = j+1
        out_dir = out_dirs[var_id]
    
        with open(f'./output/parameter_{var_id}_cycle.csv', 'r') as f:
            deltas = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])
        
        rff_years, sample2values = rff_dataset[var_id]
        unit = rff_unit_conversion_rates[var_id][1]
        var_name = var_id.upper()
        samples = sorted(list(sample2values.keys()))
        
        scenario = 'ssp245'
        ssp_years, ssp_emission_values = emissions_dataset[scenario][var_id]
        _, ssp_concentration_values = concentrations_dataset[scenario][var_id]
        #_, ssp_forcing_values = forcing_dataset[scenario][var_id]
        
        idx = np.where(ssp_years == rff_years[0])[0][0]
        hist_years = ssp_years[:idx]
        years = np.append(hist_years, rff_years)
        
        hist_emission_values = ssp_emission_values[:idx]
        hist_concentration_values = ssp_concentration_values[:idx]
        #hist_forcing_values = ssp_forcing_values[:idx]
        
        x_init = [hist_concentration_values[0] - xbars[var_id], 0, 0]
        
        for sample in tqdm(samples, desc=f"building {var_id} concentration"):
            emission_values = np.append(hist_emission_values, sample2values[sample])

            emission_pulse_values = np.zeros(len(years))
            for i, year in enumerate(years):
                if year == 2020:
                    emission_pulse_values[i] = pulse_size

            with open(os.path.join(out_dir, f'{var_id}_emission_sample_{sample}.csv'), 'w') as f:
                f.write(','.join(str(year) for year in years))
                f.write('\n')
                f.write(','.join(str(value) for value in emission_values+emission_pulse_values))

            u_emission_raw = CubicSpline(years, emission_values, extrapolate=True)
            def u_emission(t): return u_emission_raw(t) + pulse(t, var_id)
            x = odeint(dxdt, x_init, years, args=(deltas, u_emission))
            concentration_values = x[:,0] + xbars[var_id]
        
            with open(os.path.join(out_dir, f'{var_id}_concentration_sample_{sample}.csv'), 'w') as f:
                f.write(','.join(str(year) for year in years))
                f.write('\n')
                f.write(','.join(str(value) for value in concentration_values))

            with open(f"./output/parameter_{var_id}_forcing.csv", 'r') as f:
                phi, zeta = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])
            forcing_values = forcing(concentration_values, phi, zeta) - forcing(xbars[var_id], phi, zeta)
            
            with open(os.path.join(out_dir, f'{var_id}_forcing_sample_{sample}.csv'), 'w') as f:
                f.write(','.join(str(year) for year in years))
                f.write('\n')
                f.write(','.join(str(value) for value in forcing_values))

def main():
    emissions_dataset = build_emissions_dataset()
    concentrations_dataset = build_concentrations_dataset()
    forcing_dataset = build_forcing_dataset()
    rff_dataset = build_rff_dataset(sample_max=100)
    for pulse_size in [0, 1]:
        build_concentration_scenarios(emissions_dataset, concentrations_dataset, rff_dataset, pulse_size=pulse_size)

if __name__ == '__main__':
    main()
