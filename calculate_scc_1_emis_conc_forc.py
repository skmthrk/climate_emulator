import os
import csv

from tqdm.auto import tqdm
import multiprocessing as mp

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

colors = [
    '#348ABD', # blue
    '#188487', # breen
    '#E24A33', # orenge
    '#7A68A6', # purple
    '#A60628', # red
    '#467821', # green
    '#CF4457', # pink
]

class ConcentrationScenarioBuilder:

    def __init__(self):

        self.data_dir = './data_raw'
        self.output_base_dir = './output'
        self.output_dirs = {}
        self.scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
        self.base_scenario = 'ssp245'
        self.years = list(range(1750, 2501))
        self.var_ids = ['co2', 'ch4', 'n2o']
        self.var_types = ['emission', 'concentration']
        self.unit_conversion_rates = {
            'emission': {
                'co2': (1e-3 * 1, 'GtCO2 yr-1'), # MtCO2/yr to GtCO2/yr
                'ch4': (1, 'MtCH4 yr-1'), # MtCH4/yr
                'n2o': (1e-3 * 1, 'MtN2O yr-1'), # KtN2O/yr to MtN2O/yr
            },
            'concentration': {
                'co2': (7.82116, 'GtCO2'), # ppm to GtCO2
                'ch4': (2.85094, 'MtCH4'), # ppb to MtCH4
                'n2o': (7.82187, 'MtN2O'), # ppb to MtN2O
            },
            'rff': {
                'co2': (44/12, 'GtCO2 yr-1'), # GtC to GtCO2
                'ch4': (1, 'MtCH4 yr-1'), # MtCH4 to MtCH4
                'n2o': (44/28, 'MtN2O yr-1'), # MtN2 to MtN2O
            },
        }
        self.xbars = {
            'co2': 278 * self.unit_conversion_rates['concentration']['co2'][0], # GtCO2
            'ch4': 720 * self.unit_conversion_rates['concentration']['ch4'][0], # MtCH4
            'n2o': 270 * self.unit_conversion_rates['concentration']['n2o'][0], # MtN2O
        }

        self.datasets = {}

    def load_dataset(self, var_type):
        '''
        load emissions/concentration dataset (SSP scenarios, based on RCMIP data)
    
        NOTE: only ssp245 emissin data useed up until 2020
        NOTE: only ssp245 concentration data useed for the initial state value
        '''
    
        if var_type == 'emission':
            data_file = 'rcmip-emissions-annual-means-v5-1-0.csv'
            unit_conversion_rates = self.unit_conversion_rates[var_type]
            rcmip_var_name = 'Emissions'
        elif var_type == 'concentration':
            data_file = 'rcmip-concentrations-annual-means-v5-1-0.csv'
            unit_conversion_rates = self.unit_conversion_rates[var_type]
            rcmip_var_name = 'Atmospheric Concentrations'
            
        # load raw data
        data_path = os.path.join(self.data_dir, f"RCMIP/{data_file}")
        df_ssp = pd.read_csv(data_path)
        years = self.years

        # process data
        dataset = {}
        for scenario in self.scenarios:
            data = {}
            for var_id in self.var_ids:
                data[var_id] = (
                    years,
                    df_ssp.loc[
                        (df_ssp["Region"] == "World")
                        & (df_ssp["Scenario"] == scenario)
                        & (df_ssp["Variable"] == f"{rcmip_var_name}|{var_id.upper()}"),
                        f"{years[0]}":f"{years[-1]}",
                    ].interpolate(axis=1).values.squeeze()*unit_conversion_rates[var_id][0]
                )
            dataset[scenario] = data

        self.datasets[var_type] = dataset

    def plot_dataset(self, var_type):

        # load dataset if not done yet
        if var_type not in self.datasets:
            self.load_dataset(var_type)
        dataset = self.datasets[var_type]

        # plot
        unit_conversion_rates = self.unit_conversion_rates[var_type]
        for var_id in self.var_ids:
            units = unit_conversion_rates[var_id][1]
            figname = f"./output/fig_rcmip_{var_id}_{var_type}.svg"
            ylabel = f"{var_id.upper()} {var_type} ({units})"

            fig = plt.figure(frameon=False)
            ax = fig.add_subplot(1,1,1, facecolor='none')
            for idx, scenario in enumerate(self.scenarios):
                years, values = dataset[scenario][var_id]
                ax.plot(years, values, label=scenario, c=colors[idx])
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel)
            ax.legend()
            fig.set_tight_layout(True)
            fig.savefig(figname)

    def load_rff_emission_scenario(self, sample_cutoff=None):
        '''
        load future emission scenario (RFF socioeconomic projections)
        '''
    
        # RFF emission scenario
        dir_path = os.path.join(self.data_dir, 'RFF/emissions')
        dataset = {}
        unit_conversion_rates = self.unit_conversion_rates['rff']
        
        for var_id in self.var_ids:
        
            file_name = f"rffsp_{var_id}_emissions.csv"
            sample2values = {} # for generating sample paths
            years = set()
            unit_conversion_rate = unit_conversion_rates[var_id][0]
            with open(os.path.join(dir_path, file_name), 'r') as f:
                '''
                raw data format: sample, year, value
                '''
                reader = csv.reader(f, delimiter=',')
                reader.__next__() # skip the first line
                for lst in tqdm(reader, desc=f'loading {var_id} rffsp data'):
                    sample, year, value = lst

                    sample = int(sample)
                    year = int(year)
                    value = float(value) * unit_conversion_rate

                    if sample_cutoff and sample > sample_cutoff:
                        continue

                    sample2values.setdefault(sample, []).append(value)
                    years.add(year)

            years = np.array(sorted(list(years)))
            dataset[var_id] = (years, sample2values)

        self.datasets['rff'] = dataset

    def make_directory(self, dir_path):
        try:
            os.mkdir(dir_path)
        except FileExistsError:
            pass
            #print("Directory {} already exists".format(dir_path))
        

    def make_output_directories(self, var_id, pulse_size, pulse_year=2020):
        
        if pulse_size == 0:
            pulse_dir_name = "nopulse"
        else:
            pulse_dir_name = f"pulse_{pulse_size}_{pulse_year}"

        output_base_dir = self.output_base_dir
        self.make_directory(output_base_dir)

        pulse_dir = os.path.join(output_base_dir, pulse_dir_name)
        self.make_directory(pulse_dir)
    
        output_dir = os.path.join(pulse_dir, var_id)
        self.make_directory(output_dir)
        self.output_dirs[var_id] = output_dir

    def build_concentration_scenarios(self, var_id, pulse_size, pulse_year):

        # emission pulse function
        def pulse(t, pulse_size=pulse_size, pulse_year=pulse_year):
            if t >= pulse_year and t < pulse_year+1:
                return pulse_size
            return 0
    
        self.make_output_directories(var_id, pulse_size, pulse_year)
        out_dir = self.output_dirs[var_id]

        # equilibrium concentration values
        xbar = self.xbars[var_id]

        if var_id == 'co2':
            # load parameters for carbon cycle model
            joos_scenario_id = 'PI5000'
            with open(f'./output/parameter_{var_id}_cycle_linear_{joos_scenario_id}.csv', 'r') as f:
                 deltas = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])
            with open(f'./output/parameter_{var_id}_cycle_nonlinear_{joos_scenario_id}.csv', 'r') as f:
                 gammas = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])

            def build_gas_cycle_matrix(deltas):
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
            
            def dxdt(x, t, u, deltas=deltas, gammas=gammas):
                x1, x2, x3, x4 = x
                gamma0, gamma1 = gammas
                D = build_gas_cycle_matrix(deltas)/np.exp(gamma0 + gamma1*sum(x))
                dx1dt = D[0,0]*x1 + D[0,1]*x2 + D[0,2]*x3 + D[0,3]*x4 + u(t)
                dx2dt = D[1,0]*x1 + D[1,1]*x2 + D[1,2]*x3 + D[1,3]*x4
                dx3dt = D[2,0]*x1 + D[2,1]*x2 + D[2,2]*x3 + D[2,3]*x4
                dx4dt = D[3,0]*x1 + D[3,1]*x2 + D[3,2]*x3 + D[3,3]*x4
                return [dx1dt, dx2dt, dx3dt, dx4dt]
        
        else:
            with open(f'./output/parameter_{var_id}_cycle.csv', 'r') as f:
                deltas = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])

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
            
            def dxdt(x, t, u, deltas=deltas):
                x1, x2, x3 = x
                D = build_gas_cycle_matrix(deltas)
                dx1dt = D[0,0]*x1 + D[0,1]*x2 + D[0,2]*x3 + u(t)
                dx2dt = D[1,0]*x1 + D[1,1]*x2 + D[1,2]*x3
                dx3dt = D[2,0]*x1 + D[2,1]*x2 + D[2,2]*x3
                return [dx1dt, dx2dt, dx3dt]

        with open(f"./output/parameter_{var_id}_forcing.csv", 'r') as f:
            phi, zeta = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])

        def forcing(x, phi=phi, zeta=zeta):
            if zeta == 0:
                return phi * np.log(x)
            return phi * (1/zeta)*(x**zeta - 1)
        
        scenario = self.base_scenario
        ssp_years, ssp_emission_values = self.datasets['emission'][scenario][var_id]
        _, ssp_concentration_values = self.datasets['concentration'][scenario][var_id]
        
        rff_years, sample2values = self.datasets['rff'][var_id]
        unit = self.unit_conversion_rates['rff'][var_id][1]
        var_name = var_id.upper()
        samples = sorted(list(sample2values.keys()))

        # stitch ssp and rff data
        stitch_idx = np.where(ssp_years == rff_years[0])[0][0]
        years = np.append(ssp_years[:stitch_idx], rff_years)
        historical_emission_values = ssp_emission_values[:stitch_idx]

        # initial concentration of carbon reserviors (deviation from equilibrium point)
        if var_id == 'co2':
            x_init = [ssp_concentration_values[0] - xbar, 0, 0, 0]
        else:
            x_init = [ssp_concentration_values[0] - xbar, 0, 0]

        figname = f"./output/fig_{var_id}_samples.png"
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), frameon=False)
        ax_dct = {
            'emission': (axes[0], colors[0]),
            'concentration': (axes[1], colors[1]),
        }
        axes[0].plot(ssp_years, ssp_emission_values, c='k', label=scenario)
        axes[1].plot(ssp_years, ssp_concentration_values, c='k', label=scenario)
    
        for sample in tqdm(samples, desc=f"building {var_id} concentration"):
    
            # baseline emission (w/o pulse)
            emission_values = np.append(historical_emission_values, sample2values[sample])
    
            # add emission pulse if any
            emission_pulse_values = np.zeros(len(years))
            for idx, year in enumerate(years):
                if year == pulse_year:
                    emission_pulse_values[idx] = pulse_size
    
            # generate concentration path based on emission path and the estimated carbon cycle model (dxdt)
            u_emission_raw = CubicSpline(years, emission_values, extrapolate=True)
            def u_emission(t): return u_emission_raw(t) + pulse(t)
            x = odeint(dxdt, x_init, years, args=(u_emission,))
            concentration_values = x[:,0] + xbar
    
            # convert concentration into forcing based on the estimated forcing model (phi, zeta)
            forcing_values = forcing(concentration_values) - forcing(xbar)

            # save results
            for var_type in self.var_types:

                if var_type == 'emission':
                    values = emission_values + emission_pulse_values
                else:
                    values = concentration_values

                with open(os.path.join(out_dir, f'{var_id}_{var_type}_sample_{sample}.csv'), 'w') as f:
                    f.write(','.join(str(year) for year in years))
                    f.write('\n')
                    f.write(','.join(str(value) for value in values))

                if pulse_size == 0:
                    ax, color = ax_dct[var_type]
                    ax.plot(years, values, c=color, alpha=0.1, lw=0.5)

            with open(os.path.join(out_dir, f'{var_id}_forcing_sample_{sample}.csv'), 'w') as f:
                f.write(','.join(str(year) for year in years))
                f.write('\n')
                f.write(','.join(str(value) for value in forcing_values))


        if pulse_size == 0:
            for var_type in self.var_types:
                ax, _ = ax_dct[var_type]
                ax.set_xlabel("Year")
                unit = self.unit_conversion_rates[var_type][var_id][1]
                ax.set_ylabel(f"{var_id.upper()} {var_type} ({unit})")
                ax.set_facecolor('none')
                for posi in ['top', 'right']:
                    ax.spines[posi].set_visible(False)
                ax.legend()
            fig.tight_layout()
            fig.savefig(figname, dpi=300)

def build_func(args):
    builder, var_id, pulse_size, pulse_year = args
    builder.build_concentration_scenarios(var_id, pulse_size, pulse_year)

def main():

    builder = ConcentrationScenarioBuilder()

    for var_type in builder.var_types:
        builder.load_dataset(var_type)
        builder.plot_dataset(var_type)

    sample_cutoff = None
    builder.load_rff_emission_scenario(sample_cutoff=sample_cutoff)

    # build scenarios
    args_list = []
    for var_id in builder.var_ids:
        for pulse_size in [0, 1]:
            for pulse_year in [2020]:
                args_list.append(
                    (builder, var_id, pulse_size, pulse_year)
                )
    try:
        # parallel processing
        with mp.Pool() as pool:
            pool.map(build_func, args_list)
    except:
        print('Parallel processing failed')
        for args in args_list:
            build_func(args)

if __name__ == '__main__':
    main()
