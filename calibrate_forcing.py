import os

import pandas as pd
import numpy as np
from scipy import optimize
import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt

from utils import make_logger, colors
logger = make_logger()

matplotlib.rc('axes', lw=0.25, edgecolor='k')

class ForcingModel:

    def __init__(self, data_dir='./data_raw/RCMIP'):

        self.data_dir = data_dir
        self.years = list(range(1750, 2501))
        self.scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
        self.var_ids = ['co2', 'ch4', 'n2o']
        self.ubars = {'co2': 278*7.82116, 'ch4': 720*2.85094, 'n2o': 270*7.82187}
        self.var_types = ['concentration', 'forcing']
        self.methods = ['Nelder-Mead', 'Powell', 'SLSQP', 'TNC', 'COBYLA', 'BFGS', 'Newton-CG', 'L-BFGS-B']
        self.datasets = self._build_datasets()

    def _build_datasets(self):
        """ load RCMIP dataset
        var_type: concentration', 'forcing'
        var_ids: 'co2', 'ch4', 'n2o'
        scenarios: 'ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585'
        """
        datasets = {}
        for var_type in self.var_types:
            if var_type == 'forcing':
                data_file = "rcmip-radiative-forcing-annual-means-v5-1-0.csv"
                var_name = 'Effective Radiative Forcing|Anthropogenic'
                unit_conversion = {var_id: 1 for var_id in self.var_ids}
            elif var_type == 'concentration':
                data_file = "rcmip-concentrations-annual-means-v5-1-0.csv"
                var_name = 'Atmospheric Concentrations'
                unit_conversion = {
                    'co2': 7.82116, # ppm to GtCO2
                    'ch4': 2.85094, # ppb to MtCH4
                    'n2o': 7.82187, # ppb to MtN2O
                }

            data_path = os.path.join(self.data_dir, data_file)
            df_ssp = pd.read_csv(data_path)
            dataset = {}
            for scenario in self.scenarios:
                data = {}
                for var_id in self.var_ids:
                    data[var_id] = df_ssp.loc[
                        (df_ssp["Region"] == "World")
                        & (df_ssp["Scenario"] == scenario)
                        & (df_ssp["Variable"] == f"{var_name}|{var_id.upper()}"),
                        "1750":"2500",
                    ].interpolate(axis=1).values.squeeze() * unit_conversion.get(var_id,1)
                dataset[scenario] = data
            datasets[var_type] = dataset
        return datasets

    @staticmethod
    def forcing_func(u, parameters):
        alpha, theta = parameters
        if theta == 0:
            return alpha * np.log(u)
        return alpha * (1/theta)*(u**theta - 1)

    def loss_func(self, parameters, var_id):
        loss = 0
        for scenario in self.scenarios:
            x_obs = self.datasets['forcing'][scenario][var_id]
            u = self.datasets['concentration'][scenario][var_id]
            ubar = self.ubars[var_id]
            x_vals = self.forcing_func(u, parameters) - self.forcing_func(ubar, parameters)
            loss += np.linalg.norm(x_vals - x_obs)
        print(loss)
        return loss

# xxxxx Thu Aug 29 16:51:44 JST 2024

for var_id in ['co2', 'ch4', 'n2o']:
    print('===', var_id)

    units = 'W m-2'
    parameters = [1, 0]
    methods = ['Nelder-Mead', 'Powell', 'SLSQP', 'TNC', 'COBYLA', 'BFGS', 'Newton-CG', 'L-BFGS-B']
    method = methods[0]
    if var_id == 'co2':
        bounds = [(None, None), (0, 0)]
    else:
        bounds = [(None, None), (None, None)]
    tol = 1e-11
    maxiter = 5000
    res = optimize.minimize(fun=lambda parameters:Loss(parameters, var_id=var_id), x0=parameters, method=method, bounds=bounds, tol=tol, options={'maxiter': maxiter})
    print(' method:', method)
    print(' message:', res.message)
    print(' nit:', res.nit)
    print(' status:', res.status)
    print(' success:', res.success)
    print(' min:', res.fun)
    print(' minimizer:', list(res.x))

    with open(f"./output/parameter_{var_id}_forcing.csv", 'w') as f:
        keys = ['var_id', 'alpha', 'theta']
        vals = [var_id] + [str(para) for para in res.x]
        f.write(','.join(keys))
        f.write('\n')
        f.write(','.join(vals))

    alpha, theta = res.x

    fig = plt.figure(frameon=False)
    figname = f"./output/fig_{var_id}_forcing.svg"
    ax = fig.add_subplot(1,1,1, facecolor='none')
    conc_conversion_rate, _ = concentration_unit_conversion_rates[var_id]
    for j, scenario in enumerate(scenarios):
        x_obs = forcing_dataset[scenario][var_id]
        ax.plot(years, x_obs, label=f'{scenario}', c=colors[j], ls='--')
    
        u = concentrations_dataset[scenario][var_id]*conc_conversion_rate
        ubar = ubars[var_id]*conc_conversion_rate
        x_vals = func(u, alpha, theta) - func(ubar, alpha, theta)
        ax.plot(years, x_vals, label=f'{scenario} emulated', c=colors[j])
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{var_id.upper()} forcing ({units})")
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig(figname)
