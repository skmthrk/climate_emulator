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
        self.parameters_estimate = {}

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
        phi, zeta = parameters
        if zeta == 0:
            return phi * np.log(u)
        return phi * (1/zeta)*(u**zeta - 1)

    def loss_function(self, parameters, var_id):
        loss = 0
        for scenario in self.scenarios:
            x_obs = self.datasets['forcing'][scenario][var_id]
            u = self.datasets['concentration'][scenario][var_id]
            ubar = self.ubars[var_id]
            x_vals = self.forcing_func(u, parameters) - self.forcing_func(ubar, parameters)
            loss += np.linalg.norm(x_vals - x_obs)
        print(loss)
        return loss

    def estimate_parameters(self, var_id):

        logger.info(f"Estimating parameters for {var_id} forcing model")

        parameters = [1, 0] # initial guess
        bounds = [(None, None), (0, 0)] if var_id == 'co2' else [(None, None), (None, None)]
        method = self.methods[0]
        tol = 1e-11
        maxiter = 5000
        res = optimize.minimize(fun=lambda parameters:self.loss_function(parameters, var_id=var_id), x0=parameters, method=method, bounds=bounds, tol=tol, options={'maxiter': maxiter})
        parameters = list(res.x)

        logger.info(f"Optimization result: {res.message} in {res.nit} iterations")
        logger.info(f"Minvalue = {res.fun} at {parameters}")

        csv_path = f"./output/parameter_{var_id}_forcing.csv"
        with open(csv_path, 'w') as f:
            keys = ['var_id', 'phi', 'zeta']
            vals = [var_id] + [str(para) for para in res.x]
            f.write(','.join(keys))
            f.write('\n')
            f.write(','.join(vals))
        logger.info(f"Parameter values saved at {csv_path}")

        self.parameters_estimate[var_id] = parameters

    def plot_result(self):
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), frameon=False)
        years = self.years
        figname = f"./output/fig_forcing.svg"

        for var_id, ax in zip(self.var_ids, axes):
            parameters = self.parameters_estimate[var_id]
            for j, scenario in enumerate(self.scenarios):

                x_obs = self.datasets['forcing'][scenario][var_id]
                ax.plot(years, x_obs, label=f'{scenario}', c=colors[j], ls='--')
            
                u = self.datasets['concentration'][scenario][var_id]
                ubar = self.ubars[var_id]
                x_vals = self.forcing_func(u, parameters) - self.forcing_func(ubar, parameters)
                ax.plot(years, x_vals, label=f'{scenario} (model)', c=colors[j])
            ax.set_xlabel("Year")
            ax.set_ylabel(f"{var_id.upper()} forcing (W m-2)")
            ax.set_title(f"{var_id.upper()}")
            if ax == axes[-1]:
                ax.legend(loc='upper left', frameon=False, facecolor=None)
            
        for ax in axes.ravel():
            ax.set_facecolor('none')
            for posi in ['top', 'right']:
                ax.spines[posi].set_visible(False)
        fig.tight_layout()
        fig.savefig(figname)
        logger.info(f"Plot saved at {figname}")

def main():

    model = ForcingModel()

    for var_id in model.var_ids:
        model.estimate_parameters(var_id)

    model.plot_result()

if __name__ == '__main__':
    main()
