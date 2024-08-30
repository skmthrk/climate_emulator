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

class GasCycleModel:
    
    def __init__(self, data_dir='./data_raw/RCMIP'):

        self.data_dir = data_dir
        self.years = list(range(1750, 2501))
        self.scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
        self.var_ids = ['ch4', 'n2o']
        self.xbars = {'ch4': 720*2.85094, 'n2o': 270*7.82187}
        self.conc_unit = {'ch4': 'MtCH4', 'n2o': 'MtN2O'}
        self.var_types = ['emission', 'concentration']
        self.methods = ['Nelder-Mead', 'Powell', 'SLSQP', 'TNC', 'COBYLA', 'BFGS', 'Newton-CG', 'L-BFGS-B']
        self.datasets = self._build_datasets()
        self.deltas_estimate = {}

    def _build_datasets(self):
        """ load RCMIP dataset
        var_type: 'emission', 'concentration'
        var_ids: 'ch4', 'n2o'
        scenarios: 'ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585'
        """
        datasets = {}
        for var_type in self.var_types:
            if var_type == 'emission':
                data_file = "rcmip-emissions-annual-means-v5-1-0.csv"
                var_name = 'Emissions'
                unit_conversion = {
                    'ch4': 1, # MtCH4/yr
                    'n2o': 1e-3, # KtN2O/yr to MtN2O/yr
                }
            elif var_type == 'concentration':
                data_file = "rcmip-concentrations-annual-means-v5-1-0.csv"
                var_name = 'Atmospheric Concentrations'
                unit_conversion = {
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
                    ].interpolate(axis=1).values.squeeze() * unit_conversion[var_id]
                dataset[scenario] = data
            datasets[var_type] = dataset
        return datasets

    @staticmethod
    def build_A(deltas):
        """ convert parameter list into gas cycle matrix Delta
        """
        [delta11, delta21, delta31, delta12, delta22, delta32, delta13, delta23, delta33] = deltas
        A = np.zeros((3,3))
        A[1-1,1-1] = delta11
        A[2-1,1-1] = delta21
        A[3-1,1-1] = delta31
        A[1-1,2-1] = delta12
        A[2-1,2-1] = delta22
        A[3-1,2-1] = delta32
        A[1-1,3-1] = delta13
        A[2-1,3-1] = delta23
        A[3-1,3-1] = delta33
        return A

    @staticmethod
    def discretize_model(A):
        m, n = A.shape
        Ad = la.expm(A)
        B = np.zeros(m)
        B[0] = 1
        D = np.zeros((2*m, 2*n))
        D[:m,:m] = A
        D[:m,m:] = np.identity(m)
        Bd = la.expm(D)[:m,m:].dot(B)
        return Ad, Bd

    def loss_function(self, deltas, var_id):
        A = self.build_A(deltas)
        m, n = A.shape
        Ad, Bd = self.discretize_model(A)
        loss = 0
        for scenario in self.scenarios:
            x_obs = self.datasets['concentration'][scenario][var_id]
            u = self.datasets['emission'][scenario][var_id]
            x_obs_normalized = x_obs - self.xbars[var_id]
            x_vals = np.zeros((m, len(self.years)+1))
            x_vals[0,0] = x_obs_normalized[0]
            for idx, year in enumerate(self.years):
                x_vals[:, idx+1] = Ad.dot(x_vals[:, idx]) + Bd*u[idx]
            loss += np.linalg.norm(x_vals[0,:-1]-x_obs_normalized)
        return loss

    def estimate_deltas(self, var_id):

        logger.info(f"Estimating deltas for {var_id}")

        if var_id == 'ch4':
            m = 2 # number of layers in gas cycle matric
            delta0 = -0.1 # initial guess
        elif var_id == 'n2o':
            m = 1 # number of layers in gas cycle matric
            delta0 = -0.009 # initial guess
    
        deltas = np.zeros(3*3)
        bounds = [(0, 0) for delta in deltas]
        for i in range(m*m):
            bounds[i] = (None, None)
        tol = 1e-11
        maxiter = 5000
        method = self.methods[0]
        res = optimize.minimize(
            fun=lambda deltas: self.loss_function(deltas, var_id), x0=deltas, method=method, bounds=bounds, tol=tol, options={'maxiter': maxiter})
        deltas = list(res.x)
        logger.info(f"Optimization result: {res.message} in {res.nit} iterations")
        logger.info(f"Minvalue = {res.fun} at {deltas}")

        self.deltas_estimate[var_id] = deltas

        with open(f"./output/parameter_{var_id}_cycle.csv", 'w') as f:
            keys = ['var_id', 'delta11', 'delta21', 'delta31', 'delta12', 'delta22', 'delta32', 'delta13', 'delta23', 'delta33']
            vals = [var_id] + [str(delta) for delta in res.x]
            f.write(','.join(keys))
            f.write('\n')
            f.write(','.join(vals))

    def plot_result(self):

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), frameon=False)
        figname = f"./output/fig_gas_cycle.svg"

        for var_id, ax in zip(self.var_ids, axes):

            deltas = self.deltas_estimate[var_id]
            A = self.build_A(deltas)
            m, n = A.shape
            Ad, Bd = self.discretize_model(A)

            for j, scenario in enumerate(self.scenarios):
                x_obs = self.datasets['concentration'][scenario][var_id]
                x_obs_normalized = x_obs - self.xbars[var_id]
                ax.plot(self.years, x_obs_normalized + self.xbars[var_id], label=f'{scenario}', c=colors[j], ls='--')

                x_vals = np.zeros((m, len(self.years)+1))
                x_vals[0,0] = x_obs_normalized[0]
                u = self.datasets['emission'][scenario][var_id]
                for idx, year in enumerate(self.years):
                    x_vals[:, idx+1] = Ad.dot(x_vals[:, idx]) + Bd*u[idx]
                ax.plot(self.years, x_vals[0,:-1] + self.xbars[var_id], label=f'{scenario} (model)', c=colors[j])
            ax.set_xlabel("Year")
            ax.set_ylabel(f"{var_id.upper()} concentration ({self.conc_unit[var_id]})")
            ax.legend(loc='upper left')

        for ax in axes.ravel():
            ax.set_facecolor('none')
            for posi in ['top', 'right']:
                ax.spines[posi].set_visible(False)
        fig.tight_layout()
        fig.savefig(figname)
        logger.info(f"Plot saved at {figname}")

def main():

    model = GasCycleModel()

    for var_id in model.var_ids:
        model.estimate_deltas(var_id)

    model.plot_result()

if __name__ == '__main__':
    main()
