import os

import numpy as np
import scipy.linalg as la
from scipy import optimize
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from utils import make_logger, colors
logger = make_logger()

matplotlib.rc('axes', lw=0.25, edgecolor='k')

class NonlinearCarbonCycleModel:

    def __init__(self, data_dir='./data_raw/RCMIP'):

        self.data_dir = data_dir
        self.scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']
        self.years = list(range(1750, 2501))
        self.base_year = 2015
        self.base_year_idx = self.years.index(self.base_year)
        self.experiment_ids = ['PI100', 'PD100', 'PI5000']
        self.emphasis_on_base_year = 10
        self.var_ids = ['co2']
        self.var_types = ['emission', 'concentration']
        self.xbar = 278 * 7.82116 # ppm to GtCO2
        self.methods = ['Nelder-Mead', 'Powell', 'SLSQP', 'TNC', 'COBYLA', 'BFGS', 'Newton-CG', 'L-BFGS-B']
        self.datasets = self._build_datasets()
        self.deltas = self._build_deltas()
        self.gammas_estimate = {}

    def _build_datasets(self):
        """ load RCMIP dataset
        var_type: 'emission', 'concentration'
        var_ids: 'co2', 'ch4', 'n2o'
        scenarios: 'ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585'
        """
        datasets = {}
        for var_type in self.var_types:
            if var_type == 'emission':
                data_file = "rcmip-emissions-annual-means-v5-1-0.csv"
                var_name = 'Emissions'
                unit_conversion = 1e-3 # MtCO2/yr to GtCO2/yr
            elif var_type == 'concentration':
                data_file = "rcmip-concentrations-annual-means-v5-1-0.csv"
                var_name = 'Atmospheric Concentrations'
                unit_conversion = 7.82116 # ppm to GtCO2
            data_path = f"./data_raw/RCMIP/{data_file}"
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
                    ].interpolate(axis=1).values.squeeze() * unit_conversion
                dataset[scenario] = data
            datasets[var_type] = dataset
        return datasets

    def _build_deltas(self):
        # load linear carbon cycle parameters estimated based on Joos (2013)
        deltas = {}
        for experiment_id in self.experiment_ids:
            csv_path = f'./output/parameter_co2_cycle_linear_{experiment_id}.csv'
            with open(csv_path, 'r') as f:
                deltas[experiment_id] = np.array([float(v) for v in f.readlines()[1].split(',')[1:]])
        return deltas

    @staticmethod
    def build_A(deltas):
        """ 
        convert parameter list into carbon cycle matrix
        """
        [delta21, delta31, delta12, delta32, delta13, delta43, delta34] = deltas
        A = np.zeros((4,4))
        A[1-1,1-1] = -delta21 - delta31
        A[2-1,1-1] = delta21
        A[3-1,1-1] = delta31
        A[1-1,2-1] = delta12
        A[2-1,2-1] = -delta12 - delta32
        A[3-1,2-1] = delta32
        A[1-1,3-1] = delta13
        A[3-1,3-1] = -delta13 - delta43
        A[4-1,3-1] = delta43
        A[3-1,4-1] = delta34
        A[4-1,4-1] = -delta34
    
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

    def generate_x_vals(self, deltas, gammas, x_vals00, u):
        gamma0, gamma1 = gammas
        x_vals = np.zeros((4, len(self.years)+1))
        x_vals[0,0] = x_vals00
        for i, year in enumerate(self.years):
            M = sum(x_vals[:, i])
            alpha = np.exp(gamma0 + gamma1*M)
            A = self.build_A(deltas)/alpha
            Ad, Bd = self.discretize_model(A)
            x_vals[:, i+1] = Ad.dot(x_vals[:, i]) + Bd*u[i]
        return x_vals

    def loss_function(self, gammas, deltas):
        loss = 0
        for scenario in self.scenarios:
            x_obs = self.datasets['concentration'][scenario]['co2']
            x_obs_normalized = x_obs - self.xbar

            u = self.datasets['emission'][scenario]['co2']
            x_vals00 = x_obs_normalized[0]
            x_vals = self.generate_x_vals(deltas, gammas, x_vals00, u)

            loss += np.linalg.norm(x_vals[0,:-1]-x_obs_normalized)
            loss += self.emphasis_on_base_year * np.linalg.norm(x_vals[0,self.base_year_idx]-x_obs_normalized[self.base_year_idx])
        return loss

    def estimate_gammas(self, base_experiment):

        method = self.methods[0]

        logger.info(f"Estimating gammas for {base_experiment}-based model")
    
        deltas = self.deltas[base_experiment]
        gammas = [0, 0] # initial value
        bounds = [(None, None) for i in range(len(gammas))]
        tol = 1e-9
        maxiter = 5000
        res = optimize.minimize(fun=lambda gammas: self.loss_function(gammas, deltas=deltas), x0=gammas, method=method, bounds=bounds, tol=tol, options={'maxiter': maxiter})
        gammas = list(res.x)
    
        logger.info(f"Optimization result: {res.message} in {res.nit} iterations")
        logger.info(f"Minvalue = {res.fun} at {gammas}")
    
        # save estimated parameters
        csv_path = f"./output/parameter_co2_cycle_nonlinear_{base_experiment}.csv"
        with open(csv_path, 'w') as f:
            keys = ['var_id', 'gamma0', 'gamma1']
            vals = ['co2'] + [str(gamma) for gamma in gammas]
            f.write(','.join(keys))
            f.write('\n')
            f.write(','.join(vals))
        logger.info(f"Parameter values saved at {csv_path}")

        self.gammas_estimate[base_experiment] = gammas

    def plot_result(self):
    
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), frameon=False)
        figname = f"./output/fig_co2_cycle.svg"
    
        for row in [0, 1]:
            for idx, (base_experiment, ax) in enumerate(zip(self.experiment_ids, axes[row,:])):
        
                deltas = self.deltas[base_experiment]
                if row == 0:
                    # plot linear model
                    gammas = (0, 0)
                    title = f"{base_experiment} (linear model)"
                if row == 1:
                    # plot nonlinear model
                    gammas = self.gammas_estimate[base_experiment]
                    title = f"{base_experiment} (nonlinear model)"
        
                for j, scenario in enumerate(self.scenarios):
                    x_obs = self.datasets['concentration'][scenario]['co2']
                    x_obs_normalized = x_obs - self.xbar
                    ax.plot(self.years, x_obs_normalized + self.xbar, label=f'{scenario}', c=colors[j], ls='--')
        
                    u = self.datasets['emission'][scenario]['co2']
                    x_vals00 = x_obs_normalized[0]
                    x_vals = self.generate_x_vals(deltas, gammas, x_vals00, u)
                    ax.plot(self.years, x_vals[0,:-1] + self.xbar, label=f'{scenario} (model)', c=colors[j])
                ax.set_xlabel("Year")
                ax.set_ylabel("CO2 concentration (GtCO2)")
                ax.set_title(title)
                if idx == 2:
                    ax.legend(loc='upper left', frameon=False, facecolor=None)
    
        for ax in axes.ravel():
            ax.set_facecolor('none')
            for posi in ['top', 'right']:
                ax.spines[posi].set_visible(False)
        fig.tight_layout()
        fig.savefig(figname)
        logger.info(f"Plot saved at {figname}")

def main():

    model = NonlinearCarbonCycleModel()

    # estimate non-linear adjustment parameters for different base_experiment (PI100, PD100, PI5000)
    for base_experiment in model.experiment_ids:
        model.estimate_gammas(base_experiment)

    # plot the result
    model.plot_result()

if __name__ == '__main__':
    main()
