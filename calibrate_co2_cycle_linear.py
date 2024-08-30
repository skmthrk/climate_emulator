import os

import numpy as np
import scipy.linalg as la
from scipy import optimize, interpolate

import matplotlib
import matplotlib.pyplot as plt

from utils import make_logger, colors
logger = make_logger()

matplotlib.rc('axes', lw=0.25, edgecolor='k')

class CarbonCycleModel:

    def __init__(self, data_dir='./data_raw/Joos2013'):
        self.data_dir = data_dir
        self.experiment_ids = ['PI100', 'PD100', 'PI5000']
        #self.unit = {'CO2': 2.123, 'OHC': 1.0e-22} # ppm to GtC for CO2
        self.unit = {'CO2': 7.82116, 'FABCUM': 7.82116/2.123, 'FASCUM': 7.82116/2.123, 'OHC': 1.0e-22} # ppm to GtCO2 for CO2, GtC to GtCO2 for FABCUM and FASCUM
        self.dataset = self._build_dataset()
        self.years = np.arange(0.5, 1950 + 1, 1)
        self.methods = ['Nelder-Mead', 'Powell', 'SLSQP', 'TNC', 'COBYLA', 'BFGS', 'Newton-CG', 'L-BFGS-B']
        self.C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
        self.deltas_estimate = {}

    def _build_dataset(self):
        logger.info("Loading dataset")

        # import Joos 2013 dataset
        # - 'CO2': 'CO2 remaining in atmosphere (ppm)'
        # - 'FABCUM': 'Cumulative air-to-biosphere flux (GtC)'
        # - 'FASCUM': 'Cumulative air-to-sea flux (GtC)'
        # - 'T': 'Surface air temperature (K)'
        # - 'OHC': 'Ocean heat content (10^22 J)'

        var_ids = ['CO2', 'FABCUM', 'FASCUM', 'T', 'OHC']
        none_value = 1e+20

        dataset = {}
        for experiment_id in self.experiment_ids:
            data_dir = f'{self.data_dir}/IRF_{experiment_id}'
            data_func = {}
            for var_id in var_ids:
                file_path = os.path.join(data_dir, f"IRF_{experiment_id}_SMOOTHED_{var_id}.dat")
                d = self._load_data_from_file(file_path)
                values = np.array([value * self.unit.get(var_id, 1) for value in d['MULTI-MODEL MEAN'] if value != none_value])
                years = d['YEAR'][:len(values)]
                func_value = interpolate.CubicSpline(years, values)
                sigmas = np.array(d['MULTI-MODEL STDEV'][:len(values)]) * self.unit.get(var_id, 1)
                func_sigma = interpolate.CubicSpline(years, sigmas)
                data_func[var_id] = (func_value, func_sigma)
            dataset[experiment_id] = data_func
        return dataset

    @staticmethod
    def _load_data_from_file(file_path):
        d = {}
        with open(file_path, 'r') as f:
            keys = []
            for line in f.readlines():
                if line.startswith('#'):
                    if line.strip().startswith('# COLUMN'):
                        _, key = line.split(':')
                        keys.append(key.strip())
                    continue
                vals = np.array(line.split(), dtype='float')
                for key, val in zip(keys, vals):
                    d.setdefault(key, []).append(val)
        return d

    @staticmethod
    def build_delta(deltas):
        delta21, delta31, delta12, delta32, delta13, delta43, delta34 = deltas
        Delta = np.zeros((4, 4))
        Delta[0, 0] = -delta21 - delta31
        Delta[1, 0] = delta21
        Delta[2, 0] = delta31
        Delta[0, 1] = delta12
        Delta[1, 1] = -delta12 - delta32
        Delta[2, 1] = delta32
        Delta[0, 2] = delta13
        Delta[2, 2] = -delta13 - delta43
        Delta[3, 2] = delta43
        Delta[2, 3] = delta34
        Delta[3, 3] = -delta34
        return Delta

    def build_mm0(self, experiment_id):
        data = self.dataset[experiment_id]
        M = np.empty((3, len(self.years)))
        M[0, :] = data['CO2'][0](self.years)
        M[1, :] = data['FABCUM'][0](self.years)
        M[2, :] = data['FASCUM'][0](self.years)
        unit = self.unit['CO2']/2.13 # from GtC t0 GtCO2
        pulse = 5000*unit if experiment_id == 'PI5000' else 100*unit
        #pulse = 5000 if experiment_id == 'PI5000' else 100
        M0 = np.array([pulse, 0, 0, 0])
        return M, M0

    def loss_function(self, deltas, M, M0):
        Delta = self.build_delta(deltas)
        V = np.empty((Delta.shape[0], len(self.years)))
        for idx, year in enumerate(self.years):
            if idx == 0:
                V[:, idx] = la.expm(Delta * year).dot(M0)
            else:
                V[:, idx] = la.expm(Delta).dot(V[:, idx-1])
        return la.norm(M - self.C.dot(V))

    def estimate_deltas(self, experiment_id):
        logger.info(f"Estimating model parameters for {experiment_id}")
        M, M0 = self.build_mm0(experiment_id)
        #deltas0 = [0.01, 0.01, 0.01, 0.01, 0.01, 0.003, 0.001]
        if experiment_id == 'PD100':
            deltas0 = [0.01863479684654072, 0.01215988570333602, 0.030922261924123055, 1.5657535015012984e-14, 0.011990027437269814, 0.002250139010973608, 0.0011650685369245027]
        elif experiment_id == 'PI100':
            deltas0 = [0.024680152325295687, 0.029923037996215623, 0.022920299672492817, 5.821060681039502e-16, 0.028974039183124422, 0.005184990138766448, 0.0016280893267839312]
        elif experiment_id == 'PI5000':
            deltas0 = [0.002472973339047518, 0.007312124037786888, 2.0436297048656608e-13, 0.010453385780799738, 0.034537181762809246, 0.0037851011983986963, 0.000712864905036077]
        bounds = [(0, None) for _ in range(len(deltas0))]
        res = optimize.minimize(
            fun=lambda deltas: self.loss_function(deltas, M, M0),
            x0=deltas0,
            method=self.methods[0],
            bounds=bounds,
            tol=1e-10,
            options={'maxiter': 10000}
        )
        logger.info(f"Optimization result: {res.message} in {res.nit} iterations")
        logger.info(f"Minvalue = {res.fun} at {list(res.x)}")
        self.deltas_estimate[experiment_id] = list(res.x)

        # save estimated parameters
        csv_path = f"./output/parameter_co2_cycle_linear_{experiment_id}.csv"
        with open(csv_path, 'w') as f:
            keys = ['var_id', 'delta21', 'delta31', 'delta12', 'delta32', 'delta13', 'delta43', 'delta34']
            vals = ['co2'] + [str(delta) for delta in res.x]
            f.write(','.join(keys))
            f.write('\n')
            f.write(','.join(vals))
        logger.info(f"Parameter values saved at {csv_path}")

    def plot_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), frameon=False)
        years = self.years

        for idx, (experiment_id, ax) in enumerate(zip(self.experiment_ids, axes)):
            M, M0 = self.build_mm0(experiment_id)
            Delta = self.build_delta(self.deltas_estimate[experiment_id])
            V = np.empty((Delta.shape[0], len(years)))
            for i, year in enumerate(years):
                if i == 0:
                    V[:, i] = la.expm(Delta * year).dot(M0)
                else:
                    V[:, i] = la.expm(Delta).dot(V[:, i-1])

            unit = self.unit['CO2']/2.13 # convert GtC to GtCO2
            pulse_size = 5000*unit if experiment_id == 'PI5000' else 100*unit
            ax.plot(years, M[0, :] / pulse_size, label='Atmosphere (Joos, 2013)', c=colors[0], ls='--')
            ax.plot(years, V[0, :] / pulse_size, label='Atmosphere (model)', c=colors[0])
            ax.plot(years, M[1, :] / pulse_size, label='Biosphere (Joos, 2013)', c=colors[1], ls='--')
            ax.plot(years, V[1, :] / pulse_size, label='Biosphere (model)', c=colors[1])
            ax.plot(years, M[2, :] / pulse_size, label='Ocean (Joos, 2013)', c=colors[2], ls='--')
            ax.plot(years, (V[2, :] + V[3, :]) / pulse_size, label='Ocean (model)', c=colors[2])
            ax.set_xlabel('Years after pulse')
            ax.set_ylabel('CO2 fraction')
            ax.set_title(experiment_id)
            if idx == 2:
                ax.legend(loc='upper right', frameon=False, facecolor=None)

        for ax in axes.ravel():
            ax.set_facecolor('none')
            for posi in ['top', 'right']:
                ax.spines[posi].set_visible(False)
        fig.tight_layout()
        figname = './output/fig_co2_cycle_linear.svg'
        fig.savefig(figname)
        logger.info(f"Plot saved at {figname}")

def main():

    model = CarbonCycleModel()

    for experiment_id in model.experiment_ids:
        model.estimate_deltas(experiment_id)

    model.plot_results()

if __name__ == "__main__":
    main()
