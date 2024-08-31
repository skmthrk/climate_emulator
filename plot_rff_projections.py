import os
import csv
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from utils import make_logger, colors
logger = make_logger()

import string
letters = list(string.ascii_uppercase)

matplotlib.rc('axes', lw=0.25, edgecolor='k')
path_effects = [pe.withStroke(linewidth=3, foreground="w")]

def main():

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), frameon=False)
    figname = f"./output/fig_rff_projections.svg"

    data_dir = './data_processed'

    var_info = {
        'gdp': ['GDP world total', 1e-15, 'quadrillion USD'],
        'pop': ['Population', 1e-9, 'billion'],
        'gdppc_growth': ['Per capita GDP growth', 1e+2, '%'],
        'co2': ['CO2 emission', 1, 'GtCO2 yr-1'],
        'ch4': ['CH4 emission', 1, 'MtCH4 yr-1'],
        'n2o': ['N2O emission', 1, 'MtN2O yr-1'],
    }

    dx = 25
    xlim = [1750-dx, 2300+dx]
    base_year = 2020

    num_samples = 10000
    samples = list(range(1, num_samples+1))

    # quantile
    q1 = 25
    q2 = 35

    dir_names = ['gdp_pop', 'emissions']

    for row, dir_name in enumerate(dir_names):

        if dir_name == 'gdp_pop':
            var_ids = ['gdp', 'pop', 'gdppc_growth']
        elif dir_name == 'emissions':
            var_ids = ['co2', 'ch4', 'n2o']

        for var_id, ax in zip(var_ids, axes[row,:]):

            var_name, unit_conversion, unit = var_info[var_id]
            data = {}
            for sample in tqdm(samples, desc=var_id):
                if dir_name == 'gdp_pop':
                    file_name = f'{var_id}_sample_{sample}.csv'
                elif dir_name == 'emissions':
                    file_name = f'{var_id}_emission_sample_{sample}.csv'
                with open(os.path.join(data_dir, dir_name, file_name), 'r') as f:
                    reader = csv.reader(f, delimiter=',')
                    years = np.array(reader.__next__()).astype(int)
                    values = np.array(reader.__next__()).astype(float)
                    data[sample] = values*unit_conversion
            df = pd.DataFrame(data, index=years).T
            ax.fill_between(years, df.quantile(q1/100).values, df.quantile(1-q1/100).values, color='gray',alpha=0.2, edgecolor='none', label=f'{q1}-{100-q1}%')
            ax.fill_between(years, df.quantile(q2/100).values, df.quantile(1-q2/100).values, color='gray',alpha=0.4, edgecolor='none', label=f'{q2}-{100-q2}%')
            ax.plot(years, df.quantile(0.5).values, color='k', label='median', path_effects=path_effects)

            ax.set_ylabel(f'{var_name} ({unit})')
            ax.set_xlim(xlim)
            if var_id == 'gdppc_growth':
                ax.set_ylim([-1.5, 4.5])
            if var_id == 'gdp':
                ax.legend(loc='upper left', frameon=False, facecolor=None)
            if row == 1:
                ax.set_xlabel('year')

    for ax, letter in zip(axes.ravel(), letters):
        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())
        dx = (xlim[1]-xlim[0])/100
        dy = (ylim[1]-ylim[0])/100
        ax.text(xlim[0]-5*dx, ylim[1]+8*dy, letter, fontsize=16, ha='center', va='center')

        ax.set_facecolor('none')
        for posi in ['top', 'right']:
            ax.spines[posi].set_visible(False)

    fig.tight_layout()
    fig.savefig(figname)
    logger.info(f"Plot saved at {figname}")

if __name__ == '__main__':
    main()
