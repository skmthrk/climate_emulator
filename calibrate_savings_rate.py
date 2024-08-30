import os

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from utils import make_logger
logger = make_logger()

matplotlib.rc('axes', lw=0.25, edgecolor='k')

def main():

    logger.info('Estimating the savings rate')

    file_path = './data_raw/gross_savings.csv'
    df = pd.read_csv(file_path)
    years_raw = np.arange(1960, 2023+1,1)
    savings_rates_raw = df.loc[(df['Country Name'] == 'World'),"{0} [YR{0}]".format(years_raw[0]):"{0} [YR{0}]".format(years_raw[-1])].values.squeeze()

    years = []
    savings_rates = []
    for year, savings_rate in zip(years_raw, savings_rates_raw):
        if savings_rate != '..':
            years.append(year)
            savings_rates.append(savings_rate/100)
    savings_rates = np.array(savings_rates)
    savings_rate_mean = sum(savings_rates)/len(savings_rates)

    csv_path = './output/parameter_savings_rate.csv'
    with open(csv_path, 'w') as f:
        f.write(',x\n')
        f.write('1-alpha,{}\n'.format(savings_rate_mean))
    logger.info(f"Parameter values saved at {csv_path}")

    figname = f"./output/fig_savings_rate.svg"
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot()
    ax.plot(years, savings_rates*100, c='k', label='annual value')
    ax.hlines(y=savings_rate_mean*100, xmin=years[0], xmax=years[-1], color='k', label='sample mean', ls='--')
    ax.set_xlabel("years")
    ax.set_ylabel(f"savings rate (% of GDP)")
    ax.legend(loc='upper left', frameon=False, facecolor=None)

    ax.set_facecolor('none')
    for posi in ['top', 'right']:
        ax.spines[posi].set_visible(False)
    fig.tight_layout()
    fig.savefig(figname)
    logger.info(f"Plot saved at {figname}")

if __name__ == '__main__':
    main()
