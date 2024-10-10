import os
import sys
import csv
from glob import glob
from tqdm.auto import tqdm

import wbdata

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

from scipy import interpolate


HERE = os.path.dirname(os.path.realpath(__file__))

colors = [
    '#348ABD', # 0: blue
    '#7A68A6', # 1: purple
    '#A60628', # 2: red
    '#467821', # 3: green
    '#188487', # 4: breen
    '#CF4457', # 5: pink
    '#E24A33', # 6: orange
]

def build_historical_gdp_pop_data(source='maddison'):
    if source == 'maddison':
        data_dir = os.path.join(HERE, 'data/historical/social/GDP/')
        years_hist = []
        gdp_values_hist = []
        pop_values_hist = []
        thousands = 1e+3

        # GDPpc (2011 USD per person)
        # Pop (thousands)

        with open(os.path.join(data_dir, 'maddison.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            reader.__next__()
            for line in reader:
                year = line[0]
                gdppc = line[9]
                pop = line[-1]
                if gdppc and pop:
                    year = int(year)
                    gdppc = int(gdppc)
                    pop = int(pop)*thousands
                    years_hist.append(year)
                    gdp_values_hist.append(gdppc*pop)
                    pop_values_hist.append(pop)
#    else:
#        data_id = 'NY.GDP.MKTP.KD'
#        data_description = "GDP (constant 2015 US$)"
#        trillion = 1e+12
#        
#        indicator = {data_id: data_description}
#        df = wbdata.get_dataframe(indicator, country="WLD", parse_dates=False).dropna()
#        years_hist = sorted([int(s) for s in df.index[:].values])
#        values_hist = [df.loc[str(year)].values.squeeze()/trillion for year in years_hist]
    
    historical_gdp_pop_data = (np.array(years_hist), np.array(gdp_values_hist), np.array(pop_values_hist))
    return historical_gdp_pop_data

def build_rff_gdp_data(historical_gdp_pop_data):

    thousands = 1e+3
    millions = 1e+6

    years_hist, gdp_values_hist, pop_values_hist = historical_gdp_pop_data
    gdppc_values_hist = gdp_values_hist/pop_values_hist
    idx = np.where(years_hist == 2020)[0][0]

    out_dir0 = os.path.join(HERE, 'output')
    out_dir = os.path.join(out_dir0, 'gdp_pop')
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        print("Directory {} already exists".format(out_dir))

    dir_name = 'data/scenario/social'

    years = np.arange(years_hist[0], 2300+1, 1)
    num_samples = 10000
    samples = range(1, num_samples+1)
    for sample in tqdm(samples):

        # GDP (millions 2011 USD)
        # Pop (thousands)
        file_name = f"pop_income/rffsp_pop_income_run_{sample}.feather"
        df = pd.read_feather(os.path.join(HERE, dir_name, file_name))
        years_rff = np.array(sorted(set(df['Year']))) # 5 year step
        gdp_values_rff = np.array([df.loc[df['Year'] == year]['GDP'].values.sum()*millions for year in years_rff])
        pop_values_rff = np.array([df.loc[df['Year'] == year]['Pop'].values.sum()*thousands for year in years_rff])
        gdppc_values_rff = gdp_values_rff/pop_values_rff

        gdp_func = interpolate.CubicSpline(
            np.append(years_hist[:idx], years_rff),
            np.append(gdp_values_hist[:idx], gdp_values_rff)
        )

        pop_func = interpolate.CubicSpline(
            np.append(years_hist[:idx], years_rff),
            np.append(pop_values_hist[:idx], pop_values_rff)
        )

        gdppc_func = interpolate.CubicSpline(
            np.append(years_hist[:idx], years_rff),
            np.append(gdppc_values_hist[:idx], gdppc_values_rff)
        )

        gdp_values = gdp_func(years)
        with open(os.path.join(out_dir, f'gdp_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in gdp_values))

        pop_values = pop_func(years).astype(int)
        with open(os.path.join(out_dir, f'pop_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in pop_values))

        gdppc_values = gdppc_func(years).astype(int)
        with open(os.path.join(out_dir, f'gdppc_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in gdppc_values))

#        fig = plt.figure(frameon=False)
#        figname = f"fig_gdppc_sample_{sample}.svg"
#        ax = fig.add_subplot(1,1,1, facecolor='none')
#        ax.plot(years_hist, gdppc_values_hist, label='historical')
#        ax.plot(years_rff, gdppc_values_rff, label=f'rff sample {sample}')
#        ax.plot(years, gdppc_values, label='smoothed')
#        ax.set_xlabel("Year")
#        ax.set_ylabel("GDP per capita")
#        ax.legend()
#        fig.set_tight_layout(True)
#        fig.savefig(figname)
#        sys.exit()

def main():
    historical_gdp_pop_data = build_historical_gdp_pop_data()
    build_rff_gdp_data(historical_gdp_pop_data)

if __name__ == '__main__':
    main()
