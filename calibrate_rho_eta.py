import os
import csv

import pandas as pd

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib.patheffects

from tqdm.auto import tqdm

from utils import make_logger
logger = make_logger()

def setup_matplotlib():
    plt.rcParams['axes.edgecolor'] = 'k'
    plt.rcParams['axes.linewidth'] = 0.25
    return [matplotlib.patheffects.withStroke(linewidth=3.5, foreground="w")]

def load_historical_data(file_path):
    df_hist = pd.read_csv(file_path, header=None, names=['year', 'r'])
    years_data, values_data = df_hist['year'].values, df_hist['r'].values
    r_func = CubicSpline(years_data, values_data, extrapolate=True)
    years_hist = np.arange(int(years_data[0]), 2018)
    values_hist = r_func(years_hist) / 100
    return years_hist, values_hist

def load_future_data(file_path, start_year, duration):
    years = np.arange(start_year, start_year + duration)
    df = pd.read_csv(file_path) / 100
    df.drop(df.columns[0], axis=1, inplace=True)
    return years, df

def load_gdp_growth_data(file_template, num_samples):
    data = {}
    for sample in tqdm(range(1, num_samples + 1), desc='Loading GDP growth data'):
        with open(file_template.format(sample=sample), 'r') as f:
            reader = csv.reader(f)
            years = next(reader)
            values = next(reader)
            data[sample] = [float(v) for v in values]
    return pd.DataFrame(data, index=years).T

def plot_interest_rate(years, df, years_hist, values_hist, path_effects):
    fig, ax = plt.subplots()
    idx = np.where(years == 2300)[0][0]
    
    ax.fill_between(years[:idx], df.quantile(0.25).values[:idx], df.quantile(0.75).values[:idx], color='gray', alpha=0.2, edgecolor='none', label='25-75%')
    ax.fill_between(years[:idx], df.quantile(0.35).values[:idx], df.quantile(0.65).values[:idx], color='gray', alpha=0.4, edgecolor='none', label='35-65%')
    ax.plot(years[:idx], df.quantile(0.5).values[:idx], color='k', label='median', path_effects=path_effects)
    ax.plot(years_hist, values_hist, color='k', path_effects=path_effects)

    ax.set_xlabel("Year")
    ax.set_ylabel("Real Interest Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output', "fig_interest_rate.svg"))
    plt.close()

def loss_function(parameters, df_interest_rate, df_growth_rate):
    rho, eta = parameters
    df_discount_rate = rho + eta * df_growth_rate
    loss = 0
    loss += la.norm(df_interest_rate.mean().values - df_discount_rate.mean().values)
    loss += la.norm(df_interest_rate.std().values - df_discount_rate.std().values)
    for q in [0.25, 0.50, 0.75]:
        loss += la.norm(df_interest_rate.quantile(q).values - df_discount_rate.quantile(q).values)
    print(loss)
    return loss

def optimize_parameters(df_interest_rate, df_growth_rate):
    initial_guess = [1, 2]
    result = minimize(
        fun=lambda x: loss_function(x, df_interest_rate, df_growth_rate),
        x0=initial_guess,
        method='SLSQP',
        bounds=[(None, None), (None, None)],
        tol=1e-12,
        options={'maxiter': 100000}
    )
    print_optimization_result(result)
    return tuple(result.x)

def print_optimization_result(result):
    print(f" method: SLSQP")
    print(f" message: {result.message}")
    print(f" nit: {result.nit}")
    print(f" status: {result.status}")
    print(f" success: {result.success}")
    print(f" min: {result.fun}")
    print(f" minimizer: {list(result.x)}")

def save_parameters(rho, eta):
    with open(os.path.join('output', 'parameter_rho_eta.csv'), 'w') as f:
        f.write('parameter,value\n')
        f.write(f'rho,{rho}\n')
        f.write(f'eta,{eta}\n')

def plot_comparison(years, df_interest_rate, df_growth_rate, rho, eta, path_effects):
    fig, ax = plt.subplots()

    df_discount_rate = rho + eta * df_growth_rate
    plot_quantiles(ax, years, df_discount_rate, 'gray', 'rho+eta*g', path_effects)
    plot_quantiles(ax, years, df_interest_rate, 'blue', 'r', path_effects)

    ax.set_xlabel("Year")
    ax.set_ylabel("rho + eta*g / Interest Rate")
    ax.set_title(f"rho = {rho:.4f}, eta = {eta:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('output', "fig_rho_eta_g.svg"))
    plt.close()

def plot_quantiles(ax, x, df, color, label, path_effects):
    ax.fill_between(x, df.quantile(0.25), df.quantile(0.75), color=color, alpha=0.2, edgecolor='none', label=f'{label} (25-75%)')
    ax.fill_between(x, df.quantile(0.35), df.quantile(0.65), color=color, alpha=0.4, edgecolor='none', label=f'{label} (35-65%)')
    ax.plot(x, df.quantile(0.5), color=color, label=f'{label} (median)', path_effects=path_effects)

def main():
    path_effects = setup_matplotlib()

    logger.info('Load historical interest rate data')
    file_path = os.path.join('data_raw', 'Bauer2023/interest_rate_historical.csv')
    years_hist, values_hist = load_historical_data(file_path)

    logger.info('Load simulatted future interest rate data')
    file_path = os.path.join('data_processed', 'interest_rate/interest_rate_y10_2019.csv')
    years, df_future = load_future_data(file_path, 2019, 400)
    df_interest_rate = df_future.loc[:, "V1":"V280"]

    logger.info('Plot interest rate')
    plot_interest_rate(years, df_future, years_hist, values_hist, path_effects)

    # Load GDP growth rate data
    file_path_template = os.path.join('data_processed', 'gdp_pop/gdppc_growth_sample_{sample}.csv')
    df_growth = load_gdp_growth_data(file_path_template, num_samples=10000)
    df_growth_rate = df_growth.loc[:, "2020":"2299"]

    logger.info('Optimize parameters, rho and eta')
    rho, eta = optimize_parameters(df_interest_rate, df_growth_rate)

    logger.info('Save parameter values')
    save_parameters(rho, eta)

    logger.info('Plot result')
    plot_comparison(np.arange(2020, 2300), df_interest_rate, df_growth_rate, rho, eta, path_effects)

if __name__ == '__main__':
    main()
