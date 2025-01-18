import os, sys
import re
import csv
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from utils import make_logger, colors

logger = make_logger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="MIROC6", help="model ID")
    return parser.parse_args()


def load_data(model_id, var_id, variant_label, experiment_ids):
    """
    Load data from CSV files matching the specified conditions.

    Args:
        model_id (str): Model ID
        var_id (str): Variable ID
        variant_label (str): Variant label
        experiment_ids (list): List of experiment IDs

    Returns:
        dict: Dictionary with experiment IDs as keys and tuples of (years, values) as values
    """
    input_dir = "./data_processed"
    data = {}
    for experiment_id in experiment_ids:

        years = []
        values = []

        file_name = rf'{var_id}_{model_id}_{experiment_id}_{variant_label}.csv'
        file_path = os.path.join(input_dir, file_name)
        logger.info(f"Processing {file_name}")
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            reader.__next__()
            for t in reader:
                year, value = int(t[0]), float(t[1])
                years.append(year)
                values.append(value)

        years = np.array(years) - years[0] + 1850
        values = np.array(values)
        data[experiment_id] = (years, values)

    return data


def plot_experiments(model_id, var_id, variant_label):
    """
    Plot the experiment data for the specified conditions.

    Args:
        model_id (str): Model ID
        var_id (str): Variable ID
        variant_label (str): Variant label

    Returns:
        matplotlib.figure.Figure: Generated plot figure
    """
    font_size = 12
    plt.rcParams.update({'font.size': font_size})

    # load data
    experiment_ids = ['piControl', 'abrupt-4xCO2', 'abrupt-2xCO2', '1pctCO2']
    data = load_data(model_id, var_id, variant_label, experiment_ids)

    # plot
    figsize = np.array([280,210])/23
    fig = plt.figure(frameon=False, figsize=figsize)

    delta = 5
    gs = fig.add_gridspec(nrows=100, ncols=100, left=0.08, right=0.98, wspace=0.0, hspace=0.0)
    ax1 = fig.add_subplot(gs[0:50-delta,0:50-delta], facecolor='none', frameon=True) # for piControl
    ax2 = fig.add_subplot(gs[0:50-delta,50+delta:100], facecolor='none', frameon=True) # for abrupt-4xCO2 and abrupt-2xCO2
    ax3 = fig.add_subplot(gs[50+delta:100,0:50-delta], facecolor='none', frameon=True) # for 1pctCO2
    ax4 = fig.add_subplot(gs[50+delta:100,50+delta:100], facecolor='none', frameon=True) # all
    axes = [ax1, ax2, ax3, ax4]

    cs = {}
    letters = ['A', 'B', 'C', 'D']
    loc = "best"
    count = 0
    yearminmax = np.inf
    for idx, experiment_id in enumerate(['piControl', 'abrupt', '1pctCO2']):

        ax = axes[idx]
        if experiment_id  == 'abrupt':
            for exp_id in ['abrupt-4xCO2', 'abrupt-2xCO2']:
                c = colors[count]
                count += 1
                years, vals = data[exp_id]
                yearminmax = min(yearminmax, max(years))
                ax.plot(years, vals, label=f"{exp_id}", c=c)
                cs[exp_id] = c

            ax.legend(loc=loc, handletextpad=0.1, ncol=2, columnspacing=1.0)
            xlim = list(ax.get_xlim())
            ylim = list(ax.get_ylim())
            dx = (xlim[1]-xlim[0])/100
            dy = (ylim[1]-ylim[0])/100
            ax.set_ylim(ylim[0], ylim[1]+5*dy)

        else:
            c = colors[count]
            count += 1
            years, vals = data[experiment_id]
            yearminmax = min(yearminmax, max(years))
            ax.plot(years, vals, label=f"{experiment_id}", c=c)
            cs[experiment_id] = c

            ax.legend(loc=loc, handletextpad=0.1)
            xlim = list(ax.get_xlim())
            ylim = list(ax.get_ylim())
            dx = (xlim[1]-xlim[0])/100
            dy = (ylim[1]-ylim[0])/100
        ax.text(-0.01, 1.07, letters[idx], fontsize=16, ha='center', va='center', transform=ax.transAxes)

    # put them together in a single figure
    ax = axes[-1]
    for experiment_id in ['abrupt-4xCO2', 'abrupt-2xCO2', '1pctCO2','piControl']:
        years, vals = data[experiment_id]
        c = cs[experiment_id]
        j = list(years).index(yearminmax)
        ax.plot(years[:j], vals[:j], label=f"{experiment_id}", c=c)
    xlim = list(ax.get_xlim())
    ylim = list(ax.get_ylim())
    dx = (xlim[1]-xlim[0])/100
    dy = (ylim[1]-ylim[0])/100
    ylim = list(ax.get_ylim())
    ax.text(-0.01, 1.07, letters[-1], fontsize=16, ha='center', va='center', transform=ax.transAxes)

    ylabels = {
        'tas': 'Near surface temperature (K)',
        'rsdt': 'TOA incoming shortwave flux (W m-2)',
        'rsut': 'TOA outgoing shortwave flux (W m-2)',
        'rlut': 'TOA outgoing longwave flux (W m-2)',
        'rndt': 'Net downward flux (W m-2)',
    }

    for ax in axes:
        ax.set_ylabel(ylabels[var_id])
        ax.set_xlabel('Year')
        for posi in ['top', 'right']:
            # remove spines
            ax.spines[posi].set_visible(False)

    return fig

def main():

    args = parse_args()
    model_id = args.model_id
    variant_label = 'r1i1p1f1'

    var_ids = ['tas', 'rsdt', 'rsut', 'rlut']
    for var_id in var_ids:
        # plot
        fig = plot_experiments(model_id, var_id, variant_label)

        # Save the figure
        script_name, _ = os.path.splitext(os.path.basename(__file__))
        fig.savefig(f'./output/fig_{script_name}_{var_id}_{model_id}.svg')

if __name__ == '__main__':
    main()
