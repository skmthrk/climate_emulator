import os
import csv

from tqdm.auto import tqdm
import multiprocessing as mp
import numpy as np


def build_delta_tas(model_id, pulse_var, pulse_size=1, pulse_year=2020):

    if pulse_size == 0:
        pulse_dir = "nopulse"
    else:
        pulse_dir = f"pulse_{pulse_size}_{pulse_year}"

    data_dir0 = './output'
    data_dir_nopulse = os.path.join(data_dir0, 'nopulse')
    data_dir_pulse = os.path.join(data_dir0, pulse_dir)

    var_id = pulse_var
    out_dir = os.path.join(data_dir_pulse, f'delta_{var_id}')
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        print("Directory {} already exists".format(out_dir))
    out_dir_tas = os.path.join(data_dir_pulse, f'delta_tas_under_{pulse_var}_pulse_{model_id}')
    try:
        os.mkdir(out_dir_tas)
    except FileExistsError:
        print("Directory {} already exists".format(out_dir_tas))

    num_samples = 10000
    samples = range(1, num_samples+1)
    for sample in tqdm(samples, desc=f"delta temperature under {pulse_var} pulse ({model_id})"):

        for var_type in ['emission', 'concentration', 'forcing']:
            with open(os.path.join(os.path.join(data_dir_pulse, var_id), f"{var_id}_{var_type}_sample_{sample}.csv"), 'r') as f:
                reader = csv.reader(f, delimiter=',')
                years = np.array(reader.__next__()).astype(int)
                values = np.array(reader.__next__()).astype(float)

            with open(os.path.join(os.path.join(data_dir_nopulse, var_id), f"{var_id}_{var_type}_sample_{sample}.csv"), 'r') as f:
                reader = csv.reader(f, delimiter=',')
                years = np.array(reader.__next__()).astype(int)
                values -= np.array(reader.__next__()).astype(float)

            with open(os.path.join(out_dir, f'{var_id}_{var_type}_sample_{sample}.csv'), 'w') as f:
                f.write(','.join(str(year) for year in years))
                f.write('\n')
                f.write(','.join(str(value) for value in values))

        with open(os.path.join(os.path.join(data_dir_pulse, f'tas_under_{pulse_var}_pulse_{model_id}'), f"tas_sample_{sample}.csv"), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            years = np.array(reader.__next__()).astype(int)
            values = np.array(reader.__next__()).astype(float)

        with open(os.path.join(os.path.join(data_dir_nopulse, f'tas_{model_id}'), f"tas_sample_{sample}.csv"), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            years = np.array(reader.__next__()).astype(int)
            values -= np.array(reader.__next__()).astype(float)

        with open(os.path.join(out_dir_tas, f'tas_sample_{sample}.csv'), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in values))

def process_func(args):
    '''
    function for multiprocessing
    '''
    build_delta_tas(*args)

def main():
    model_ids = [
        'MIROC6',
#        'CanESM5',
#        "ACCESS-CM2",
#        "BCC-CSM2-MR",
#        "CESM2",
#        "CMCC-CM2-SR5",
#        "CNRM-CM6-1",
#        "FGOALS-f3-L",
#        "GISS-E2-1-G",
#        "HadGEM3-GC31-LL",
#        "INM-CM5-0",
#        "IPSL-CM6A-LR",
#        "KACE-1-0-G",
#        "MPI-ESM1-2-LR",
#        "MRI-ESM2-0",
#        "NorESM2-LM",
    ]

    pulse_vars = ['co2', 'ch4', 'n2o']
    args_list = [(model_id, pulse_var, 1, 2020) for model_id in model_ids for pulse_var in pulse_vars]
    with mp.Pool() as pool:
        pool.map(process_func, args_list)

if __name__ == '__main__':
    main()
