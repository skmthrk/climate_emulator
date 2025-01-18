import os
import csv

from tqdm.auto import tqdm
import multiprocessing as mp
import numpy as np

def damage_func(T, parameters):
    a1 = parameters['a1']
    a2 = parameters['a2']
    a3 = parameters['a3']
    return a1*T + a2*T**a3

def build_damage(model_id, pulse_var, pulse_size=1, pulse_year=2020):

    if pulse_size == 0:
        pulse_dir = "nopulse"
    else:
        pulse_dir = f"pulse_{pulse_size}_{pulse_year}"

    data_dir0 = './output'
    data_dir_nopulse = os.path.join(data_dir0, 'nopulse')
    data_dir_pulse = os.path.join(data_dir0, pulse_dir)

    conversion_rate = 1
#    giga = 1e+9 # from Gt to t
#    mega = 1e+6 # from Gt to t
#
#    if pulse_var == 'co2':
#        conversion_rate = 1/giga
#    else:
#        conversion_rate = 1/mega

    out_dir_damage = os.path.join(data_dir_pulse, f'delta_damage_under_{pulse_var}_pulse_{model_id}')
    try:
        os.mkdir(out_dir_damage)
    except FileExistsError:
        print("Directory {} already exists".format(out_dir_damage))

    parameters = {}
    with open(os.path.join('./output', 'parameter_damage.csv'), 'r') as csvfile:
        f = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, t in enumerate(f):
            if i == 0:
                continue
            key, val = t
            parameters[key] = float(val)

    num_samples = 10000
    samples = range(1, num_samples+1)

    for sample in tqdm(samples, desc=f'{pulse_var} ({model_id})'):

        with open(os.path.join('./data_processed', 'gdp_pop', f'gdp_sample_{sample}.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            gdp_years = np.array(reader.__next__()).astype(int)
            gdp_values = np.array(reader.__next__()).astype(float)

        with open(os.path.join(data_dir_nopulse, f'tas_{model_id}', f'tas_sample_{sample}.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            years = np.array(reader.__next__()).astype(int)
            tas_values_nopulse = np.array(reader.__next__()).astype(float)
        damage_values_nopulse = damage_func(tas_values_nopulse, parameters)

        with open(os.path.join(data_dir_pulse, f'tas_under_{pulse_var}_pulse_{model_id}', f'tas_sample_{sample}.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            years = np.array(reader.__next__()).astype(int)
            tas_values_pulse = np.array(reader.__next__()).astype(float)
        damage_values_pulse = damage_func(tas_values_pulse, parameters)

        gdp_values = np.append(np.zeros(len(years)-len(gdp_years)), gdp_values)
        delta_damage_values = (damage_values_pulse - damage_values_nopulse)*gdp_values*conversion_rate

        with open(os.path.join(out_dir_damage, f"damage_sample_{sample}.csv"), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in delta_damage_values))

        with open(os.path.join(data_dir_nopulse, f'tas_{model_id}', f"damage_frac_sample_{sample}.csv"), 'w') as f:
            f.write(','.join(str(year) for year in years))
            f.write('\n')
            f.write(','.join(str(value) for value in damage_values_nopulse))

def process_func(args):
    '''
    function for multiprocessing
    '''
    build_damage(*args)

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
