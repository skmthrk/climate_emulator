import os
import csv

from tqdm.auto import tqdm
import multiprocessing as mp
import numpy as np

colors = [
    '#348ABD', # 0: blue
    '#7A68A6', # 1: purple
    '#A60628', # 2: red
    '#467821', # 3: green
    '#188487', # 4: breen
    '#CF4457', # 5: pink
    '#E24A33', # 6: orange
]

var_ids = ['co2', 'ch4', 'n2o']

def calculate_present_value(model_id, pulse_var, pulse_size=1, pulse_year=2020):

    if pulse_size == 0:
        pulse_dir = "nopulse"
    else:
        pulse_dir = f"pulse_{pulse_size}_{pulse_year}"

    dir_output = './output'
    data_dir_gdppc = os.path.join('./data_processed', 'gdp_pop')
    data_dir_pulse = os.path.join(dir_output, pulse_dir)
    data_dir_nopulse = os.path.join(dir_output, 'nopulse')

    base_year = 2020

    parameters = {}
    with open(os.path.join(dir_output, 'parameter_rho_eta.csv'), 'r') as csvfile:
        f = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, t in enumerate(f):
            if i == 0:
                continue
            key, val = t
            parameters[key] = float(val)

    with open(os.path.join(dir_output, 'parameter_savings_rate.csv'), 'r') as csvfile:
        f = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, t in enumerate(f):
            if i == 0:
                continue
            key, val = t
            parameters[key] = float(val)

    rho = parameters['rho']
    eta = parameters['eta']
    alpha = 1 - parameters['1-alpha']
    #print('rho:', rho)
    #print('eta:', eta)
    #print('alpha:', alpha)

    num_samples = 10000
    samples = list(range(1, num_samples+1))

    data_dir_damage = os.path.join(data_dir_pulse, f'delta_damage_under_{pulse_var}_pulse_{model_id}')

    present_values = []
    for sample in tqdm(samples, desc=pulse_var):

        with open(os.path.join(data_dir_nopulse, f'tas_{model_id}', f"damage_frac_sample_{sample}.csv"), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            damage_frac_years = np.array(reader.__next__()).astype(int)
            damage_frac_values = np.array(reader.__next__()).astype(float)

        with open(os.path.join(data_dir_gdppc, f'gdppc_sample_{sample}.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            gdppc_years = np.array(reader.__next__()).astype(int)
            gdppc_values = np.array(reader.__next__()).astype(float)
            
        with open(os.path.join(data_dir_damage, f'damage_sample_{sample}.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            years = np.array(reader.__next__()).astype(int)
            damage_values = np.array(reader.__next__()).astype(float)
            idx = np.where(years == base_year)[0][0]

        damage_adjustments = np.log( (alpha-damage_frac_values) / (alpha-damage_frac_values[idx]) )
        damage_adjustments[damage_adjustments == -np.inf] = 0
            
        gdppc_values = np.append(np.zeros(len(years)-len(gdppc_years)), gdppc_values)

        growth_rates = np.log( gdppc_values / gdppc_values[idx] )
        growth_rates[growth_rates == -np.inf] = 0
        ts = years - base_year

#        discount_factors = np.exp(-rho*ts - eta*growth_rates)
        discount_factors = np.exp(-rho*ts - eta*growth_rates - eta*damage_adjustments)
        with open(os.path.join(data_dir_damage, f"discount_factor_sample_{sample}.csv"), 'w') as f:
            f.write(','.join(str(year) for year in years[idx:]))
            f.write('\n')
            f.write(','.join(str(value) for value in discount_factors[idx:]))

        present_value = sum(discount_factors[idx:]*damage_values[idx:])
        present_values.append(present_value)

        with open(os.path.join(data_dir_damage, f"damage_present_value_sample_{sample}.csv"), 'w') as f:
            f.write(','.join(str(year) for year in years[idx:]))
            f.write('\n')
            f.write(','.join(str(value) for value in discount_factors[idx:]*damage_values[idx:]))

    unit_conversion_rate = {
        'co2': (1e-9, 'tCO2'),
        'ch4': (1e-6, 'tCH4'),
        'n2o': (1e-6, 'tN2O'),
    }
    
    values = [v for v in present_values if not np.isnan(v)]
    estimate = sum(values)/len(values) # per unit pulse
    social_cost = estimate*unit_conversion_rate[pulse_var][0]
    unit = unit_conversion_rate[pulse_var][1]
    print(f'social cost of {pulse_var} ({model_id}): {social_cost} (USD/{unit})')

    with open(os.path.join(data_dir_pulse, f"present_values_under_{pulse_var}_pulse_{model_id}.csv"), 'w') as f:
        f.write(','.join(str(sample) for sample in samples))
        f.write('\n')
        f.write(','.join(str(value) for value in present_values))

def process_func(args):
    '''
    function for multiprocessing
    '''
    calculate_present_value(*args)

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

#    for model_id in model_ids:
#        for pulse_var in ['co2', 'ch4', 'n2o']:
#            calculate_present_value(model_id=model_id, pulse_var=pulse_var)

if __name__ == '__main__':
    main()
