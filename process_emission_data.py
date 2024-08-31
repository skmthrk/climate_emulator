import os
import csv
from tqdm.auto import tqdm
from tqdm.contrib import tenumerate

import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline

import matplotlib
import matplotlib.pyplot as plt

from utils import make_logger, colors
logger = make_logger()

matplotlib.rc('axes', lw=0.25, edgecolor='k')

class DataProcessor:

    def __init__(self):

        self.var_ids = ['co2', 'ch4', 'n2o']
        self.scenarios = ['ssp119', 'ssp245', 'ssp370', 'ssp460', 'ssp585']

        self.rcmip_data_dir = './data_raw/RCMIP'
        self.rcmip_data_file = {
            'emission': "rcmip-emissions-annual-means-v5-1-0.csv",
            'concentration': "rcmip-concentrations-annual-means-v5-1-0.csv",
            'forcing': "rcmip-radiative-forcing-annual-means-v5-1-0.csv",
        }

        self.samples = range(1, 10000+1)
        self.rff_data_dir = './data_raw/RFF/emissions'
        self.rff_data_file_pattern = 'rffsp_{var_id}_emissions.csv'

        self.ssp_datasets = self._build_ssp_datasets()
        self.rff_dataset = self._build_rff_dataset()

        self.output_dir = './data_processed/emissions'
        os.makedirs(self.output_dir, exist_ok=True)

    def _build_ssp_datasets(self):

        datasets = {}
        for var_type in ['emission', 'concentration']:

            data_path = os.path.join(self.rcmip_data_dir, self.rcmip_data_file[var_type])
            logger.info(f'Building SSP {var_type} data from {data_path}')

            df_ssp = pd.read_csv(data_path)
    
            if var_type == 'emission':
                var_name = 'Emissions'
                unit_conversion = {
                    'co2': 1e-3, # MtCO2/yr to GtCO2/yr
                    'ch4': 1, # MtCH4/yr
                    'n2o': 1e-3, # KtN2O/yr to MtN2O/yr
                }
            elif var_type == 'concentration':
                var_name = 'Atmospheric Concentrations'
                unit_conversion = {
                    'co2': 7.82116, # ppm to GtCO2
                    'ch4': 2.85094, # ppb to MtCH4
                    'n2o': 7.82187, # ppb to MtN2O
                }
            elif var_type == 'forcing':
                var_name = 'Effective Radiative Forcing|Anthropogenic'
                unit_conversion = {
                    'co2': 1,
                    'ch4': 1,
                    'n2o': 1,
                }
    
            dataset = {}
            year0, year1 = 1750, 2500
            years = np.arange(year0, year1+1)
            for scenario in self.scenarios:
                data = {}
                for var_id in self.var_ids:
                    data[var_id] = (
                        years,
                        df_ssp.loc[
                            (df_ssp["Region"] == "World")
                            & (df_ssp["Scenario"] == scenario)
                            & (df_ssp["Variable"] == f"{var_name}|{var_id.upper()}"),
                            f"{year0}":f"{year1}",
                        ].interpolate(axis=1).values.squeeze() * unit_conversion[var_id]
                    )
                dataset[scenario] = data
            datasets[var_type] = dataset

        return datasets

    def _build_rff_dataset(self):
        
        dataset = {}

        unit_conversion = {
            'co2': 44/12, # GtC to GtCO2
            'ch4': 1,  # MtCH4 to MtCH4
            'n2o': 44/28, # MtN2 to MtN2O
        }

        dataset_raw = {}
        for var_id in self.var_ids:
        
            data_path = os.path.join(self.rff_data_dir, self.rff_data_file_pattern.format(var_id=var_id))
            logger.info(f'Building RFF {var_id} data from {data_path}')

            data_raw = {}
            with open(data_path, 'r') as f:
                '''
                each line = sample, year, value
                '''
                reader = csv.reader(f, delimiter=',')
                for i, lst in tenumerate(reader, desc=f'loading {var_id} rffsp data'):
                    if i == 0:
                        continue
                    sample, year, value = lst
                    sample = int(sample)
                    year = int(year)
                    value = float(value) * unit_conversion[var_id]
                    data_raw.setdefault(sample, []).append((year, value))
            dataset_raw[var_id] = data_raw

        dataset = {}
        for sample in self.samples:
            data = {}
            for var_id in self.var_ids:
                data_raw = dataset_raw[var_id]
                sorted_lst = sorted(data_raw[sample])
                years = np.array([t[0] for t in sorted_lst])
                values = np.array([t[1] for t in sorted_lst])
                data[var_id] = (years, values)
            dataset[sample] = data

        return dataset

    def process_data(self, base_scenario='ssp245'):

        for var_id in self.var_ids:

            logger.info(f'Processing {var_id} emission')

            years_ssp, values_ssp = self.ssp_datasets['emission'][base_scenario][var_id]
    
            # stitch ssp data and rff data
            for sample in tqdm(self.samples):
                years_rff, values_rff = self.rff_dataset[sample][var_id]
                idx = np.where(years_ssp == years_rff[0])[0][0]
                years = np.append(years_ssp[:idx], years_rff)
                values = np.append(values_ssp[:idx], values_rff)
            
                # save
                file_path = os.path.join(self.output_dir, f'{var_id}_emission_sample_{sample}.csv')
                with open(file_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(years)
                    writer.writerow(values)
        logger.info(f'Processed data saved at {self.output_dir}')
        
def main():

    dp = DataProcessor()
    dp.process_data()

if __name__ == '__main__':
    main()
