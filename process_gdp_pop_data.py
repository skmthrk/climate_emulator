import os
import csv
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

from utils import make_logger, colors

logger = make_logger()

class DataProcessor:

    def __init__(self):

        self.data_dir = './data_raw'
        self.maddison_file = 'maddison.csv'
        self.rff_file_pattern = 'RFF/pop_income/rffsp_pop_income_run_{}.feather'
        self.thousands = 1e+3
        self.millions = 1e+6
        self.historical_data = None

        self.output_dir = './data_processed/gdp_pop'
        os.makedirs(self.output_dir, exist_ok=True)

    def build_historical_data(self):
        # units expected for output
        # gdp: 2011 USD
        # pop: person
        # gdppc: 2011 USD per person

        logger.info(f'Building historical data based on {self.maddison_file}')

        data = {}
        with open(os.path.join(self.data_dir, self.maddison_file), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            reader.__next__()
            for line in reader:
                year = line[0]
                gdppc = line[9] # 2011 USD per person
                pop = line[-1] # thousands
                if gdppc and pop:
                    year = int(year)
                    gdppc = int(gdppc)
                    pop = int(pop)*self.thousands # unit conversion
                    data.setdefault('year', []).append(year)
                    data.setdefault('gdp', []).append(gdppc*pop)
                    data.setdefault('pop', []).append(pop)

        for var_id in ['year', 'gdp', 'pop']:
            data[var_id] = np.array(data[var_id])
        data['gdppc'] = data['gdp']/data['pop']

        self.historical_data = data

    def process_data(self, sample):

        # load historical data
        historical_data = self.historical_data

        # Load RFF data
        # GDP (millions 2011 USD)
        # Pop (thousands)

        df = pd.read_feather(os.path.join(self.data_dir, self.rff_file_pattern.format(sample)))
        rff_data = {}
        rff_data['year'] = np.array(sorted(set(df['Year']))) # 5 year step
        rff_data['gdp'] = np.array([df.loc[df['Year'] == year]['GDP'].values.sum()*self.millions for year in rff_data['year']])
        rff_data['pop'] = np.array([df.loc[df['Year'] == year]['Pop'].values.sum()*self.thousands for year in rff_data['year']])
        rff_data['gdppc'] = rff_data['gdp'] / rff_data['pop'] # 2011 USD per person

        # stitch historical and rff data
        idx = np.where(historical_data['year'] == 2020)[0][0] # stitching point
        for var_id in ['gdp', 'pop', 'gdppc']:
            interpolate_func = interpolate.CubicSpline(
                np.append(historical_data['year'][:idx], rff_data['year']),
                np.append(historical_data[var_id][:idx], rff_data[var_id])
            )
            years = np.arange(historical_data['year'][0], 2300+1, 1)
            values = interpolate_func(years)
            if var_id == 'pop':
                values = values.astype(int)

            # save
            file_path = os.path.join(self.output_dir, f'{var_id}_sample_{sample}.csv')
            with open(file_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(years)
                writer.writerow(values)

            # compute GDP per capita growth rate
            if var_id == 'gdppc':
                growth_values = np.log(values[1:]/values[:-1])
                file_path = os.path.join(self.output_dir, f'{var_id}_growth_sample_{sample}.csv')
                with open(file_path, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(years[:-1])
                    writer.writerow(growth_values)

def main():

    dp = DataProcessor()

    # historical data
    dp.build_historical_data()

    # Combine historical and RFF data for each sample
    num_samples = 10000
    logger.info(f'Processing data for {num_samples} samples')
    samples = range(1, num_samples+1)
    for sample in tqdm(samples):
        dp.process_data(sample)
    logger.info(f'Processed data saved at {dp.output_dir}')

if __name__ == '__main__':
    main()
