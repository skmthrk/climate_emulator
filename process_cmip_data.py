import os
import re
import argparse

import xarray as xr
import numpy as np

from utils import deg2rad, area, list_files, make_logger

logger = make_logger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="MIROC6", help="model ID")
    parser.add_argument("--var_ids", nargs='+', default=['tas', 'rsdt', 'rsut', 'rlut'], help="variable IDs")
    return parser.parse_args()

def build_data(input_dir, output_dir, model_id, experiment_id, var_id):
    """
    Build processed data from raw data files.

    Args:
        input_dir (str): Input directory containing raw data files.
        output_dir (str): Output directory to save processed data files.
        model_id (str): Model ID.
        experiment_id (str): Experiment ID.
        var_id (str): Variable ID.

    Returns:
        None
    """

    variant_label = 'r1i1p1f1'
    if model_id in ["CNRM-CM6-1"]:
        variant_label = 'r1i1p1f2'
    if model_id in ["HadGEM3-GC31-LL"]:
        variant_label = 'r1i1p1f3'

    pattern = rf'{var_id}_.*{model_id}_{experiment_id}_{variant_label}.*\.nc$'
    file_names = list_files(input_dir, pattern)

    output_data = []
    for file_name in file_names:
        logger.info(f"Processing {file_name}")

        file_path = os.path.join(input_dir, file_name)
        ds = xr.open_dataset(file_path)
        variable_id = ds.variable_id
        da = ds[variable_id]

        # for output file name
        var_id, _, model_id, experiment_id, variant_id, grid_type, duration = file_name.split('_')

        try:
            # load areacella file
            area_file_name = f"areacella_fx_{model_id}_{experiment_id}_{variant_id}_{grid_type}.nc"
            area_file_path = os.path.join(input_dir, area_file_name)
            area_ds = xr.open_dataset(area_file_path)
            area_da = area_ds[area_ds.variable_id]
            area_da.data = area_da.data * 1e-6 # convert m2 to km2
            area_da.attrs["units"] = "km2"
        except FileNotFoundError:
            logger.warning(f"areacella file not found: {area_file_name}. Generating area data array.")
            area_da = area(da)

        # compute annual mean
        years = []
        annual_values = []

        # for month weight
        numdays_of_month = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}

        time = da['time']
        current_year = int(time[0].dt.year.values)
        month_values = []
        for idx_t, t in enumerate(time):
            year = int(t.dt.year.values)
            month = int(t.dt.month.values)

            # spatial aggregation
            month_value = np.nansum(da.sel(time=t).data * area_da.data/area_da.values.sum())
            month_weight = numdays_of_month[month]/365
            month_value *= month_weight

            if year == current_year:
                # keep adding month_value until year switch
                month_values.append(month_value)
            else:
                # switch to next year
                if len(month_values) == 12:
                    annual_value = sum(month_values)
                    annual_values.append(annual_value)
                    years.append(current_year)
                month_values = [month_value]
                current_year = year

            # last index
            if idx_t == len(time)-1:
                # save only if values exist for full year
                if len(month_values) == 12:
                    annual_values.append(sum(month_values))
                    years.append(year)

        # generate output file
        lines = [f"{year},{annual_value}" for year, annual_value in zip(years, annual_values)]
        output_data.append((years[0], lines))

    # save
    output_data.sort()
    output = []
    for _, lines in output_data:
        output += lines
    file_name = f"{var_id}_{model_id}_{experiment_id}_{variant_label}.csv"
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w') as f:
        f.write(f"year,{var_id}\n")
        f.write('\n'.join(output))

def main():

    input_dir = './data_raw/CMIP6'
    output_dir = './data_processed'

    args = parse_args()
    model_id = args.model_id
    var_ids = args.var_ids

    calibration_experiment_ids = [
        'piControl',
        'abrupt-2xCO2',
        'abrupt-4xCO2',
        '1pctCO2',
    ]

    evaluation_experiment_ids = [
        'historical',
        'ssp119',
        'ssp245',
        'ssp370',
        'ssp460',
        'ssp585',
    ]

    for var_id in var_ids:
        if var_id == 'tas':
            experiment_ids = calibration_experiment_ids + evaluation_experiment_ids
        else:
            experiment_ids = calibration_experiment_ids

        for experiment_id in experiment_ids:
            build_data(input_dir, output_dir, model_id, experiment_id, var_id)

if __name__ == '__main__':
    main()
