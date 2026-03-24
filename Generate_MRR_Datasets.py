#!/usr/bin/env python3
"""
Generate_MRR_Datasets.py

Reads Micro Rain Radar (MRR) data from Alta and Highland Bowl sites for
specified event-type periods, filters out 12-h windows with insufficient
observations, and saves the resulting datasets as NetCDF files.

Authors: Jim Steenburgh, Ashley Evans, Michael Wasserstein
Date: 12 June 2024

Usage
-----
    python Generate_MRR_Datasets.py [--period {ALL,FR,SIVT,PF}]
                                    [--min_obs MIN_OBS]
                                    [--output_dir OUTPUT_DIR]

Event-type periods
------------------
    ALL   All available times (Nov–Apr, both seasons)
    FR    Frontal events
    SIVT  South/southwest IVT events
    PF    Northwest post-cold-frontal events
"""

import os
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

from MRR_functions import read_data


# ------------------ USER SPECIFIED VARIABLES ----------------
# Event type to process; Options: ALL, FR, SIVT, PF
parser = argparse.ArgumentParser(description='Generate PARSIVEL datasets for a given event type.')
parser.add_argument('--period', choices=['ALL', 'FR', 'SIVT', 'PF'], default='ALL',
                    help='Synoptic classification to process (default: ALL)')
args = parser.parse_args()
period = args.period

# Minimum number of MRR obs needed to use 12-h period in processing
# Maximum is 360 to 361
min_obs = 355

# Path for saving data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_save_path = os.path.join(SCRIPT_DIR, 'Data', 'MRR') + os.sep
# ---------------- END USER SPECIFIED VARIABLES --------------


if period == 'ALL':
    periods_to_analyze = [('2022-11-15 11:00:00', '2023-05-01 10:00:00'),
                          ('2023-11-15 11:00:00', '2024-05-01 10:00:00')]

elif period == 'FR':
    periods_to_analyze = [('2022-12-04 23:00:00', '2022-12-05 11:00:00'),
                          ('2022-12-27 23:00:00', '2022-12-28 11:00:00'),
                          ('2023-01-10 23:00:00', '2023-01-11 11:00:00'),
                          ('2023-02-05 11:00:00', '2023-02-05 23:00:00'),
                          ('2023-02-21 23:00:00', '2023-02-22 11:00:00'),
                          ('2023-03-05 23:00:00', '2023-03-06 11:00:00'),
                          ('2023-03-10 23:00:00', '2023-03-11 11:00:00'),
                          ('2023-03-15 10:00:00', '2023-03-15 22:00:00'),
                          ('2023-03-24 10:00:00', '2023-03-24 22:00:00'),
                          ('2024-02-15 11:00:00', '2024-02-15 23:00:00'),
                          ('2024-02-26 23:00:00', '2024-02-27 11:00:00'),
                          ('2024-03-23 22:00:00', '2024-03-24 10:00:00')]

elif period == 'SIVT':
    periods_to_analyze = [('2022-12-01 23:00:00', '2022-12-02 11:00:00'),
                          ('2022-12-11 23:00:00', '2022-12-12 11:00:00'),
                          ('2022-12-31 23:00:00', '2023-01-01 11:00:00'),
                          ('2023-01-01 11:00:00', '2023-01-01 23:00:00'),
                          ('2023-01-09 23:00:00', '2023-01-10 11:00:00'),
                          ('2023-01-14 23:00:00', '2023-01-15 11:00:00'),
                          ('2023-03-14 22:00:00', '2023-03-15 10:00:00'),
                          ('2024-02-01 23:00:00', '2024-02-02 11:00:00'),
                          ('2024-02-05 11:00:00', '2024-02-05 23:00:00'),
                          ('2024-02-06 23:00:00', '2024-02-07 11:00:00'),
                          ('2024-02-19 23:00:00', '2024-02-20 11:00:00'),
                          ('2024-03-28 10:00:00', '2024-03-28 22:00:00'),
                          ('2024-03-30 22:00:00', '2024-03-31 10:00:00')]

elif period == 'PF':
    periods_to_analyze = [('2022-11-28 23:00:00', '2022-11-29 11:00:00'),
                          ('2022-12-12 23:00:00', '2022-12-13 11:00:00'),
                          ('2022-12-13 11:00:00', '2022-12-13 23:00:00'),
                          ('2022-12-13 23:00:00', '2022-12-14 11:00:00'),
                          ('2022-12-14 11:00:00', '2022-12-14 23:00:00'),
                          ('2022-12-15 11:00:00', '2022-12-15 23:00:00'),
                          ('2022-12-28 11:00:00', '2022-12-28 23:00:00'),
                          ('2022-12-28 23:00:00', '2022-12-29 11:00:00'),
                          ('2023-01-06 11:00:00', '2023-01-06 23:00:00'),
                          ('2023-01-11 11:00:00', '2023-01-11 23:00:00'),
                          ('2023-01-24 23:00:00', '2023-01-25 11:00:00'),
                          ('2023-01-27 23:00:00', '2023-01-28 11:00:00'),
                          ('2023-01-28 11:00:00', '2023-01-28 23:00:00'),
                          ('2023-02-06 11:00:00', '2023-02-06 23:00:00'),
                          ('2023-03-24 22:00:00', '2023-03-25 10:00:00'),
                          ('2023-03-27 10:00:00', '2023-03-27 22:00:00'),
                          ('2023-12-08 11:00:00', '2023-12-08 23:00:00'),
                          ('2023-12-08 23:00:00', '2023-12-09 11:00:00'),
                          ('2024-01-05 11:00:00', '2024-01-05 23:00:00'),
                          ('2024-01-07 11:00:00', '2024-01-07 23:00:00'),
                          ('2024-01-14 11:00:00', '2024-01-14 23:00:00'),
                          ('2024-02-02 23:00:00', '2024-02-03 11:00:00'),
                          ('2024-02-03 23:00:00', '2024-02-04 11:00:00'),
                          ('2024-02-09 11:00:00', '2024-02-09 23:00:00'),
                          ('2024-03-24 10:00:00', '2024-03-24 22:00:00'),
                          ('2024-04-06 10:00:00', '2024-04-06 22:00:00'),
                          ('2024-04-06 22:00:00', '2024-04-07 10:00:00'),
                          ('2024-04-15 22:00:00', '2024-04-16 10:00:00')]


# Read the data
xr.set_options(keep_attrs=True)

# First period
alta180    = read_data('Alta',     180, pd.Timestamp(periods_to_analyze[0][0]), pd.Timestamp(periods_to_analyze[0][1]))
alta30     = read_data('Alta',      30, pd.Timestamp(periods_to_analyze[0][0]), pd.Timestamp(periods_to_analyze[0][1]))
highland180 = read_data('Highland', 180, pd.Timestamp(periods_to_analyze[0][0]), pd.Timestamp(periods_to_analyze[0][1]))
highland30  = read_data('Highland',  30, pd.Timestamp(periods_to_analyze[0][0]), pd.Timestamp(periods_to_analyze[0][1]))

# Loop through the rest of the periods
for start_time, end_time in periods_to_analyze[1:]:
    print('Reading:')
    print(start_time, end_time)
    tempdata = read_data('Alta', 180, pd.Timestamp(start_time), pd.Timestamp(end_time))
    alta180 = xr.merge([alta180, tempdata], join='outer')
    tempdata = read_data('Alta', 30, pd.Timestamp(start_time), pd.Timestamp(end_time))
    alta30 = xr.merge([alta30, tempdata], join='outer')
    tempdata = read_data('Highland', 180, pd.Timestamp(start_time), pd.Timestamp(end_time))
    highland180 = xr.merge([highland180, tempdata], join='outer')
    tempdata = read_data('Highland', 30, pd.Timestamp(start_time), pd.Timestamp(end_time))
    highland30 = xr.merge([highland30, tempdata], join='outer')
    del tempdata


# If processing everything, need to focus on days with sufficient data at both sites
if period == 'ALL':

    def delete_period(data, current_time, current_time_p12h):
        mask = ((data['time'] >= np.datetime64(current_time_p12h)) | (data['time'] < np.datetime64(current_time)))
        data = data.where(mask, drop=True)
        return data

    totalperiods_mrravail = 0.0
    for start_time, end_time in periods_to_analyze:

        current_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

        while current_time < datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S'):
            current_time_p12h = current_time + timedelta(hours=12)
            alta180count    = len(alta180['time'].sel(time=slice(current_time.strftime('%Y-%m-%dT%H:%M:%S.000000000'),
                                current_time_p12h.strftime('%Y-%m-%dT%H:%M:%S.000000000'))).values)
            alta30count     = len(alta30['time'].sel(time=slice(current_time.strftime('%Y-%m-%dT%H:%M:%S.000000000'),
                                current_time_p12h.strftime('%Y-%m-%dT%H:%M:%S.000000000'))).values)
            highland180count = len(highland180['time'].sel(time=slice(current_time.strftime('%Y-%m-%dT%H:%M:%S.000000000'),
                                current_time_p12h.strftime('%Y-%m-%dT%H:%M:%S.000000000'))).values)
            highland30count  = len(highland30['time'].sel(time=slice(current_time.strftime('%Y-%m-%dT%H:%M:%S.000000000'),
                                current_time_p12h.strftime('%Y-%m-%dT%H:%M:%S.000000000'))).values)

            if alta180count < min_obs or alta30count < min_obs or highland180count < min_obs or highland30count < min_obs:
                print('\033[31mDELETING ' + current_time.strftime('%Y-%m-%d %H:%M:%S') + ' to ' + current_time_p12h.strftime('%Y-%m-%d %H:%M:%S') +
                      ':\tAlta 180: ' + str(alta180count) + '\tAlta 30: ' + str(alta30count) +
                      '\tHighland 180: ' + str(highland180count) + '\tHighland 30: ' + str(highland30count) + '\033[0m')
                alta180     = delete_period(alta180,     current_time, current_time_p12h)
                alta30      = delete_period(alta30,      current_time, current_time_p12h)
                highland180 = delete_period(highland180, current_time, current_time_p12h)
                highland30  = delete_period(highland30,  current_time, current_time_p12h)
            else:
                print('Keeping ' + current_time.strftime('%Y-%m-%d %H:%M:%S') + ' to ' + current_time_p12h.strftime('%Y-%m-%d %H:%M:%S') +
                      ':\tAlta 180: ' + str(alta180count) + '\tAlta 30: ' + str(alta30count) +
                      '\tHighland 180: ' + str(highland180count) + '\tHighland 30: ' + str(highland30count))
                totalperiods_mrravail += 1

            current_time += timedelta(hours=12)

    print('Total 12-h Periods with >= ' + str(min_obs) + ' MRR obs available:\t ' + str(int(totalperiods_mrravail)))


# Save datasets
highland180.to_netcdf(data_save_path + f'Highland/highland180_{period}.nc')
highland30.to_netcdf( data_save_path + f'Highland/highland030_{period}.nc')
alta180.to_netcdf(    data_save_path + f'Alta/alta180_{period}.nc')
alta30.to_netcdf(     data_save_path + f'Alta/alta030_{period}.nc')
