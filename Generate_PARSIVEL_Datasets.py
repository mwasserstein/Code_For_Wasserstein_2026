#!/usr/bin/env python3
"""
Generate_PARSIVEL_Datasets.py

Reads raw Particle Size and Velocity (PARSIVEL) disdrometer text files from
the Wasatch Deployment (2022–2024), aggregates drop-size/fall-speed matrices
for each storm-classification event type, and saves the results as NumPy
arrays for downstream analysis.

Author: Michael Wasserstein
Date: 2024-02-11

Usage
-----
    python Generate_PARSIVEL_Datasets.py [--period {ALL,FR,SIVT,PF}]

Event-type periods
------------------
    ALL   All available times (Nov–Apr, both seasons)
    FR    Frontal events
    SIVT  South/southwest IVT events
    PF    Northwest post-cold-frontal events

Sites
-----
    Highland (HGH)
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd

from Parsivel_inputs_hgh import *

# ------------------ USER SPECIFIED VARIABLES ----------------
# Event type to process; Options: ALL, FR, SIVT, PF
parser = argparse.ArgumentParser(description='Generate PARSIVEL datasets for a given event type.')
parser.add_argument('--period', choices=['ALL', 'FR', 'SIVT', 'PF'], default='ALL',
                    help='Synoptic classification to process (default: ALL)')
args = parser.parse_args()
period = args.period

# Minimum number of 6-second observations required for a 1-h file to
# be treated as a precipitating period (maximum possible is 360).
MIN_OBS = 150

# Output directory for processed .npy files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_save_path = os.path.join(SCRIPT_DIR, 'Data', 'PARSIVEL') + os.sep
# ---------------- END USER SPECIFIED VARIABLES --------------


HOME = '/uufs/chpc.utah.edu/common/home'
PARSIVEL_DIR = os.path.join(HOME, 'steenburgh-group12/peter/WasatchDeployment_2022_2024_fromLaptop')

site_symbol_dict = {'Highland': 'HGH'}


if period == 'ALL':
    periods_to_analyze = [
        ('2022-11-15 11:00:00', '2023-05-01 10:00:00'),
        ('2023-11-15 11:00:00', '2024-05-01 10:00:00'),
    ]

elif period == 'FR':
    periods_to_analyze = [
        ('2022-12-04 23:00:00', '2022-12-05 11:00:00'),
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
        ('2024-03-23 22:00:00', '2024-03-24 10:00:00'),
    ]

elif period == 'SIVT':
    periods_to_analyze = [
        ('2022-12-01 23:00:00', '2022-12-02 11:00:00'),
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
        ('2024-03-30 22:00:00', '2024-03-31 10:00:00'),
    ]

elif period == 'PF':
    periods_to_analyze = [
        ('2022-11-28 23:00:00', '2022-11-29 11:00:00'),
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
        ('2024-04-15 22:00:00', '2024-04-16 10:00:00'),
    ]


def get_files(site, start_time, end_time):
    """
    Return absolute paths to raw PARSIVEL text files for *site* within
    [start_time, end_time).

    Files are identified by scanning the Raw directory for the expected
    filename pattern ``<symbol>0<YY><JJJ><HH>_raw.txt``, where ``YY`` is the
    two-digit year, ``JJJ`` is the Julian day, and ``HH`` is the UTC hour.
    The interval is right-exclusive: the file for *end_time* itself is not
    included (prevents double-counting the final hour).

    Parameters
    ----------
    site : str
        Site name (e.g. ``'Highland'``).
    start_time : datetime-like
        Start of the desired time window (inclusive).
    end_time : datetime-like
        End of the desired time window (exclusive).

    Returns
    -------
    list of str
        Absolute paths to files that exist on disk, sorted chronologically.
    """
    site_symbol = site_symbol_dict[site]
    data_dir = os.path.join(PARSIVEL_DIR, f'{site}/PARSIVEL/Raw/')

    # Build an hourly range; subtract 1 h so end_time is not included.
    date_range = pd.date_range(
        start=start_time,
        end=end_time - datetime.timedelta(hours=1),
        freq='1h',
    )

    files = []
    for date in date_range:
        YY = f'{date.year % 100:02d}'
        JJJ = f'{date.dayofyear:03d}'
        HH = f'{date.hour:02d}'
        filename = f'{site_symbol.lower()}0{YY}{JJJ}{HH}_raw.txt'
        file_path = os.path.join(data_dir, filename)

        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print(f'File not found: {site} {date:%Y-%m-%d %H}:00 UTC')

    return files


def parse_data(site, start_time, end_time):
    """
    Accumulate the 32×32 PARSIVEL drop-size/fall-speed matrix over all
    precipitating 1-h files for *site* in [start_time, end_time).

    Each raw file covers one UTC hour and contains one line per 6-second
    sample.  A file is accepted as a precipitating period only when it
    contains at least ``MIN_OBS`` data lines.  Within accepted files the
    three smallest diameter bins (rows 0–2) are zeroed following
    Yuter et al. (2006).

    Parameters
    ----------
    site : str
        Site name (e.g. ``'Highland'``).
    start_time : datetime-like
        Start of the desired time window (inclusive).
    end_time : datetime-like
        End of the desired time window (exclusive).

    Returns
    -------
    accum_matrix : numpy.ndarray, shape (32, 32)
        Accumulated drop-count matrix summed over all accepted files.
    num_files_used : int
        Number of 1-h files accepted as precipitating periods.
    num_lines_used : int
        Total number of 6-second samples included in the accumulation.
    """
    files = get_files(site, start_time, end_time)

    accum_matrix = np.zeros((32, 32))
    num_files_used = 0
    num_lines_used = 0

    for file in files:
        # Count lines to check for precipitation
        with open(file, 'r') as fh:
            num_lines = sum(1 for _ in fh)

        if num_lines <= MIN_OBS:
            print(f'Skipping (non-precipitating): {os.path.basename(file)}')
            continue

        num_lines_used += num_lines
        num_files_used += 1
        count = 0

        with open(file, 'r') as fh:
            for line in fh:
                if len(line) < 50:
                    # Header line: log the date/hour being processed
                    date_str = line[0:10]
                    hour_str = line[11:13]
                    print(f'  Processing {date_str} {hour_str}:00 UTC')
                else:
                    data = line.split(',')[62:]  # extract the 32×32 matrix columns
                    if len(data) == 1024:
                        # The first and last tokens require trimming due to
                        # inconsistent delimiters at line boundaries.
                        data[0] = data[0][-1]
                        data[-1] = data[-1][0]

                        arr = np.array(data).astype(float)
                        matrix = arr.reshape(32, 32)
                        matrix[0:3, :] = 0  # zero smallest 3 diameter bins (Yuter et al. 2006)
                        accum_matrix += matrix
                        count += 1

        print(f'  {count} samples accumulated from {os.path.basename(file)}')

    return accum_matrix, num_files_used, num_lines_used


def save_data(site, accum_matrix, num_files_used, num_lines_used, event_type):
    """
    Save accumulated PARSIVEL data for a site/event-type to disk.

    Three NumPy binary files are written to ``data_save_path/<site>/``:

    * ``PARSIVEL_Matrix_<event_type>.npy``  — the 32×32 accumulation matrix
    * ``num_files_used_<event_type>.npy``   — count of accepted 1-h files
    * ``num_lines_used_<event_type>.npy``   — total 6-second samples used

    Parameters
    ----------
    site : str
        Site name (e.g. ``'Highland'``).
    accum_matrix : numpy.ndarray, shape (32, 32)
        Accumulated drop-count matrix.
    num_files_used : int
        Number of 1-h files used in the accumulation.
    num_lines_used : int
        Total number of 6-second samples used.
    event_type : str
        Event-type label (e.g. ``'FR'``, ``'PF'``, ``'SIVT'``, ``'ALL'``).
    """
    out_dir = os.path.join(data_save_path, site)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f'PARSIVEL_Matrix_{event_type}.npy'), accum_matrix)
    np.save(os.path.join(out_dir, f'num_files_used_{event_type}.npy'),  num_files_used)
    np.save(os.path.join(out_dir, f'num_lines_used_{event_type}.npy'),  num_lines_used)
    print(f'Saved {event_type} data for {site} → {out_dir}')


# ------------------------------------------------------------------
# Run the accumulation for Highland across all event periods
# ------------------------------------------------------------------
print(f'Processing PARSIVEL data — period: {period}')

accum_matrix = np.zeros((32, 32))
num_files_used = 0
num_lines_used = 0

for start_time, end_time in periods_to_analyze:
    print(f'Reading: {start_time}  {end_time}')
    mat, nf, nl = parse_data('Highland', pd.Timestamp(start_time), pd.Timestamp(end_time))
    accum_matrix   += mat
    num_files_used += nf
    num_lines_used += nl

save_data('Highland', accum_matrix, num_files_used, num_lines_used, period)
