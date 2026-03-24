#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to read and plot Micro Rain Radar (MRR) data from the Wasatch Deployment
(2022-2024). Provides functions for data ingest, CFAD generation, and time-height
plotting.
"""

# %% imports
import os
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

np.seterr(divide='ignore')  # suppress divide-by-zero warnings

HOME = '/path/to/home/directory/'
DATA_DIR = os.path.join(HOME, '/path/to/MRR/data/')

variable_config = {
    'Ze': {
        'long_name': 'reflectivity',
        'units': 'dBZ',
        'label': 'Reflectivity $[dBZ_e]$',
        'cmap': plt.cm.terrain,
        'vmin': -10,
        'vmax': 40},
    'spectralWidth': {
        'long_name': 'spectral_width',
        'units': 'm_s',
        'label': 'Spectral \nWidth \n[m s$^{-1}$]',
        'cmap': plt.get_cmap('jet'),
        'vmin': 0,
        'vmax': 2},
    'W': {
        'long_name': 'upward_radial_velocity',
        'units': 'm_s',
        'label': 'Radial Velocity $V_R$ [m s$^{-1}$]',
        'cmap': plt.get_cmap('bwr'),
        'vmin': -4,
        'vmax': 4},
}

YLIM = [None, None]


def get_files(site, range_gate, start_time, end_time):
    """
    Return a list of absolute file paths for a given site/range-gate combination
    between ``start_time`` and ``end_time``.

    Parameters
    ----------
    site : str
        Site name (e.g. ``'Alta'``).
    range_gate : int
        Range gate spacing in metres (e.g. ``30`` or ``180``).
    start_time : datetime-like
        Start of the desired time window.
    end_time : datetime-like
        End of the desired time window.

    Returns
    -------
    list of str
        Absolute paths to NetCDF files that exist on disk.
    """
    data_dir = os.path.join(DATA_DIR, f'{site}/MRR/NetCDF')
    date_range = pd.date_range(start=start_time.date(), end=end_time.date())

    files = []
    for date in date_range:
        year = f'{date.year}'
        month = f'{date.month:02d}'
        day = f'{date.day:02d}'
        file = f'{year}{month}/condensed_processed_{month}{day}_{range_gate:03d}m.nc'
        file_path = os.path.join(data_dir, file)

        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print(f"File not found for {site} on {date:%Y-%m-%d}")

    return files


def parse(files):
    """
    Read MRR NetCDF files and return a cleaned xarray Dataset.

    Processing steps applied:

    * Scalar variables (``MRR_elevation``, ``lat``, ``lon``) are promoted to
      dataset attributes.
    * The ``range`` dimension is replaced with ``height_ASL`` (height above sea
      level in metres).  A secondary ``height`` coordinate (height above the MRR)
      is also added.
    * Duplicate and out-of-order time steps are removed.
    * Values outside [-1000, 1000] are masked as NaN.
    * The sign of the vertical velocity ``W`` is flipped so that negative values
      indicate motion toward the MRR.

    Parameters
    ----------
    files : list of str
        Paths to NetCDF files to read.

    Returns
    -------
    xr.Dataset
    """
    def extract_excess_var(data, var):
        if var in data:
            if data[var].ndim > 0:
                return data[var][0].values
            else:
                return data[var].item()
        return None

    nfiles = len(files)
    if nfiles > 1:
        data = xr.open_mfdataset(files)
    elif nfiles == 1:
        data = xr.open_dataset(files[0])
    else:
        raise ValueError("No files found!")

    data.attrs.update({
        var: extract_excess_var(data, var)
        for var in ['MRR_elevation', 'lat', 'lon']})

    # Format time and sort
    data['time'] = pd.to_datetime(data.time.values, unit='s')
    data = data.drop_duplicates(dim='time', keep=False)
    data = data.sortby('time')

    # Promote height to a coordinate before dropping the variable
    height = data.height.values[0]
    data = data.drop_vars(['MRR_elevation', 'lat', 'lon', 'height'])

    # Set height above sea level as the range dimension
    height_ASL = height + data.MRR_elevation
    data = data.rename_dims({'range': 'height_ASL'}).assign_coords(height_ASL=height_ASL)
    data = data.assign_coords(height=('height_ASL', height))

    # Mask physically unreasonable values
    for variable in list(data.keys()):
        conditions = (data[variable] > -1000) & (data[variable] < 1000)
        data[variable] = data[variable].where(conditions, other=np.nan)
        data[variable].attrs.update({
            'long_name': variable_config[variable]['long_name'],
            'units': variable_config[variable]['units']})

    # Flip sign of W so that negative values indicate downward motion
    data['W'] *= -1

    return data


def read_data(site, range_gate, start_time, end_time):
    """
    Read and return MRR data for a site/range-gate combination.

    Parameters
    ----------
    site : str
        Site name (e.g. ``'Alta'``).
    range_gate : int
        Range gate spacing in metres.
    start_time : datetime-like
        Start of the desired time window.
    end_time : datetime-like
        End of the desired time window.

    Returns
    -------
    xr.Dataset
    """
    files = get_files(site, range_gate, start_time, end_time)
    data = parse(files)

    data.attrs['site'] = site
    data.attrs['range_gate'] = range_gate

    data = data.sel(time=slice(start_time, end_time))

    return data


def merge_range_gates(data30, data180):
    """
    Merge datasets from the 30 m and 180 m range gates into a single Dataset
    with a ``range_gate`` dimension.

    Parameters
    ----------
    data30 : xr.Dataset
        Data from the 30 m range gate.
    data180 : xr.Dataset
        Data from the 180 m range gate.

    Returns
    -------
    xr.Dataset
    """
    data30 = data30.expand_dims({'range_gate': [30]})
    data180 = data180.expand_dims({'range_gate': [180]})
    merged = xr.merge([data30, data180], combine_attrs='drop_conflicts')
    return merged


def read_merged_data(site, start_time, end_time):
    """
    Read and merge data from both range gates (30 m and 180 m) for a site.

    Parameters
    ----------
    site : str
        Site name.
    start_time : datetime-like
        Start of the desired time window.
    end_time : datetime-like
        End of the desired time window.

    Returns
    -------
    xr.Dataset
    """
    data30 = read_data(site, 30, start_time, end_time)
    data180 = read_data(site, 180, start_time, end_time)
    return merge_range_gates(data30, data180)


def add_ground(ax, MRR_elevation, bottom):
    """
    Add a shaded ground polygon to *ax* spanning from *bottom* to
    *MRR_elevation* with a horizontal line at the MRR elevation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    MRR_elevation : float
        MRR elevation in metres MSL.
    bottom : float
        Lower bound of the shaded region (metres MSL).
    """
    ax.axhspan(bottom, MRR_elevation, fc='#d9b38c')
    ax.axhline(MRR_elevation, color='black')


def plot_var(data, variable, ax=None, ylim=YLIM):
    """
    Plot a single MRR variable as a time-height diagram.

    Parameters
    ----------
    data : xr.Dataset
        MRR dataset for a single site/range-gate.
    variable : str
        Variable name (``'Ze'``, ``'W'``, or ``'spectralWidth'``).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if not provided.
    ylim : list of two floats, optional
        [y_min, y_max] in metres MSL.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    config = variable_config[variable]

    data[variable].plot(x='time', y='height_ASL', ax=ax,
                        cmap=config['cmap'],
                        vmin=config['vmin'], vmax=config['vmax'],
                        cbar_kwargs={'label': config['label'], 'pad': 0.01})

    add_ground(ax, data.MRR_elevation, ylim[0])

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator, show_offset=True)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set(
        title=f'Site: {data.site}; Range Gate: {data.range_gate}',
        xlabel='Time [UTC]',
        ylabel='Height [m MSL]',
        ylim=ylim)

    return ax


def plot_all_vars(data, axes=None, ylim=YLIM, figsize=None):
    """
    Plot reflectivity, vertical velocity, and spectral width in a three-panel
    time-height diagram.

    Parameters
    ----------
    data : xr.Dataset
        MRR dataset for a single site/range-gate.
    axes : array-like of matplotlib.axes.Axes, optional
        Three axes to draw on.  A new figure is created if not provided.
    ylim : list of two floats, optional
        [y_min, y_max] in metres MSL.
    figsize : tuple, optional
        Figure size passed to :func:`matplotlib.pyplot.subplots`.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
    """
    if axes is None:
        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=figsize)

    variables = ['Ze', 'W', 'spectralWidth']
    for var, ax in zip(variables, axes):
        plot_var(data, var, ax, ylim)
        ax.set(title=None, xlabel=None, ylabel=None)

    axes[0].set(title=f'Site: {data.site}; Range Gate: {data.range_gate}')
    axes[1].set(ylabel='Height [m MSL]')
    axes[2].set(xlabel='Time [UTC]')

    return axes


def plot_compare_data(data1, data2, variable='all', axes=None, ylim=YLIM,
                      figsize=None):
    """
    Side-by-side comparison of two MRR datasets.

    Parameters
    ----------
    data1 : xr.Dataset
    data2 : xr.Dataset
    variable : str, optional
        Variable to plot, or ``'all'`` for all three variables.
    axes : array-like of matplotlib.axes.Axes, optional
    ylim : list of two floats, optional
    figsize : tuple, optional

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
    """
    nrows = 3 if variable == 'all' else 1
    if axes is None:
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize,
                                 sharex=True, sharey=True)

    if variable == 'all':
        data1_axes = axes[:, 0]
        data2_ylabel_ax = axes[1, 1]
        plot_all_vars(data1, data1_axes, ylim)
        plot_all_vars(data2, axes[:, 1], ylim)
    else:
        data1_axes = [axes[0]]
        data2_ylabel_ax = axes[1]
        plot_var(data1, variable, data1_axes[0], ylim)
        plot_var(data2, variable, axes[1], ylim)

    # Remove colorbar from left column to avoid duplication
    for ax in data1_axes:
        ax.collections[-1].colorbar.remove()

    # Remove redundant ylabel from right column
    data2_ylabel_ax.set(ylabel=None)

    return axes


def hist2d(var1_arr, var2_arr, var1_bins=None, var2_bins=None,
           var1_bins_start=None, var1_bins_stop=None, var1_bins_width=None,
           var2_bins_start=None, var2_bins_stop=None, var2_bins_width=None):
    """
    Compute a 2-D histogram of two variables.

    Bin edges can be supplied directly via *var1_bins*/*var2_bins*, or
    constructed from start/stop/width parameters.

    Returns
    -------
    hist : numpy.ndarray, shape (M, N)
    var1_edges : numpy.ndarray, shape (M+1,)
    var2_edges : numpy.ndarray, shape (N+1,)
    """
    if any((var1_bins, var2_bins)) and any((var1_bins_start, var1_bins_stop,
                                            var2_bins_start, var2_bins_stop,
                                            var1_bins_width, var2_bins_width)):
        raise ValueError("Cannot supply both bins and start/stop/width arguments.")

    if var1_bins is None:
        var1_bins = np.arange(var1_bins_start, var1_bins_stop, var1_bins_width)
    if var2_bins is None:
        var2_bins = np.arange(var2_bins_start, var2_bins_stop, var2_bins_width)

    hist, var1_edges, var2_edges = np.histogram2d(var1_arr, var2_arr,
                                                  bins=[var1_bins, var2_bins])
    return hist, var1_edges, var2_edges


def hist2d_height(data, variable, variable_bins_width,
                  variable_bins_start, variable_bins_stop,
                  height_dim='height_ASL', height_bins_width=200,
                  height_bins_start=1300, height_bins_stop=5600):
    """
    Compute a 2-D histogram with height on one axis and a MRR variable on the
    other, aggregated over all time steps.

    Parameters
    ----------
    data : xr.Dataset
    variable : str
    variable_bins_width : float
    variable_bins_start : float
    variable_bins_stop : float
    height_dim : str, optional
    height_bins_width : float, optional
    height_bins_start : float, optional
    height_bins_stop : float, optional

    Returns
    -------
    hist : numpy.ndarray, shape (M, N)
        Rows correspond to height bins; columns to variable bins.
    height_edges : numpy.ndarray, shape (M+1,)
    variable_edges : numpy.ndarray, shape (N+1,)
    """
    N = data.time.size
    height_values = np.tile(data[height_dim].values, N)
    variable_values = data[variable].values.ravel()

    results = hist2d(
        var1_arr=height_values, var2_arr=variable_values,
        var1_bins_start=height_bins_start,
        var1_bins_stop=height_bins_stop,
        var1_bins_width=height_bins_width,
        var2_bins_start=variable_bins_start,
        var2_bins_stop=variable_bins_stop,
        var2_bins_width=variable_bins_width)

    hist, height_edges, variable_edges = results
    return hist, height_edges, variable_edges


def CFAD(data, variable='Ze', variable_bins_width=1.5,
         variable_bins_start=-20, variable_bins_stop=40,
         top='standard', height_bins_width=None, height_bins_start=None,
         height_bins_stop=None, ax=None, cmap='viridis', cbar=True,
         vmax=None, vmin=None, bottom=None, title=None,
         zero_line=False, comparison_median=False, legend=False):
    """
    Create a Contoured Frequency by Altitude Diagram (CFAD).

    Frequency is computed relative to the total number of time steps, i.e.
    the fraction of observations (expressed as a percentage) that fall within
    each height/variable bin.  The 25th percentile, 75th percentile, and
    median profiles are overlaid.

    Parameters
    ----------
    data : xr.Dataset
        MRR dataset for a single site/range-gate.
    variable : str, optional
        Variable to plot (default ``'Ze'``).
    variable_bins_width : float, optional
        Bin width for the variable axis.
    variable_bins_start : float, optional
        Lower bound of variable bins.
    variable_bins_stop : float, optional
        Upper bound of variable bins.
    top : {'standard', 'data'} or float, optional
        Upper y-limit.  ``'standard'`` uses the MRR maximum range;
        ``'data'`` trims to the highest range gate with observations;
        a numeric value sets the limit explicitly.
    height_bins_width : float, optional
        Height bin width in metres.  Defaults to the range gate spacing.
    height_bins_start : float, optional
        Lower height bin edge in metres MSL.  Defaults to MRR elevation.
    height_bins_stop : float, optional
        Upper height bin edge in metres MSL.  Defaults to the top of the data.
    ax : matplotlib.axes.Axes, optional
    cmap : str or Colormap, optional
    cbar : bool, optional
        Whether to draw a colorbar (currently unused; reserved for future use).
    vmax : float, optional
    vmin : float, optional
    bottom : float, optional
        Lower y-limit in metres MSL.
    title : str, optional
        Custom axes title.  Auto-generated from site/time metadata if omitted.
    zero_line : bool, optional
        Draw a vertical line at x=0 (useful for velocity variables).
    comparison_median : bool, optional
        Overlay a vertical line at the height-averaged median value.
    legend : bool, optional
        Display a legend below the axes.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if height_bins_width is None:
        height_bins_width = data.range_gate
    if height_bins_start is None:
        height_bins_start = data.MRR_elevation
    if height_bins_stop is None:
        height_bins_stop = data.height_ASL[-1] + data.range_gate

    # Compute 2-D histogram
    results = hist2d_height(data, variable, variable_bins_width,
                            variable_bins_start, variable_bins_stop,
                            'height_ASL', height_bins_width,
                            height_bins_start, height_bins_stop)
    hist, height_edges, variable_edges = results

    # Determine upper y-limit
    if top == 'standard':
        y_max = None
    elif top == 'data':
        consecutive_nan_count = 0
        for row in hist[::-1]:
            if np.all(row == 0):
                consecutive_nan_count += 1
            else:
                break
        y_max = height_edges[-1 * consecutive_nan_count + 1]
    elif isinstance(top, (int, float)):
        y_max = top

    # Determine lower y-limit
    if bottom is None:
        bottom = data.MRR_elevation - 2 * data.range_gate

    if ax is None:
        fig, ax = plt.subplots()

    # Frequency relative to total observation time
    freq = (hist / len(data.time)) * 100
    nan_freq = np.where(freq == 0, np.nan, freq)

    # Plot CFAD
    x_mesh, y_mesh = np.meshgrid(variable_edges, height_edges)
    p = ax.pcolormesh(x_mesh, y_mesh, nan_freq, cmap=cmap, vmax=vmax, vmin=vmin,
                      edgecolors='white', linewidth=0.4, antialiased=True)

    # Ground polygon
    add_ground(ax, data.MRR_elevation, 0)

    # Optional zero-line
    if zero_line:
        ax.axvline(x=0.0, color='#c20893', linestyle='-')

    # Percentile and median profiles
    q25 = np.nanpercentile(data[variable].values, 25, axis=0)
    q75 = np.nanpercentile(data[variable].values, 75, axis=0)
    ax.plot(q25, data.height_ASL, color='black', linestyle='--',
            label='25th Percentile', lw=2)
    ax.plot(q75, data.height_ASL, color='black', linestyle='--',
            label='75th Percentile', lw=2)
    median = data[variable].median(dim='time')
    median.plot(y='height_ASL', ax=ax, color='red', label='Median')

    # Optional column-averaged median line
    if comparison_median:
        median_avg = median.mean().values.item()
        ax.axvline(x=median_avg, color='black', linestyle='-',
                   label=f'Avg of Median = {median_avg:.3f}')

    # Title
    if title is None:
        start, end = pd.to_datetime([data.time.values[0], data.time.values[-1]])
        title = (f'Contoured Frequency by Altitude - {data.site}\n'
                 f'{data.range_gate} m Range Gate\n'
                 f'{start:%Y-%m-%d %H:%M} \u2013 {end:%Y-%m-%d %H:%M}')

    if legend:
        ax.legend(bbox_to_anchor=(.5, -0.2), loc='upper center')

    ax.set(
        ylim=(bottom, y_max),
        title=title,
        xlabel=variable_config[variable]['label'],
        ylabel='Height [m MSL]')

    return ax


def sideBYside(sites, range_gate, start_time, end_time):
    """
    Compare two MRRs in close proximity over the same time period.

    Plots a three-panel time-height diagram and a reflectivity CFAD for the
    difference field (site[0] minus site[1]).

    Parameters
    ----------
    sites : list of str
        Two-element list of site names to compare.
    range_gate : int
        Range gate spacing in metres.
    start_time : datetime-like
    end_time : datetime-like
    """
    data1 = read_data(sites[0], range_gate, start_time, end_time)
    data2 = read_data(sites[1], range_gate, start_time, end_time)

    diff = data1 - data2

    CFAD(diff)
    plot_all_vars(diff)
    
def open_ds(data_path, period):
    """
    Open the 30-m and 180-m range-gate MRR datasets for both sites
    for a given event-type period (e.g. 'SIVT', 'FR', 'PF', 'ALL').
    """
    highland180 = xr.open_dataset(data_path + f'Highland/highland180_{period}.nc')
    highland30  = xr.open_dataset(data_path + f'Highland/highland030_{period}.nc')
    alta180     = xr.open_dataset(data_path + f'Alta/alta180_{period}.nc')
    alta30      = xr.open_dataset(data_path + f'Alta/alta030_{period}.nc')
    return highland180, highland30, alta180, alta30


if __name__ == '__main__':
    import datetime as dt

    end_time = dt.datetime(2022, 12, 14, 21, 0)
    start_time = end_time - dt.timedelta(hours=48)

    for site in ['Alta', 'Highland']:
        for range_gate in [30, 180]:
            data = read_data(site, range_gate, start_time, end_time)
            CFAD(data, vmax=2.5, top='standard')
            plt.show()
