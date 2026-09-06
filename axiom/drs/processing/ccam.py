"""Pre and post-processing functions for CCAM."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import axiom.drs.utilities as adu
import axiom.utilities as au
import xarray as xr
import math
import cftime
import calendar as pycal

def add_month(year, month):
    """Add one month to the given year and month, adjusting the year if necessary."""
    if month == 12:
        return year + 1, 1
    else:
        return year, month + 1


def get_midpoint(year, month, calendar='standard'):
    """Returns the midpoint of the specified month, accounting for the calendar type.

    Args:
        year (int): The year.
        month (int): The month (1-12).
        calendar_type (str): The type of calendar.

    Returns:
        datetime: Midpoint of the month.
    """
    logger = au.get_logger(__name__)
    if calendar == '360_day':
        days_in_month = 30
    elif calendar in ['365_day', 'noleap']:
        days_in_month = 28 if month ==2 else pycal.monthrange(2001, month)[1]
    else:
        # Default to 'standard' or 'gregorian' (leap years included)
        days_in_month = pycal.monthrange(year, month)[1]

    total_hours = days_in_month * 24
    half_hours = total_hours / 2

    start = datetime(year, month, 1, 0 ,0)
    midpoint = start + timedelta(hours=half_hours)
    if calendar == '360_day':
        return cftime.Datetime360Day(midpoint.year, midpoint.month, midpoint.day,
                                     midpoint.hour, midpoint.minute, midpoint.second)
    elif calendar in ['365_day', 'noleap']:
        return cftime.DatetimeNoLeap(midpoint.year, midpoint.month, midpoint.day,
                                     midpoint.hour, midpoint.minute, midpoint.second)
    else:
        return midpoint

def center_times(ds, output_frequency):
    """Centers the times in the dataset.

    Args:
        ds (xarray.Dataset): Data.

    Returns:
        xarray.Dataset : Data with times centered.
    """
    logger = au.get_logger(__name__)
    # non-monthly data is simple, just halve the delta
    if output_frequency != '1M':
        original_calendar = ds.time.encoding.get('calendar')
        dt = ds.time.data[1] - ds.time.data[0]
        ds['time'] = ds.time + (dt / 2)
        ds.time.encoding['calendar'] = original_calendar
        ds['time'].attrs['units'] = 'days since 1950-01-01'
        return ds

    reference_date = datetime(1950, 1, 1, 0, 0)
    original_calendar = ds.time.encoding.get('calendar')

    times = ds['time'].values
    years = [int(str(dt)[:4]) for dt in times]

    # Check if the first month is in a different year than the rest
    if len(set(years)) > 1:
        shift_months = True
    else:
        shift_months = False

    adjusted_times = []
    for dt in ds.time.values:
        try:
            year, month = dt.year, dt.month  # works for datetime or cftime
        except AttributeError:
            # fallback if dt is string or numpy.datetime64
            dt_str = str(dt)
            year, month = int(dt_str[:4]), int(dt_str[5:7])

        # Shift the month forward by one if needed
        if shift_months:
            year, month = add_month(year, month)

        adjusted_times.append(get_midpoint(year, month, original_calendar))

    # Update the time coordinates with the new values
    ds['time'] = xr.DataArray(adjusted_times, dims='time')
    ds['time'].attrs['units'] = ds['time'].attrs.get('units', 'days since 1950-01-01 00:00:00')
    ds.time.encoding['calendar'] = original_calendar
    return ds


def generate_time_bounds(resampled_ds, output_frequency):
    """Generates time_bnds for resampled data

    Args:
        ds (xarray.Dataset): Resampled Dataset
        output_frequency: The frequency the data was resampled to

    Returns:
        time_bnds (float)
    """
    logger = au.get_logger(__name__)
    start_times = []
    end_times = []

    # Determine the calendar type from the dataset
    calendar_type = resampled_ds['time'].dt.calendar

    # Loop through each time value in the resampled dataset
    for current_date in resampled_ds['time'].values:

        # Check if cftime object (for non-standard calendars)
        if isinstance(current_date, cftime.datetime):
            current_datetime = current_date
            if output_frequency == '1H':
                start_time = current_datetime - timedelta(minutes=30)
                end_time = current_datetime + timedelta(minutes=30)
            elif output_frequency == '6H':
                start_time = current_datetime - timedelta(hours=3)
                end_time = current_datetime + timedelta(hours=3)
            elif output_frequency == '1D':
                start_time = current_datetime - timedelta(hours=12)
                end_time = current_datetime + timedelta(hours=12)
            elif output_frequency == '5min':
                start_time = current_datetime - timedelta(seconds=150)
                end_time = current_datetime + timedelta(seconds=150)
            elif output_frequency == '1M':
                start_time = cftime.datetime(current_datetime.year, current_datetime.month, 1, calendar=calendar_type)
                if current_datetime.month == 12:
                    end_time = cftime.datetime(current_datetime.year + 1, 1, 1, calendar=calendar_type)
                else:
                    end_time = cftime.datetime(current_datetime.year, current_datetime.month + 1, 1, calendar=calendar_type)
            else:
                raise ValueError(f'Unsupported frequency: {output_frequency}')
        else:
            # Handle numpy datetime64 objects
            if output_frequency == '1H':
                start_time = current_date + np.timedelta64(-30, 'm')
                end_time = current_date + np.timedelta64(30, 'm')
            elif output_frequency == '6H':
                start_time = current_date + np.timedelta64(-3, 'h')
                end_time = current_date + np.timedelta64(3, 'h')
            elif output_frequency == '1D':
                start_time = current_date + np.timedelta64(-12, 'h')
                end_time = current_date + np.timedelta64(12, 'h')
            elif output_frequency == '5min':
                start_time = current_date + np.timedelta64(-150, 's')
                end_time = current_date + np.timedelta64(150, 's')
            elif output_frequency == '1M':
                year = current_date.astype('datetime64[Y]').astype(int) + 1970
                month = (current_date.astype('datetime64[M]').astype(int) % 12) + 1
                start_time = np.datetime64(datetime(year, month, 1))
                if month == 12:
                    end_time = np.datetime64(datetime(year + 1, 1, 1))
                else:
                    end_time = np.datetime64(datetime(year, month + 1, 1))
            else:
                raise ValueError(f'Unsupported frequency: {output_frequency}')

        # Append the calculated times to the lists
        start_times.append(start_time)
        end_times.append(end_time)
    # Create an xarray DataArray for time bounds
    time_bnds = xr.DataArray(
        data=np.array([start_times, end_times]).T,
        dims=['time', 'bnds'],  # Define dimensions
        name='time_bnds'  # Name the DataArray
    )

    # Set the time for time_bnds from the resampled dataset
    time_bnds['time'] = resampled_ds['time']

    return time_bnds


def _detect_version(ds):
    """The CCAM version can be detected from the history metadata.

    Args:
        ds (xarray.Dataset): Dataset

    Returns:
        str : Version.
    """
    history = ds.attrs['history']
    yymm = datetime.strptime(history.split()[2], '%Y-%m-%d').strftime('%y%m')
    return yymm


def _set_version_metadata(ds, version):
    """Set the version metadata on the DataSet.

    Args:
        ds (xarray.DataSet): Dataset.
        version (str): Version string.

    Returns:
        xarray.Dataset : Dataset with version metadata on it.
    """
    ds.attrs['rcm_model'] = f'CCAM-{version}'
    ds.attrs['rcm_model_cordex'] = f'CCAM-{version}'
    ds.attrs['rcm_model_version'] = version
    ds.attrs['rcm_version'] = version
    ds.attrs['rcm_version_cordex'] = version
    return ds


def preprocess_ccam(ds, **kwargs):
    """Preprocess the data upon loading for CORDEX requirments.

    Args:
        ds (xarray.Dataset): Dataset.
        variable (str): Variable to extract along with bnds. Must be used as part of a lambda in open_mfdataset

    Returns:
        xarray.Dataset: Dataset with preprocessing applied.
    """
    variable = kwargs['variable']

    # Map raw CCAM variables to standard DRS variable names
    if variable == 'LUTYPE' and 'vegt' in ds.variables:
        ds = ds.rename({'vegt': 'LUTYPE'})
    elif variable == 'SOILTYPE' and 'soilt' in ds.variables:
        ds = ds.rename({'soilt': 'SOILTYPE'})

    # Rename metadata keys if needed
    if 'rlat0' in ds.attrs.keys():
        ds.attrs['rlon'] = ds.attrs.pop('rlong0')
        ds.attrs['rlat'] = ds.attrs.pop('rlat0')

    # Automatically detect version from inputs
    if 'model_id' not in kwargs['kwargs'].keys():
        version = _detect_version(ds)
    else:
        version = kwargs['kwargs']['model_id'].split('-')[-1]

    ds = _set_version_metadata(ds, version)

    # Check if time_bnds and height coordinates exist
    _has_time_bnds, tbnds = has_time_bnds(ds)
    _has_height, hcoord = has_height(ds, kwargs['variable'])

    # Start with the basic list of variables to keep
    vars_to_keep = ['lat_bnds', 'lon_bnds', 'crs']

    # Include time_bnds if present 
    if _has_time_bnds:
        vars_to_keep.append(tbnds)

    # Include height coordinate if present
    if _has_height and hcoord:
        vars_to_keep.append(hcoord)

    # Finally, include the main variable
    if variable:
        vars_to_keep.insert(0, variable)  # Ensure variable is first

    ds = ds[vars_to_keep]

    return ds


def postprocess_ccam(ds, **kwargs):
    """For CORDEX processing, there is some minor postprocessing that happens.

    Args:
        ds (xarray.Dataset): Data.

    Returns:
        xarray.Dataset: Data with postprocessing applied.
    """
    logger = au.get_logger(__name__)

    # Strip out the extra metadata keys (Marcus 20220802)
    remove_keys = 'ensemble,rcm_institute,rcm_model_cordex,rcm_model,rcm_version_cordex'.split(',')
    for rk in remove_keys:
        if rk in ds.attrs.keys():
            logger.debug(f'Removing metadata key {rk}')
            ds.attrs.pop(rk)

    # Strip out the extra dimensions from bnds (reduces filesize considerably)
    if 'lat_bnds' in ds.data_vars.keys():

        # Drop surplus coordinates
        ds['lat_bnds'] = au.isolate_coordinate(ds.lat_bnds, 'lat', drop=True)
        ds['lon_bnds'] = au.isolate_coordinate(ds.lon_bnds, 'lon', drop=True)
        if not adu.is_time_invariant(ds):
            if 'time' in ds['crs'].dims:
                ds['crs'] = ds['crs'].isel(time=0).drop('time') # drop time dimension for crs
            _has_height_attr, hcoord = has_height_attr(ds, kwargs['variable'])
            if _has_height_attr and 'time' in ds[hcoord].dims:
                ds[hcoord] = ds[hcoord].isel(time=0).drop('time')

    # Center the times for non-instantaneous data.
    _is_instantaneous_or_fixed = is_instantaneous_or_fixed(ds, kwargs['variable'])
    _resampling_applied = kwargs['resampling_applied']
    _output_frequency = kwargs['output_frequency']

    logger.debug(f'is_instantaneous_or_fixed = {_is_instantaneous_or_fixed}')
    logger.debug(f'resampling_applied = {_resampling_applied}')
    if _resampling_applied == True:
        logger.debug(f"TIME CENTERING TRIGGERED")
        ds = center_times(ds, output_frequency=_output_frequency)
        logger.debug(f"GENERATING TIME BOUNDS")
        ds['time_bnds'] = generate_time_bounds(ds, output_frequency=_output_frequency)
        ds['time'].attrs['axis'] = 'T'
        ds['time'].attrs['standard_name'] = 'time'
        ds['time'].attrs['bounds'] = 'time_bnds'

        # remove units from time
        if 'units' in ds['time'].attrs:
            del ds['time'].attrs['units']

    return ds


def is_instantaneous_or_fixed(ds, variable):
    """Checks for the presence of CCAM-specific flags indicating that a variable is instantaneous.

    Args:
        ds (xarray.Dataset): Data.
        variable (str): Variable currently being processed.
Returns:
        bool: True if the variable is instantaneous or fixed, False otherwise.
    """
    logger = au.get_logger(__name__)

    # Safety check: if variable isn't in dataset, we can't check it
    if variable not in ds:
        logger.debug(f"ERROR: Variable '{variable}' not found in the dataset.")
        return True

    # Get cell_methods, default to empty string if missing
    cell_methods = ds[variable].attrs.get('cell_methods', '')

    # If cell_methods is empty string, assume instantaneous
    if not cell_methods:
        return True

    instantaneous_patterns = ['time: point', 'time: fixed']
    if any(pattern in cell_methods for pattern in instantaneous_patterns):
        return True

    return False


def has_height(ds, variable):
    """Checks for the presence of a scalar coordinate (e.g., a fixed height like h2 or height) 
    indicating that a variable is at a single level.

    Args:
        ds (xarray.Dataset): Data.
        variable (str): Variable currently being processed.

    Returns:
        tuple: (bool, str or None) - True and name of scalar coordinate if present, else False and None.
    """
    da = ds[variable]
    for name, coord in da.coords.items():
        if coord.ndim == 0:
            return True, name
    return False, None


def has_height_attr(ds, variable):
    """Checks for the presence of attribute 'coordinates'
    indicating that a variable is at a single level.

    Args:
        ds (xarray.Dataset): Data.
        variable (str): Variable currently being processed.

    Returns:
        tuple: (bool, str or None) - True and name of the coordinate if present, else False and None.
    """
    da = ds[variable]
    hcoordinate = da.attrs.get('coordinates', None)

    # if coordinates is present
    if 'coordinates' in da.attrs.keys():
        return True, hcoordinate

    return False, None


def has_time_bnds(ds):
    """Checks if the dataset contains a time bounds variable.

    Args:
        ds (xarray.Dataset): The dataset to check.

    Returns:
        tuple: (bool, str or None) - True and name of the bounds variable, else False and None.
    """
    # Check for variable 'time_bnds' in dataset
    if 'time_bnds' in ds.variables:
        return True, 'time_bnds'

    # check 'bounds' attribute of the time_coordinate to determine time_bnds name
    # handle files where bounds might be named 'tbnds', etc.
    time_vars = [v for v in ds.coords if ds[v].attrs.get('axis') == 'T' or 'time' in v.lower()]
    for t_var in time_vars:
        bnds_attr = ds[t_var].attrs.get('bounds')
        if bnds_attr in ds.variables:
            return True, bnds_attr

    return False, None
