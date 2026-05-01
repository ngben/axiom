"""Main entrypoint for DRS processing."""
from genericpath import isfile
import os
import argparse
from datetime import datetime
from uuid import uuid4
import xarray as xr
import axiom.utilities as au
import axiom.drs.utilities as adu
from axiom.drs.domain import Domain
import axiom.schemas as axs
import json
import sys
from distributed import Client, LocalCluster
from axiom.config import load_config
from axiom import __version__ as axiom_version
from axiom.exceptions import NoFilesToProcessException, DRSContextInterpolationException
import shutil
from dask.distributed import progress, wait
import numpy as np
from axiom.supervisor import Supervisor
from axiom.drs.processing.ccam import is_instantaneous_or_fixed
from axiom.drs.processing.ccam import has_height
from axiom.drs.processing.ccam import has_height_attr
import cftime
import re
import pandas as pd

DATASET_TABLE = None
def load_dataset_table():
    """
    Centralized function to load the CORDEX datasets.csv file once.
    """
    global DATASET_TABLE
    if DATASET_TABLE is not None:
        return True

    url = "https://raw.githubusercontent.com/WCRP-CORDEX/data-request-table/main/cmor-table/datasets.csv"
    local_path = os.path.join(au.get_installed_data_root(), 'datasets.csv')

    try:
        DATASET_TABLE = pd.read_csv(url)
        print("✅ Successfully loaded dataset from GitHub.")
        return True
    except Exception as e:
        print(f"🌐 Remote download failed: {e}")
        if os.path.exists(local_path):
            try:
                DATASET_TABLE = pd.read_csv(local_path)
                print(f"🏠 Loaded fallback version from {local_path}.")
                return True
            except Exception as local_e:
                print(f"⚠️ Error reading local file: {local_e}")
                return False
        else:
            print(f"❌ Critical Error: Remote failed and {local_path} not found.")
            return False

def get_official_cell_method(variable_id, freq):
    """
    Fetches the official cell_method from CSV file
    """
    if not load_dataset_table(): return None

    var_col = 'out_name' if 'out_name' in DATASET_TABLE.columns else 'variable_id'
    freq_col = 'frequency'

    if freq_col not in DATASET_TABLE.columns: return None

    try:
        # Case-insensitive lookup for the variable name
        match = DATASET_TABLE[
            (DATASET_TABLE[var_col].str.lower() == variable_id.lower()) & 
            (DATASET_TABLE[freq_col] == freq)
        ]

        if not match.empty:
            # Return the first matching cell_method
            return str(match.iloc[0]['cell_methods']).strip()
    except KeyError as e:
        print(f"⚠️ Table KeyError: Could not find column {e}. Available columns: {DATASET_TABLE.columns.tolist()}")
        return None

def consume(json_filepath):
    """Consume a json payload (for message passing)

    Args:
        json_filepath (str): Path to the JSON file.
    """
    logger = au.get_logger(__name__)

    # Check if the file has already been consumed
    consumed_filepath = json_filepath.replace('.json', '.consumed')
    if os.path.isfile(consumed_filepath):
        logger.info(
            f'{json_filepath} has already been consumed and needs to be cleaned up by another process. Terminating.')
        sys.exit()

    # Check if the file is locked
    if au.is_locked(json_filepath):
        logger.info(
            f'{json_filepath} is locked, possibly by another process. Terminating.')
        sys.exit()

    # Lock the file
    au.lock(json_filepath)

    # Convert to dict
    payload = json.loads(open(json_filepath, 'r').read())

    # Allow rerun of failed variables (do this after all other variables have been processed!)
    config = load_config('drs_shep')
    failures_path = f'{json_filepath}_001.failed'
    if config.rerun_failures and os.path.exists(failures_path):
        failed_variables = open(failures_path, 'r').read().splitlines()
        payload['variables'] = failed_variables

    # Process
    process_multi(**payload)

    # Mark consumed by touching another file.
    au.touch(consumed_filepath)

    # Unlock
    au.unlock(json_filepath)

    # Explicit exit (#125)
    sys.exit(0)

def process(
    input_files,
    output_directory,
    variable,
    project,
    model,
    domain_id,
    start_year, end_year,
    output_frequency,
    level=None,
    input_resolution=None,
    overwrite=True,
    preprocessor=None,
    postprocessor=None,
    **kwargs
):
    """Method to process a single variable/domain/resolution combination.

    Args:
        input_files (str or list): Globbable string or list of filepaths.
        output_directory (str) : Path from which to build DRS structure.
        variable (str): Variable to process.
        project (str): Project metadata to apply (loaded from user config).
        model (str): Model metadata to apply (loaded from user config).
        start_year (int): Start year.
        end_year (int): End year.
        output_frequency (str): Output frequency to process.
        input_resolution (float, optional): Input resolution in km. Leave black to auto-detect from filepaths.
        overwrite (bool): Overwrite the data at the destination. Defaults to True.
        preprocessor (str): Data preprocessor to activate on input data. Defaults to None.
        postprocesser (str): Data postprocess to activate before writing data. Defaults to None.
        **kwargs: Additional keyword arguments used in metadata interpolation.
    """

    # Start the clock
    timer = au.Timer()
    timer.start()

    # Capture what was passed into this method for interpolation context later.
    local_args = locals()

    # Load the logger and configuration
    logger = au.get_logger(__name__)
    config = load_config('drs_shep')

    # Dump the job id if available
    if 'PBS_JOBID' in os.environ.keys():
        jobid = os.getenv('PBS_JOBID')
        logger.info(f'My PBS_JOBID is {jobid}')

    logger.info('Searching for files matching the following path:')
    logger.info(input_files)

    # Get a list of the potential filepaths
    input_files = au.auto_glob(input_files)
    num_files = len(input_files)
    logger.debug(f'{num_files} to consider before filtering.')

    # Filter by those that actually have the variable in the filename.
    if config.filename_filtering['variable']:

        input_files = adu.filter_by_variable_name(input_files, variable)
        num_files = len(input_files)
        logger.debug(
            f'{num_files} to consider after filename variable filtering.')

    # Filter by those that actually have the year in the filename (plus or minus an offset).
    if config.filename_filtering['year']:

        input_files = filter_years(
            input_files, start_year, offset=config.filename_filtering['year_offset'])
        num_files = len(input_files)
        logger.debug(f'{num_files} to consider after filename year filtering.')

    # Is there anything left to process?
    if len(input_files) == 0:
        raise NoFilesToProcessException()

    # Dump filepaths prior to loading
    if config.get('dump_filepaths_prior_to_loading', default=False):
        logger.debug('Dumping filepaths prior to loading... there may be a lot.')
        for input_file in input_files:
            logger.debug(input_file)

    # Detect the input resolution if it it not supplied
    if input_resolution is None:
        logger.debug('No input resolution supplied, auto-detecting')
        input_resolution = adu.detect_resolution(input_files)
        logger.debug(f'Input resolution detected as {input_resolution} km')

    # Load project config
    logger.info(f'Loading project config ({project})')
    project_key = project
    project = load_config('projects')[project_key]

    # Ensure it actually exists
    assert isinstance(project, dict), f'Project {project_key} not found, does it exist in projects.json?'

    # Load model config
    logger.info(f'Loading model config ({model})')
    model_key = model
    model = load_config('models')[model_key]

    # Ensure it actually exists
    assert isinstance(model, dict), f'Model {model_key} not found, does it exist in models.json?'

    logger.debug(
        'Loading files into distributed memory, this may take some time.')

    # TODO: Remove!!!! This is just to make CCAM work in the short term
    if preprocessor is None and 'ccam' in input_files[0] and config.auto_detect_ccam == True:
        logger.warn('CCAM preprocessor override used')
        preprocessor = 'ccam'
        postprocessor = 'ccam'

    # Load a preprocessor, if one exists.
    preprocessor = adu.load_preprocessor(preprocessor)
    def preprocess(ds, *args, **kwargs): return preprocessor(ds, **local_args)

    # Load the open_dataset configuration
    open_dataset_kwargs = config['xarray']['open_dataset']

    # Account for fixed variables, if defined
    if 'variables_fixed' in project.keys() and variable in project['variables_fixed']:

        # Load just the first file
        ds = xr.open_dataset(input_files[0], **open_dataset_kwargs)
        ds = preprocess(ds, variable=variable)

    else:
        ds = xr.open_mfdataset(
            input_files,
            preprocess=preprocess,
            **open_dataset_kwargs
        )

    # remove height variable as a scalar coordinate
    _has_height, hcoord = has_height(ds, variable)
    if _has_height:
        ds = ds.reset_coords(hcoord, drop=False)
        ds[variable].attrs["coordinates"] = hcoord

    # Subset temporally
    if not adu.is_time_invariant(ds):
        logger.info(f'Subsetting times to {start_year}')
        time_slice = slice(f'{start_year}-01-01', f'{start_year}-12-31')
        ds['time'] = ds['time'].dt.round('1s')
        if 'time_bnds' in ds:
            ds['time_bnds'] = ds['time_bnds'].dt.round('1s')
        ds = ds.sel(time=time_slice, drop=True)

    # Skip over the file if subdaily resampling is disabled, this will stop 
    native_frequency = adu.detect_input_frequency(ds)

    # Ensure blank output frequency is indeed fixed and only one can be written
    if adu.is_time_invariant(ds):
        output_frequency = 'fx'
        overwrite = False

    logger.info(f'native_frequency = {native_frequency}, output_frequency = {output_frequency}')
    if config.allow_subdaily_resampling == False and native_frequency != output_frequency and 'H' in output_frequency:
        logger.info(f'Subdaily resampling has been disabled and input/output frequencies do not match, skipping {variable}.')
        return

    # Persist now, get it on the cluster while the rest of the metadata assembly continues
    ds = ds.persist()

    # Determine time-invariance
    time_invariant = 'time' not in list(ds.coords.keys())

    # Assemble the context object (order dependent!)
    logger.debug('Assembling interpolation context.')
    context = config.metadata_defaults.copy()

    # Add metadata from the input data
    context.update(ds.attrs)

    # Add user-supplied metadata
    context.update(kwargs)

    # Add project and model metadata
    context.update(project)
    context.update(model)

    # Add additional args
    context.update(local_args)
    context['res_km'] = input_resolution

    # Sort the dimensions (fixes domain subsetting)
    logger.debug('Sorting data')
    sort_coords = list()
    for coord in 'time,lev,lat,lon'.split(','):
        if coord in ds.coords.keys():
            sort_coords.append(coord)

    ds = ds.sortby(sort_coords)

    logger.debug('Applying metadata schema')

    # Load a user-supplied schema, if one exists.
    if 'schema' in kwargs.keys():
        schema_key = kwargs['schema']
    else:
        schema_key = config['default_schema']

    schema = axs.load_schema(schema_key)
    ds = au.apply_schema(ds, schema)

    logger.info(f'Parsing domain_id {domain_id}')
    if isinstance(domain_id, str):

        # Registered domain_id
        if adu.is_registered_domain(domain_id):
            domain_id = adu.get_domain(domain_id)

        # Attempt to parse
        else:
            domain_id = Domain.from_directive(domain_id)

    # We will only otherwise accept a domain object.
    elif isinstance(domain_id, Domain) == False:
        raise Exception(f'Unable to parse domain_id {domain_id}.')

    logger.debug('Domain: ' + domain_id.to_directive())
    rounding = int(domain_id.rounding)

    # Subset the geographical domain
    logger.debug('Subsetting geographical domain.')
    ds = domain_id.subset_xarray(ds, drop=True)

    # TODO: Need to find a less manual way to do this.
    for year in adu.generate_years_list(start_year, end_year):

        logger.info(f'Processing {year}')

        # Subset the data into just this year
        if not time_invariant:
            time_slice = slice(f'{year}-01-01', f'{year}-12-31')
            _ds = ds.sel(time=time_slice, drop=True)
        else:
            _ds = ds.copy()

        # Historical cutoff is defined in $HOME/.axiom/drs.json
        if config.enable_historical_cutoff == True:
            context['experiment'] = 'historical' if year < config.historical_cutoff else context['rcp']

        logger.info(f'Native frequency of data detected as {native_frequency}')

        # Automatically detect the output_frequency from the input data, this will not require resampling

        # Flag to trigger cell_method update below.
        resampling_applied = False

        if output_frequency == 'from_input' or output_frequency == native_frequency:
            output_frequency = adu.detect_input_frequency(_ds)
            logger.info(
                f'output_frequency detected from inputs ({output_frequency})')
            logger.info(f'No need to resample.')
            # Map the frequency to something DRS-compliant
            context['frequency_mapping'] = config['frequency_mapping'][output_frequency]

        # Fixed variables, just change the frequency_mapping
        elif adu.is_time_invariant(_ds):
            output_frequency = 'fx'
            logger.info(
                'Data is time-invariant (fixed variable), overriding frequency_mapping to fx')
            context['frequency_mapping'] = 'fx'

        # Actually perform the resample
        else:
            logger.debug(f'Resampling to {output_frequency} mean.')
            context['frequency_mapping'] = config['frequency_mapping'][output_frequency]

            # Check if the time variable uses a non-standard calendar
            if isinstance(_ds.time.values[0], cftime.datetime):
                original_calendar = _ds.time.encoding.get('calendar', 'standard')
            else:
                original_calendar = 'standard'

            # Resample the data
            _ds = _ds.resample(time=output_frequency, label='left').mean()

            # Retain original calendar
            _ds.time.encoding['calendar'] = original_calendar
            _ds.time.attrs['calendar'] = original_calendar

            # Update the cell methods below
            resampling_applied = True

        # Start persisting the computation now
        _ds = _ds.persist()

        # Monthly data should have the days truncated
        context['start_date'], context['end_date'] = adu.get_start_and_end_dates(year, output_frequency)

        # Tracking info
        context['creation_date'] = datetime.utcnow().isoformat(timespec='seconds')+'Z'
        context['uuid'] = uuid4()

        # Interpolate context
        logger.info('Interpolating context.')
        context = adu.interpolate_context(context)

        # Assemble the global meta, add axiom details
        logger.debug('Assembling global metadata.')
        global_attrs = dict(
            axiom_version=axiom_version,
            axiom_schema=schema_key
        )

        for key, value in config.metadata_defaults.items():
            global_attrs[key] = str(value) % context

        # Strip and reapply metadata
        logger.debug('Applying metadata')
        _ds.attrs = global_attrs

        # Add in the variable to the context
        context['variable'] = variable

        # Reapply the schema
        logger.info('Reapplying schema')
        _ds = au.apply_schema(_ds, schema)

        # Copy coordinate attributes straight off the inputs
        if config.copy_coordinates_from_inputs:
            for coord in list(_ds.coords.keys()):
                _ds[coord].attrs = ds[coord].attrs

        # Assemble the encoding dictionaries (to ensure time units work!)
        logger.debug('Applying encoding')
        encoding = dict()

        for coord in list(_ds.coords.keys()):
            if coord not in config.encoding.keys():
                logger.warn(
                    f'Coordinate {coord} is not specified in drs.json file, omitting encoding.')
                continue
            encoding[coord] = config.encoding[coord]

        # Apply a blanket variable encoding.
        encoding[variable] = config.encoding['variables']
        encoding['lat_bnds'] = config.encoding['lat_bnds']
        encoding['lon_bnds'] = config.encoding['lon_bnds']
        encoding['crs'] = config.encoding['crs']

        # Add chunking output encoding
        num_dims = len(ds[variable].dims)
        if num_dims == 4:
#            encoding[variable]['chunksizes'] = (1, 1, 48, 48)
            encoding[variable]['chunksizes'] = (92, 1, 62, 82)
        elif num_dims == 3:
#            encoding[variable]['chunksizes'] = (1, 48, 48)
            encoding[variable]['chunksizes'] = (92, 62, 82)
        else:
            encoding[variable]['chunksizes'] = None

        # Postprocess data if required
        postprocessor = adu.load_postprocessor(postprocessor)

        def postprocess(_ds, *args, **kwargs):
            combined = dict()
            combined.update(kwargs)
            combined.update(local_args)
            combined['resampling_applied'] = resampling_applied
            combined['output_frequency'] = output_frequency

            return postprocessor(_ds, **combined)

        _ds = postprocess(_ds)

        logger.debug(f'Postprocessor done, continue postprocessing')

        # Update time_bnds encoding, drop time_bnds attributes
        if resampling_applied or not is_instantaneous_or_fixed(_ds, variable):
            _ds['time_bnds'].attrs = {}
            encoding['time_bnds'] = config.encoding['time_bnds']

        # Update height scalar coordinate encoding
        _has_height, hcoord = has_height_attr(ds, variable)
        if _has_height:
            encoding[hcoord] = config.encoding[hcoord]

        _ds = update_cell_methods(_ds, variable, output_frequency)

        # Update time_bounds after updating cell_methods
        from axiom.drs.processing.ccam import center_times, generate_time_bounds

        cell_methods = _ds[variable].attrs.get('cell_methods', '').lower()
        time_agg_patterns = ["time: mean", "time: maximum", "time: minimum"]
        is_time_aggregated = any(pattern in cell_methods for pattern in time_agg_patterns)
        has_time_bnds = 'time_bnds' in _ds.data_vars or 'time_bnds' in _ds.coords

        # only apply to data which is 1) not resampled, 2) does not have time_bounds, and 3) has cell_methods time aggregated
        if is_time_aggregated and not has_time_bnds and not resampling_applied:
            logger.info(f"Variable {variable} has aggregated cell_methods but missing time_bnds. Generating now.")

            # generate_time_bounds creates the actual bounds array
            _ds['time_bnds'] = generate_time_bounds(_ds, output_frequency=output_frequency)

            # Strip units/calendar from attrs to avoid the encoding conflict
            if 'units' in _ds['time'].attrs:
                del _ds['time'].attrs['units']

            # Ensure time_bnds is in the encoding for the netCDF write
            encoding['time_bnds'] = config.encoding['time_bnds']

        logger.debug(f'Postprocessing done')

        # Get the output format from config
        output_format = config.get('output_format', 'NETCDF4')

        # round lat/lon coords and convert to double
        _ds.coords['lon'] = _ds.coords['lon'].astype('float64')
        _ds.coords['lat'] = _ds.coords['lat'].astype('float64')
        _ds.coords['lon'] = _ds.coords['lon'].round(decimals=rounding)
        _ds.coords['lat'] = _ds.coords['lat'].round(decimals=rounding)
        _ds['lon_bnds'] = _ds['lon_bnds'].astype('float64')
        _ds['lat_bnds'] = _ds['lat_bnds'].astype('float64')
        _ds['lon_bnds'] = _ds['lon_bnds'].round(decimals=rounding+1)
        _ds['lat_bnds'] = _ds['lat_bnds'].round(decimals=rounding+1)

        # remove encoding in variable
        if 'coordinates' in _ds[variable].encoding:
            del _ds[variable].encoding['coordinates']

        # Get the full output filepath with string interpolation
        logger.debug('Working out output paths and chunking dataset')

        # CHUNK THE DATASET IF 5MIN
        if not time_invariant and output_frequency == '5min':
            unique_days = np.unique(_ds.time.dt.strftime('%Y-%m-%d').data)
            write_chunks = [_ds.sel(time=day) for day in unique_days]
            logger.info(f"Chunking 5min data into {len(write_chunks)} daily files.")

        else:
            write_chunks = [_ds]

        for _chunk_ds in write_chunks:

            # recalculate time_bnds
            has_time_bnds = 'time_bnds' in _chunk_ds.data_vars or 'time_bnds' in _chunk_ds.coords
            if has_time_bnds:
                # generate_time_bounds creates the actual bounds array
                _chunk_ds['time_bnds'] = generate_time_bounds(_chunk_ds, output_frequency=output_frequency)

                # Strip units/calendar from attrs to avoid the encoding conflict
                if 'units' in _chunk_ds['time'].attrs:
                    del _chunk_ds['time'].attrs['units']

                # Ensure time_bnds is in the encoding for the netCDF write
                encoding['time_bnds'] = config.encoding['time_bnds']

            # Derive the start/end date strings from the actual timeseries and override
            if config.derive_filename_times_from_data or output_frequency == '5min':
                # Determine the format based on the output_frequency
                if output_frequency == '1D':
                    date_format = '%Y%m%d'
                elif output_frequency == '1M':
                    date_format = '%Y%m'
                elif output_frequency == 'fx':
                    date_format = None
                else:
                    date_format = '%Y%m%d%H%M'

                if date_format is not None:
                    str_times = _chunk_ds.time.dt.strftime(date_format).data
                    if len(str_times) > 0:
                        context['start_date'] = str_times[0]
                        context['end_date'] = str_times[-1]
                    logger.debug(
                        'start_date = %(start_date)s, end_date = %(end_date)s' % context)

            drs_path = adu.get_template(config, 'drs_path') % context
            filename_template = adu.get_template(config, 'filename')

            # Override for fixed variables
            if adu.is_time_invariant(_chunk_ds):
                logger.debug('Overriding output filename template with fixed alternative.')
                filename_template = adu.get_template(config, 'filename_fixed')

            # Assemble the output filepath
            output_filename = filename_template % context
            output_filepath = os.path.join(
                output_directory, drs_path, output_filename)
            logger.debug(f'output_filepath = {output_filepath}')

            # Skip if already there and overwrite is not set, otherwise continue
            if os.path.isfile(output_filepath) and overwrite == False:
                logger.debug(
                    f'{output_filepath} exists and overwrite is set to False, skipping.')
                continue

            # Check for uninterpolated keys in the output path, which should fail at this point.
            uninterpolated_keys = adu.get_uninterpolated_placeholders(
                output_filepath)

            if len(uninterpolated_keys) > 0:
                logger.error('Uninterpolated keys remain in the output filepath.')
                logger.error(f'output_filepath = {output_filepath}')
                raise DRSContextInterpolationException(uninterpolated_keys)

            # Create the output directory
            output_dir = os.path.dirname(output_filepath)
            logger.debug(f'Creating {output_dir}')
            os.makedirs(output_dir, exist_ok=True)

            # Supervise this job to ensure that it does in fact complete.
            with Supervisor(seconds=config.processing_timeout_seconds, error_msg=f'Variable {variable} took too long to complete, moving on.'):
                logger.info('Waiting for computations to finish.')
                progress(_chunk_ds)

            logger.debug(f'Writing {output_filepath}')

            # Round time/time_bnds to avoid floating point issues (can remove if time units is changed to "minutes since")
            if not adu.is_time_invariant(_chunk_ds):
                _chunk_ds['time'] = _chunk_ds['time'].dt.round('1s')
                if 'time_bnds' in _chunk_ds:
                    _chunk_ds['time_bnds'] = _chunk_ds['time_bnds'].dt.round('1s')

            write_kwargs = {
                "path": output_filepath,
                "format": output_format,
                "encoding": encoding,
            }

            if not adu.is_time_invariant(_chunk_ds):
                if 'time' in _chunk_ds.dims:
                    write_kwargs['unlimited_dims'] = ['time']

            write = _chunk_ds.to_netcdf(**write_kwargs)

    elapsed_time = timer.stop()
    logger.info(f'DRS processing task took {elapsed_time} seconds.')
    

def load_variable_config(project_config):
    """Extract the variable configuration out of the project configuration.

    Args:
        project_config (dict-like): Project configuration.

    Returns:
        dict: Variable dictionary with name: [levels] (single level will have a list containing None.)
    """

    # Extract the different rank variables
    v2ds = project_config['variables_2d']
    v3ds = project_config['variables_3d']

    # Create a dictionary of variables to process keyed to an empty list of levels for 2D
    variables = {v2d: [None] for v2d in v2ds}

    # Add in the 3D variables, with levels this time
    for v3d, levels in v3ds.items():
        variables[v3d] = levels

    return variables


def process_multi(variables, domain_id, project, **kwargs):
    """Start a processing chain of multiple variables.

    Args:
        variables (list): List of variables to process.
        domain_id (str): Domain to process from domains.json.
        project (str): Project metadata to use from projects.json.
        **kwargs: Additional keyword arguments to pass to the processing chain.
    """

    logger = au.get_logger(__name__)

    # Load the project metadata
    project_config = load_config('projects')[project]
    config = load_config('drs_shep')

    # Load all variables if nothing was supplied
    if not variables:

        # Select a default schema
        schema_key = config['default_schema']

        # Override if requested, this might be a direct filepath
        if 'schema_key' in kwargs.keys():
            schema_key = kwargs['schema']

        # Load it
        logger.info(f'No variables supplied, loading from schema as defined in configuration ({schema_key}).')
        schema = axs.load_schema(schema_key)
        variables = list(schema['variables'].keys())


    else:
        logger.debug('User has supplied the following variables')
        logger.debug(variables)

    num_variables = len(variables)
    logger.info(f'{num_variables} variable(s) to process.')

    # Start the cluster if requested
    if config.dask['enable']:
        logger.info('Starting dask client.')

        cluster_config = config.dask['cluster']

        # Add PBS_JOBFS if set
        if 'PBS_JOBFS' in os.environ.keys():
            cluster_config['local_directory'] = os.getenv('PBS_JOBFS')

        cluster = LocalCluster(**cluster_config)
        client = Client(cluster)
        logger.info(client)

    output_frequencies = au.pluralise(kwargs['output_frequency'])

    # Yes this is a nested loop, but a single variable/domain/output_freq combination could still be 10K+ files, which WILL be processed in parallel.
    for variable in variables:
        for output_frequency in output_frequencies:

            attempt = 1
            no_files = False
            success = False

            rerun_attempts = config.get('rerun_attempts', default=1)

            while attempt <= rerun_attempts and no_files == False and success == False:

                logger.info(f'Processing {variable} {output_frequency}')
                instance_kwargs = kwargs.copy()
                instance_kwargs['variable'] = variable
                instance_kwargs['domain_id'] = domain_id
                instance_kwargs['project'] = project
                instance_kwargs['output_frequency'] = output_frequency

                if config.dask['enable']:
                    logger.info('Waiting for dask workers')
                    client.wait_for_workers(1, timeout=config['dask']['restart_timeout_seconds'])
                    logger.info(client)
                    logger.info(f'Dashboard located at {client.dashboard_link}')

                try:

                    process(**instance_kwargs)
                    success = True

                # Not technically an error, filtering has discounted all available files.
                except NoFilesToProcessException as ex:

                    logger.info(f'No files to process for {variable}')
                    no_files = True

                # Something wrong with the inputs regarding time.
                except IndexError as ex:

                    # Log the exception
                    log_exception(f'Timeseries inconsistency, check input data.', ex)

                    # Check recoverability
                    if is_error_recoverable(ex) and attempt <= rerun_attempts:
                        logger.info('Error is recoverable, incrementing attempts.')
                        attempt += 1
                        continue

                    # Track the failure and max out the attempts to execute the finally clause
                    track_failure(variable, ex)
                    attempt = rerun_attempts + 1

                # Unknown exception
                except Exception as ex:

                    log_exception(
                        f'Variable {variable} failed for output_frequency {output_frequency}. Error to follow',
                        ex
                    )

                    # Check recoverability
                    if is_error_recoverable(ex) and attempt <= rerun_attempts:
                        logger.info(
                            'Error is recoverable, incrementing attempts.')
                        attempt += 1
                        continue

                    # Track the failure and max out the attempts to execute the finally clause
                    track_failure(variable, ex)
                    attempt = rerun_attempts + 1

                # Run regardless of success/failure
                finally:

                    if config.dask['enable'] and config.dask['restart_client_between_variables'] and no_files == False:
                        logger.info('User has requested dask client restarts between each variable (for resilience), restarting now.')
                        client.restart()
                        logger.info(client)

    logger.info('DRS processing complete, please see consumed/lock/log files for further detail.')


def filter_years(filepaths, year, offset=0):
    """Filter filepaths based on a year, plus or minus an offset.

    Args:
        filepaths (list): List of filepaths.
        year (int): Year.
        offset (int, optional): Number of years either side of YEAR to include. Defaults to 0.

    Returns:
        list : List of filtered filepaths.
    """
    _filepaths = list()
    years = range(year-offset, year+offset+1)
    for filepath in filepaths:
        for year in years:
###            if str(year) in os.path.basename(filepath):
            if f".{year}" in os.path.basename(filepath):
                _filepaths.append(filepath)

    return _filepaths


def update_cell_methods(ds, variable, output_frequency):
    """
    Update the cell_methods attribute using the CORDEX CSV table.
    Fixes conflicts where variables have time_bnds but claim to be 'point'.
    """
    logger = au.get_logger(__name__)
    da = ds[variable]
    
    # Map internal frequency codes to CSV-compatible strings
    FREQ_MAP = {
        "5min": "5min",
        "1H": "1hr",
        "3H": "3hr",
        "6H": "6hr",
        "1D": "day",
        "1M": "mon",
        "FX": "fx"
    }
    
    target_freq = FREQ_MAP.get(output_frequency, output_frequency.lower())
    
    # 1. Primary Lookup: Specific Frequency
    cell_methods = get_official_cell_method(variable, target_freq)
    
    # 2. Secondary Lookup: Fallback to 1hr if primary fails (common for 6hr variables)
    if not cell_methods and target_freq != "1hr":
        logger.info(f"Entry for {variable} at {target_freq} not found in CSV. Trying 1hr fallback...")
        cell_methods = get_official_cell_method(variable, "1hr")

    # 3. Final safety: Default if no entry exists
    if not cell_methods:
        logger.warning(f"No CSV entry found for {variable}. Using default 'area: mean'.")
        cell_methods = "area: mean"

    # Standardize the retrieved methods
    new_methods = ' '.join(cell_methods.split())
    current_methods = str(da.attrs.get('cell_methods', '')).lower()
    has_time_bnds = 'time_bnds' in ds.variables or 'time_bnds' in ds.coords

    # Check for the specific conflict: File has bounds but metadata says 'point'
    # AND the CORDEX table confirms it should actually be 'mean' (or max/min)
    is_supposed_to_be_agg = any(m in new_methods.lower() for m in ["mean", "maximum", "minimum"])
    
    if "time: point" in current_methods and has_time_bnds and is_supposed_to_be_agg:
        logger.info(f"Fixing metadata conflict for {variable}: 'time: point' -> '{new_methods}' (time_bnds detected)")
        da.attrs['cell_methods'] = new_methods
    else:
        # Standard update for all other cases
        da.attrs['cell_methods'] = new_methods

    ds[variable] = da
    
    return ds


def log_exception(message, ex):
    """Log the exception.

    Args:
        message (str): Human-readable error message.
        ex (Exception): Stack trace.
    """
    logger = au.get_logger(__name__)
    logger.error(message)
    logger.exception(ex)


def is_error_recoverable(exception):
    """Check if the error is recoverable.

    Args:
        exception (Exception): Raised exception to check.

    Returns:
        bool : True if recoverable, False otherwise.
    """
    config = load_config('drs_shep')
    return adu.is_error_recoverable(exception, config.get('recoverable_errors', list()))


def track_failure(variable, exception):
    """Track the failure for reprocessing.

    Args:
        variable (str): Variable name.
        exception (Exception): Exception raised.
    """

    config = load_config('drs_shep')

    if config.track_failures and 'AXIOM_LOG_DIR' in os.environ.keys() and 'PBS_JOBNAME' in os.environ.keys():

        failed_filepath = os.path.join(
            os.getenv('AXIOM_LOG_DIR'),
            os.getenv('PBS_JOBNAME') + '.failed'
        )

        exname = type(exception).__name__

        with open(failed_filepath, 'a') as failed:  
            failed.write(f'{variable},{exname}\n')
