'''
Module to provide convenience functions for dealing with netCDF4 files.

NetCDF files are not thread-safe and can become corrupted if more than one thread
or process tries to access them at the same time. To prevent this, we write to the
the file "results.progress" and periodically copy the results to "results.nc". This
module provides tools for opening the correct files depending on if we are reading,
writing, or appending, and to handle the periodic copying.

'''

import netCDF4 as cdf
import shutil


# global variable to handle the netCDF Dataset object
_dataset = None

PROGRESS_FILE = 'results.progress'
NC_FILE = 'results.nc'


def open_dataset_for_write():
    '''Open netCDF file for writing and return a dataset object'''
    global _dataset
    _check_dataset()
    _dataset = cdf.Dataset(PROGRESS_FILE, mode='w', clobber=False)
    return _dataset


def open_dataset_for_append():
    '''Open netCDF file for appending and return a dataset object'''
    global _dataset
    _check_dataset()
    _dataset = cdf.Dataset(PROGRESS_FILE, mode='a')
    return _dataset


def open_dataset_for_read():
    '''Open netCDF file for reading and return a dataset object'''
    global _dataset
    _check_dataset()
    _dataset = cdf.Dataset(NC_FILE, mode='r')
    return _dataset


def transfer_dataset_progress():
    '''Copy progress file onto nc file'''
    global _dataset
    _dataset.sync()
    shutil.copy(PROGRESS_FILE, NC_FILE)


def _drop():
    # This method is for unit testing only. It will mak this module forget that
    # it has been previously initialized.
    # DO NOT CALL IN PRODUCTION CODE!
    global _dataset
    _dataset = None


def _check_dataset():
    global _dataset
    if not _dataset is None:
        raise RuntimeError('Tried to open netCDF file after dataset already opened')
