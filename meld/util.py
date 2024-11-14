#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Miscellaneous utilities
"""

import contextlib
import logging
import logging.handlers
import os
import shutil
import tempfile
import time
from functools import wraps

from openmm import unit as u  # type: ignore


def strip_unit(value, unit: u.Unit) -> float:
    assert isinstance(
        value, u.Quantity
    ), f"Expected a Quantity, but {value} has type {type(value)}"
    return value.value_in_unit(unit)


@contextlib.contextmanager
def in_temp_dir():
    """
    Context manager to run in temporary directory

    This is used to run certain operations, e.g. tleap, in a temporary
    directory. It is also used during the unit tests.
    """
    try:
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        yield

    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)


def log_timing(dest_logger):
    """
    A function decorator to record timing

    Args:
        dest_logger: the logger to record timings to
    """

    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            t1 = time.time()
            res = func(*args, **kwds)
            t2 = time.time()
            dest_logger.debug("%s took %0.3f ms" % (func.__name__, (t2 - t1) * 1000.0))
            return res

        return wrapper

    return wrap


class HostNameContextFilter(logging.Filter):
    """
    Filter class that adds hostid information to logging records.
    """

    def __init__(self, hostid: str):
        """
        Initialize a HostNameContextFilter

        Args:
            hostid: the host id to add to logging output
        """
        logging.Filter.__init__(self)
        self.hostid = hostid

    def filter(self, record):
        record.hostid = self.hostid
        return True
