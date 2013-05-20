import os
import shutil
import tempfile
import contextlib
import time
from functools import wraps


@contextlib.contextmanager
def in_temp_dir():
    """Context manager to run in temporary directory"""
    try:
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        yield

    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)


def log_timing(dest_logger):
    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwds):
            t1 = time.time()
            res = func(*args, **kwds)
            t2 = time.time()
            dest_logger.debug('%s took %0.3f ms' % (func.func_name, (t2-t1)*1000.0))
            return res
        return wrapper
    return wrap

