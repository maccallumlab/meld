import os
import shutil
import tempfile
import contextlib


@contextlib.contextmanager
def in_temp_dir():
    '''Context manager to run in temporary directory'''
    try:
        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        yield

    finally:
        os.chdir(cwd)
        shutil.rmtree(tmpdir)
