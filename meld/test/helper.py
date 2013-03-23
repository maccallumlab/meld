'''
File with various classes and functions to make testing easier.
'''

import contextlib
import os
import tempfile
import shutil


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


class TempDirHelper(object):
    '''
    Class to add convenience methods for running tests in temp dirs.

    Inherit from this class and call setUpTempDir and tearDownTempDir from setUp and tearDown, respectively.

    '''
    def setUpTempDir(self):
        # create and change to temp dir
        self.cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp()
        os.chdir(self.tmpdir)

    def tearDownTempDir(self):
        # switch to original dir and clean up
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)


class FakeSystem(object):
    '''
    Fake system class to test REMD.
    '''
    def get_runner(self):
        return FakeSystemRunner()


class FakeSystemRunner(object):
    '''
    Fake system runner class. Doens't actually run anything. For testing REMD.
    '''
    def initialize(self):
        pass

    def minimize_then_run(self, state):
        return state

    def run(self, state):
        return state

    def get_energy(self, state):
        return 0.

    def set_lambda(self, lambda_):
        pass
