#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
File with various classes and functions to make testing easier.
"""

import os
import tempfile
import shutil


class TempDirHelper:
    """
    Class to add convenience methods for running tests in temp dirs.

    Inherit from this class and call setUpTempDir and tearDownTempDir from setUp and tearDown, respectively.

    """

    def setUpTempDir(self):
        # create and change to temp dir
        self.cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp()
        os.chdir(self.tmpdir)

    def tearDownTempDir(self):
        # switch to original dir and clean up
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)


class FakeSystem:
    """
    Fake system class to test REMD.
    """

    pass


class FakeSystemRunner:
    """
    Fake system runner class. Doens't actually run anything. For testing REMD.
    """

    def minimize_then_run(self, state):
        return state

    def run(self, state):
        return state

    def get_energy(self, state):
        return 0.

    def set_alpha(self, alpha):
        pass
