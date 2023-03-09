#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import runner
from meld.system import options

import unittest
from unittest import mock  #type: ignore


class TestGetRunner(unittest.TestCase):
    def test_runner_equals_openmm_should_create_openmm_runner(self):
        meld_system = mock.Mock()
        comm = mock.Mock()
        opt = options.RunOptions(runner="openmm")
        with mock.patch("meld.runner.openmm_runner.OpenMMRunner") as mock_runner:
            runner.get_runner(meld_system, opt, comm, platform="Reference")
            self.assertEqual(mock_runner.call_count, 1)

    def test_runner_equals_fake_runner_should_create_fake_runner(self):
        meld_system = mock.Mock()
        comm = mock.Mock()
        opt = options.RunOptions(runner="fake_runner")
        with mock.patch("meld.runner.fake_runner.FakeSystemRunner") as mock_runner:
            runner.get_runner(meld_system, opt, comm, platform="Reference")
            self.assertEqual(mock_runner.call_count, 1)
