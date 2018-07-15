#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from unittest import mock  #type: ignore
from meld.system import get_runner, RunOptions


class TestGetRunner(unittest.TestCase):
    def test_runner_equals_openmm_should_create_openmm_runner(self):
        system = mock.Mock()
        comm = mock.Mock()
        options = RunOptions()
        options.runner = "openmm"
        with mock.patch("meld.system.runner.OpenMMRunner") as mock_runner:
            get_runner(system, options, comm)
            self.assertEqual(mock_runner.call_count, 1)

    def test_runner_equals_fake_runner_should_create_fake_runner(self):
        system = mock.Mock()
        comm = mock.Mock()
        options = RunOptions()
        options.runner = "fake_runner"
        with mock.patch("meld.system.runner.FakeSystemRunner") as mock_runner:
            get_runner(system, options, comm)
            self.assertEqual(mock_runner.call_count, 1)
