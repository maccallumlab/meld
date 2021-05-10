#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from unittest import mock  #type: ignore
from meld.remd import follower, leader, launch
from meld.system.openmm_runner import OpenMMRunner
from meld import comm, vault
import logging


class TestLaunchNotLeader(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch("meld.remd.launch.vault")
        self.mock_vault = self.patcher.start()

        self.log_patcher = mock.patch("meld.remd.launch.logging")
        self.log_patcher.start()

        self.get_runner_patcher = mock.patch("meld.remd.launch.get_runner")
        self.mock_get_runner = self.get_runner_patcher.start()
        self.mock_runner = mock.Mock(spec=OpenMMRunner)
        self.mock_get_runner.return_value = self.mock_runner

        self.MockDataStore = mock.Mock(spec_set=vault.DataStore)
        self.mock_vault.DataStore = self.MockDataStore
        self.mock_store = mock.Mock(spec_set=vault.DataStore)
        self.MockDataStore.load_data_store.return_value = self.mock_store

        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.is_leader.return_value = False
        self.mock_comm.receive_logger_address_from_leader.return_value = (
            "127.0.0.1",
            32768,
        )
        self.mock_comm.rank = 0
        self.mock_store.load_communicator.return_value = self.mock_comm

        self.mock_system = mock.Mock()
        self.mock_store.load_system.return_value = self.mock_system

        self.mock_remd_leader = mock.Mock(
            spec_set=leader.LeaderReplicaExchangeRunner
        )
        self.mock_remd_follower = mock.Mock(
            spec_set=follower.FollowerReplicaExchangeRunner
        )
        self.mock_remd_leader.to_follower.return_value = self.mock_remd_follower
        self.mock_store.load_remd_runner.return_value = self.mock_remd_leader

        self.mock_store.load_run_options.return_value.runner = "openmm"

        self.log_handler = logging.StreamHandler()

    def cleanUp(self):
        self.patcher.stop()
        self.get_runner_patcher.stop()
        self.log_patcher.stop()

    def test_load_datastore(self):
        "should call vault.DataStore.load_data_store to load the data_store"
        launch.launch("Reference", self.log_handler)

        self.MockDataStore.load_data_store.assert_called_once_with()

    def test_should_init_comm(self):
        "should initialize the communicator"
        launch.launch("Reference", self.log_handler)

        self.mock_comm.initialize.assert_called_once_with()

    def test_should_call_to_follower(self):
        "should call to_follower on remd_runner"
        launch.launch("Reference", self.log_handler)

        self.mock_remd_leader.to_follower.assert_called_once_with()

    def test_should_run(self):
        "should run remd runner with correct parameters"
        launch.launch("Reference", self.log_handler)

        self.mock_remd_follower.run.assert_called_once_with(
            self.mock_comm, self.mock_runner
        )

    def test_should_not_init_store(self):
        "should not init store"
        launch.launch("Reference", self.log_handler)

        self.assertEqual(self.mock_store.initialize.call_count, 0)


class TestLaunchLeader(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch("meld.remd.launch.vault")
        self.mock_vault = self.patcher.start()

        self.get_runner_patcher = mock.patch("meld.remd.launch.get_runner")
        self.mock_get_runner = self.get_runner_patcher.start()
        self.mock_runner = mock.Mock(spec=OpenMMRunner)
        self.mock_get_runner.return_value = self.mock_runner

        self.MockDataStore = mock.Mock(spec_set=vault.DataStore)
        self.mock_vault.DataStore = self.MockDataStore
        self.mock_store = mock.Mock(spec_set=vault.DataStore)
        self.MockDataStore.load_data_store.return_value = self.mock_store

        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.is_leader.return_value = True
        self.mock_comm.rank = 0
        self.mock_store.load_communicator.return_value = self.mock_comm

        self.mock_system = mock.Mock()
        self.mock_store.load_system.return_value = self.mock_system

        self.mock_remd_leader = mock.Mock(
            spec_set=leader.LeaderReplicaExchangeRunner
        )
        self.mock_remd_follower = mock.Mock(
            spec_set=follower.FollowerReplicaExchangeRunner
        )
        self.mock_remd_leader.to_follower.return_value = self.mock_remd_follower
        self.mock_store.load_remd_runner.return_value = self.mock_remd_leader

        self.mock_store.load_run_options.return_value.runner = "openmm"

        self.log_handler = logging.StreamHandler()

    def tearDown(self):
        self.patcher.stop()
        self.get_runner_patcher.stop()

    def test_load_datastore(self):
        "should call load the datastore"
        launch.launch("Reference", self.log_handler)

        self.MockDataStore.load_data_store.assert_called_once_with()

    def test_should_init_comm(self):
        "should initialize the communicator"
        launch.launch("Reference", self.log_handler)

        self.mock_comm.initialize.assert_called_once_with()

    def test_should_init_store(self):
        "should initialize the store"
        launch.launch("Reference", self.log_handler)

        self.mock_store.initialize.assert_called_once_with(mode="a")

    def test_should_run(self):
        "should run remd runner with correct parameters"
        launch.launch("Reference", self.log_handler)

        self.mock_remd_leader.run.assert_called_once_with(
            self.mock_comm, self.mock_runner, self.mock_store
        )
