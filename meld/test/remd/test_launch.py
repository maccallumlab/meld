import unittest
import mock
from meld.remd import slave_runner, master_runner, launch
from meld.system import OpenMMRunner
from meld import comm, vault


class TestLaunchNotMaster(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch('meld.remd.launch.vault')
        self.mock_vault = self.patcher.start()

        self.get_runner_patcher = mock.patch('meld.remd.launch.get_runner')
        self.mock_get_runner = self.get_runner_patcher.start()
        self.mock_runner = mock.Mock(spec=OpenMMRunner)
        self.mock_get_runner.return_value = self.mock_runner

        self.MockDataStore = mock.Mock(spec_set=vault.DataStore)
        self.mock_vault.DataStore = self.MockDataStore
        self.mock_store = mock.Mock(spec_set=vault.DataStore)
        self.MockDataStore.load_data_store.return_value = self.mock_store

        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.is_master.return_value = False
        self.mock_store.load_communicator.return_value = self.mock_comm

        self.mock_system = mock.Mock()
        self.mock_store.load_system.return_value = self.mock_system

        self.mock_remd_master = mock.Mock(spec_set=master_runner.MasterReplicaExchangeRunner)
        self.mock_remd_slave = mock.Mock(spec_set=slave_runner.SlaveReplicaExchangeRunner)
        self.mock_remd_master.to_slave.return_value = self.mock_remd_slave
        self.mock_store.load_remd_runner.return_value = self.mock_remd_master

        self.mock_store.load_run_options.return_value.runner = 'openmm'

    def cleanUp(self):
        self.patcher.stop()
        self.get_runner_patcher.stop()

    def test_load_datastore(self):
        "should call vault.DataStore.load_data_store to load the data_store"
        launch.launch()

        self.MockDataStore.load_data_store.assert_called_once_with()

    def test_should_init_comm(self):
        "should initialize the communicator"
        launch.launch()

        self.mock_comm.initialize.assert_called_once_with()

    def test_should_call_to_slave(self):
        "should call to_slave on remd_runner"
        launch.launch()

        self.mock_remd_master.to_slave.assert_called_once_with()

    def test_should_run(self):
        "should run remd runner with correct parameters"
        launch.launch()

        self.mock_remd_slave.run.assert_called_once_with(self.mock_comm, self.mock_runner)

    def test_should_not_init_store(self):
        "should not init store"
        launch.launch()

        self.assertEqual(self.mock_store.initialize.call_count, 0)


class TestLaunchMaster(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch('meld.remd.launch.vault')
        self.mock_vault = self.patcher.start()

        self.get_runner_patcher = mock.patch('meld.remd.launch.get_runner')
        self.mock_get_runner = self.get_runner_patcher.start()
        self.mock_runner = mock.Mock(spec=OpenMMRunner)
        self.mock_get_runner.return_value = self.mock_runner

        self.MockDataStore = mock.Mock(spec_set=vault.DataStore)
        self.mock_vault.DataStore = self.MockDataStore
        self.mock_store = mock.Mock(spec_set=vault.DataStore)
        self.MockDataStore.load_data_store.return_value = self.mock_store

        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.is_master.return_value = True
        self.mock_store.load_communicator.return_value = self.mock_comm

        self.mock_system = mock.Mock()
        self.mock_store.load_system.return_value = self.mock_system

        self.mock_remd_master = mock.Mock(spec_set=master_runner.MasterReplicaExchangeRunner)
        self.mock_remd_slave = mock.Mock(spec_set=slave_runner.SlaveReplicaExchangeRunner)
        self.mock_remd_master.to_slave.return_value = self.mock_remd_slave
        self.mock_store.load_remd_runner.return_value = self.mock_remd_master

        self.mock_store.load_run_options.return_value.runner = 'openmm'

    def tearDown(self):
        self.patcher.stop()
        self.get_runner_patcher.stop()

    def test_load_datastore(self):
        "should call load the datastore"
        launch.launch()

        self.MockDataStore.load_data_store.assert_called_once_with()

    def test_should_init_comm(self):
        "should initialize the communicator"
        launch.launch()

        self.mock_comm.initialize.assert_called_once_with()

    def test_should_init_store(self):
        "should initialize the store"
        launch.launch()

        self.mock_store.initialize.assert_called_once_with(mode='existing')

    def test_should_run(self):
        "should run remd runner with correct parameters"
        launch.launch()

        self.mock_remd_master.run.assert_called_once_with(self.mock_comm, self.mock_runner, self.mock_store)
