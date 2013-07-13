import unittest
import mock
from meld.remd import slave_runner, master_runner
from meld.system import runner
from meld import comm


sentinel = mock.sentinel


class TestSlaveInitFromMaster(unittest.TestCase):
    "Initialize slave runner from master runner"

    def setUp(self):
        self.mock_master = mock.Mock(spec_set=master_runner.MasterReplicaExchangeRunner)
        self.mock_master.step = 42
        self.mock_master.max_steps = 43

    def test_sets_step(self):
        "creating slave from_master should set the step"
        slave = slave_runner.SlaveReplicaExchangeRunner.from_master(self.mock_master)

        self.assertEqual(slave.step, 42)

    def test_sets_max_steps(self):
        "creating slave from_master should set max_steps"
        slave = slave_runner.SlaveReplicaExchangeRunner.from_master(self.mock_master)

        self.assertEqual(slave.max_steps, 43)


class TestSlaveSingle(unittest.TestCase):
    "Make sure the SlaveReplicaExchangeRunner works for a single round"

    def setUp(self):
        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.receive_alpha_from_master.return_value = sentinel.ALPHA
        self.mock_comm.receive_state_from_master.return_value = mock.sentinel.STATE
        self.mock_comm.receive_states_for_energy_calc_from_master.return_value = [
            sentinel.STATE_1,
            sentinel.STATE_2,
            sentinel.STATE_3,
            sentinel.STATE_4]

        self.mock_system_runner = mock.Mock(spec_set=runner.ReplicaRunner)
        self.mock_system_runner.minimize_then_run.return_value = mock.sentinel.STATE
        self.mock_system_runner.get_energy.side_effect = [
            sentinel.ENERGY_1,
            sentinel.ENERGY_2,
            sentinel.ENERGY_3,
            sentinel.ENERGY_4]

        self.runner = slave_runner.SlaveReplicaExchangeRunner(step=1, max_steps=1)

    def test_calls_receive_alpha(self):
        "should call receive_alpha"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.receive_alpha_from_master.assert_called_once_with()

    def test_sets_alpha_on_system_runner(self):
        "should set alpha on the replica runner"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_system_runner.set_alpha.assert_called_once_with(sentinel.ALPHA, 1.0)

    def test_calls_receive_state(self):
        "should receive state from master"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.receive_state_from_master.assert_called_once_with()

    def test_runs_replica_with_state(self):
        "should call minimize_then_run on the system_runner with the received state"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_system_runner.minimize_then_run.assert_called_once_with(mock.sentinel.STATE)

    def test_calls_send_state(self):
        "should send the state from the system_runner to the master"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.send_state_to_master.assert_called_once_with(mock.sentinel.STATE)

    def test_calls_revieve_all_states(self):
        "should call receive_states_for_energy_calc from the master"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.receive_states_for_energy_calc_from_master.assert_called_once_with()

    def test_calls_get_energy_for_each_state(self):
        "should call get_energy on each state received"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        calls = [mock.call(sentinel.STATE_1), mock.call(sentinel.STATE_2),
                 mock.call(sentinel.STATE_3), mock.call(sentinel.STATE_4)]
        self.mock_system_runner.get_energy.assert_has_calls(calls)

    def test_sends_energies_back_to_master(self):
        "should send energies back to the master"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.send_energies_to_master.assert_called_once_with(
            [sentinel.ENERGY_1, sentinel.ENERGY_2, sentinel.ENERGY_3, sentinel.ENERGY_4])


class TestSlaveMultiple(unittest.TestCase):
    "Make sure the SlaveReplicaExchangeRunner works for a multiple rounds"

    def setUp(self):
        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.receive_states_for_energy_calc_from_master.return_value = [
            sentinel.STATE_1,
            sentinel.STATE_2,
            sentinel.STATE_3,
            sentinel.STATE_4]
        self.mock_system_runner = mock.Mock(spec_set=runner.ReplicaRunner)
        self.runner = slave_runner.SlaveReplicaExchangeRunner(step=1, max_steps=4)

    def test_runs_correct_number_of_steps(self):
        "should run the correct number of steps"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.assertEqual(self.mock_comm.receive_state_from_master.call_count, 4)

    def test_minimize_then_run_called_once(self):
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.assertEqual(self.mock_system_runner.minimize_then_run.call_count, 1)

    def test_run_called_three_times(self):
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.assertEqual(self.mock_system_runner.run.call_count, 3)
