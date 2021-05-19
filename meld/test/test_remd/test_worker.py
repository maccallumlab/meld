#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from unittest import mock  # type: ignore
from meld.remd import worker, leader
from meld import interfaces
from meld import comm


sentinel = mock.sentinel


class TestWorkerSingle(unittest.TestCase):
    "Make sure the WorkerReplicaExchangeRunner works for a single round"

    def setUp(self):
        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.receive_alpha_from_leader.return_value = sentinel.ALPHA
        self.mock_comm.receive_state_from_leader.return_value = mock.sentinel.STATE
        self.mock_comm.receive_states_for_energy_calc_from_leader.return_value = [
            sentinel.STATE_1,
            sentinel.STATE_2,
            sentinel.STATE_3,
            sentinel.STATE_4,
        ]
        self.mock_state_1 = mock.Mock()
        self.mock_state_1.positions = sentinel.pos1
        self.mock_state_1.velocities = 1.0
        self.mock_state_2 = mock.Mock()
        self.mock_state_2.velocities = 1.0
        self.mock_state_2.positions = sentinel.pos2
        self.mock_state_3 = mock.Mock()
        self.mock_state_3.velocities = 1.0
        self.mock_state_3.positions = sentinel.pos3
        self.mock_state_4 = mock.Mock()
        self.mock_state_4.velocities = 1.0
        self.mock_state_4.positions = sentinel.pos4
        self.fake_states_after_run = [
            self.mock_state_1,
            self.mock_state_2,
            self.mock_state_3,
            self.mock_state_4,
        ]
        self.mock_comm.exchange_states_for_energy_calc.return_value = (
            self.fake_states_after_run
        )

        self.mock_system_runner = mock.Mock(spec_set=interfaces.IRunner)
        self.mock_system_runner.minimize_then_run.return_value = mock.sentinel.STATE
        self.FAKE_ENERGIES_AFTER_GET_ENERGY = [
            sentinel.E1,
            sentinel.E2,
            sentinel.E3,
            sentinel.E4,
            sentinel.E5,
            sentinel.E6,
        ]
        self.mock_system_runner.get_energy.side_effect = (
            self.FAKE_ENERGIES_AFTER_GET_ENERGY
        )

        self.runner = worker.WorkerReplicaExchangeRunner(step=1, max_steps=1)

    def test_calls_receive_alpha(self):
        "should call receive_alpha"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.receive_alpha_from_leader.assert_called_once_with()

    def test_sets_alpha_on_system_runner(self):
        "should set alpha on the replica runner"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_system_runner.prepare_for_timestep.assert_called_once_with(
            sentinel.STATE, sentinel.ALPHA, 1
        )

    def test_calls_receive_state(self):
        "should receive state from leader"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.receive_state_from_leader.assert_called_once_with()

    def test_runs_replica_with_state(self):
        "should call minimize_then_run on the system_runner with the received state"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_system_runner.minimize_then_run.assert_called_once_with(
            mock.sentinel.STATE
        )

    def test_calls_exchange_states_for_energy_calc(self):
        "should call receive_states_for_energy_calc from the leader"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.exchange_states_for_energy_calc.assert_called_once_with(
            mock.sentinel.STATE
        )

    def test_calls_get_energy_for_each_state(self):
        "should call get_energy on each state received"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        calls = [mock.call(s) for s in self.fake_states_after_run]
        self.mock_system_runner.get_energy.assert_has_calls(calls)

    def test_sends_energies_back_to_leader(self):
        "should send energies back to the leader"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.mock_comm.send_energies_to_leader.assert_called_once_with(
            [sentinel.E1, sentinel.E2, sentinel.E3, sentinel.E4]
        )


class TestWorkerMultiple(unittest.TestCase):
    "Make sure the worker works for a multiple rounds"

    def setUp(self):
        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.receive_states_for_energy_calc_from_leader.return_value = [
            sentinel.STATE_1,
            sentinel.STATE_2,
            sentinel.STATE_3,
            sentinel.STATE_4,
        ]
        self.mock_state_1 = mock.Mock()
        self.mock_state_1.positions = sentinel.pos1
        self.mock_state_1.velocities = 1.0
        self.mock_state_2 = mock.Mock()
        self.mock_state_2.velocities = 1.0
        self.mock_state_2.positions = sentinel.pos2
        self.mock_state_3 = mock.Mock()
        self.mock_state_3.velocities = 1.0
        self.mock_state_3.positions = sentinel.pos3
        self.mock_state_4 = mock.Mock()
        self.mock_state_4.velocities = 1.0
        self.mock_state_4.positions = sentinel.pos4
        self.fake_states_after_run = [
            self.mock_state_1,
            self.mock_state_2,
            self.mock_state_3,
            self.mock_state_4,
        ]
        self.mock_comm.exchange_states_for_energy_calc.return_value = (
            self.fake_states_after_run
        )

        self.mock_system_runner = mock.Mock(spec_set=interfaces.IRunner)
        self.mock_system_runner.minimize_then_run.return_value = mock.sentinel.STATE
        self.FAKE_ENERGIES_AFTER_GET_ENERGY = [
            sentinel.E1,
            sentinel.E2,
            sentinel.E3,
            sentinel.E4,
        ] * 4
        self.mock_system_runner.get_energy.side_effect = (
            self.FAKE_ENERGIES_AFTER_GET_ENERGY
        )
        self.runner = worker.WorkerReplicaExchangeRunner(step=1, max_steps=4)

    def test_runs_correct_number_of_steps(self):
        "should run the correct number of steps"
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.assertEqual(self.mock_comm.receive_state_from_leader.call_count, 4)

    def test_minimize_then_run_called_once(self):
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.assertEqual(self.mock_system_runner.minimize_then_run.call_count, 1)

    def test_run_called_three_times(self):
        self.runner.run(self.mock_comm, self.mock_system_runner)

        self.assertEqual(self.mock_system_runner.run.call_count, 3)
