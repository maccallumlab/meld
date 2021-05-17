#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from unittest import mock  #type: ignore
from unittest.mock import sentinel  #type: ignore
from meld.remd import leader, ladder, adaptor
from meld.system import runner
from meld import comm, vault
from numpy.testing import assert_almost_equal  #type: ignore


class TestSingleStep(unittest.TestCase):
    def setUp(self):
        self.N_REPS = 6
        self.MAX_STEPS = 1
        self.mock_ladder = mock.Mock(spec_set=ladder.NearestNeighborLadder)
        self.PERM_VECTOR = list(reversed(range(self.N_REPS)))
        self.mock_ladder.compute_exchanges.return_value = self.PERM_VECTOR
        self.mock_adaptor = mock.Mock(adaptor.EqualAcceptanceAdaptor)
        self.runner = leader.LeaderReplicaExchangeRunner(
            self.N_REPS, self.MAX_STEPS, self.mock_ladder, self.mock_adaptor
        )
        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.n_replicas = 6
        self.mock_comm.broadcast_states_to_workers.return_value = sentinel.MY_STATE_INIT

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
        self.mock_state_5 = mock.Mock()
        self.mock_state_5.positions = sentinel.pos5
        self.mock_state_5.velocities = 1.0
        self.mock_state_6 = mock.Mock()
        self.mock_state_6.positions = sentinel.pos6
        self.mock_state_6.velocities = 1.0
        self.fake_states_after_run = [
            self.mock_state_1,
            self.mock_state_2,
            self.mock_state_3,
            self.mock_state_4,
            self.mock_state_5,
            self.mock_state_6,
        ]

        self.mock_comm.gather_states_from_workers.return_value = (
            self.fake_states_after_run
        )
        self.mock_energy_matrix = mock.MagicMock()
        self.mock_comm.gather_energies_from_workers.return_value = (
            self.mock_energy_matrix
        )
        self.mock_comm.exchange_states_for_energy_calc.return_value = (
            self.fake_states_after_run
        )

        self.mock_system_runner = mock.Mock(spec=runner.ReplicaRunner)
        self.mock_system_runner.minimize_then_run.return_value = sentinel.MY_STATE
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
        self.mock_system_runner.temperature_scaler = mock.MagicMock()
        self.mock_system_runner.temperature_scaler.return_value = 1.0

        self.mock_store = mock.Mock(spec_set=vault.DataStore)
        self.mock_store.n_replicas = 6
        self.mock_store.load_states.return_value = sentinel.ALL_STATES

    def test_raises_on_comm_n_replicas_mismatch(self):
        "should raise AssertionError if n_replicas on comm does not match"
        self.mock_comm.n_replicas = 42  # 42 != 6

        with self.assertRaises(AssertionError):
            self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

    def test_raises_on_store_n_replicas_mismatch(self):
        "should raise AssertionError if n_replicas on store does not match"
        self.mock_store.n_replicas = 42  # 42 != 6

        with self.assertRaises(AssertionError):
            self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

    def test_alpha_begins_uniform(self):
        "alphas should be initialized with uniform spacing"
        assert_almost_equal(self.runner.alphas, [0., 0.2, 0.4, 0.6, 0.8, 1.])

    def test_step_begins_one(self):
        "step should begin at one"
        self.assertEqual(self.runner.step, 1)

    def test_run_should_load_previous_step(self):
        "calling run should load the states from step 0"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_store.load_states.assert_called_once_with(stage=0)

    def test_should_set_alphas_on_system_runner(self):
        "should set alphas on the system runner"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        # the leader is always alphas = 0.
        self.mock_system_runner.prepare_for_timestep.assert_called_once_with(0., 1)

    def test_should_broadcast_alphas(self):
        "calling run should broadcast all of the alpha values"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.assertEqual(self.mock_comm.broadcast_alphas_to_workers.call_count, 1)

    def test_should_broadcast_states(self):
        "calling run should broadcast states"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_comm.broadcast_states_to_workers.assert_called_once_with(
            sentinel.ALL_STATES
        )

    def test_should_call_minimize_then_run(self):
        "should call minimize_then_run on the system_runner"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_system_runner.minimize_then_run.assert_called_once_with(
            sentinel.MY_STATE_INIT
        )

    def test_should_exchange_states_for_energy_calc(self):
        "should gather all states"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_comm.exchange_states_for_energy_calc.assert_called_once_with(
            sentinel.MY_STATE
        )

    def test_calls_get_energy_on_each_state(self):
        "should call get_energy on each state"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        calls = [mock.call(s) for s in self.fake_states_after_run]
        self.mock_system_runner.get_energy.assert_has_calls(calls)

    def test_calls_gather_energies_from_workers(self):
        "should call gather_energies_from_workers"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_comm.gather_energies_from_workers.assert_called_once_with(
            self.FAKE_ENERGIES_AFTER_GET_ENERGY
        )

    def test_calls_ladder(self):
        "should call ladder"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_ladder.compute_exchanges.assert_called_once_with(
            self.mock_energy_matrix, self.mock_adaptor
        )

    def test_states_are_saved_in_permuted_form(self):
        "states should be saved to store in properly permuted order"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        # our permutation matrix is reversed
        correct_order = list(
            reversed(
                [
                    sentinel.pos1,
                    sentinel.pos2,
                    sentinel.pos3,
                    sentinel.pos4,
                    sentinel.pos5,
                    sentinel.pos6,
                ]
            )
        )
        observed_order = [s.positions for s in self.fake_states_after_run]
        self.assertEqual(observed_order, correct_order)

        self.mock_store.save_states.assert_called_once_with(
            self.fake_states_after_run, 1
        )

    def test_should_save_remd_runner(self):
        "should save ourselves to disk"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_store.save_remd_runner.assert_called_once_with(self.runner)

    def test_should_write_traj(self):
        "should write trajectory to disk"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_store.append_traj.assert_called_once_with(self.mock_state_1, 1)

    def test_should_save_alphas(self):
        "should write alphas to disk"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_store.save_alphas.assert_called_once_with(mock.ANY, 1)

    def test_should_save_permutation_matrix(self):
        "should write permutation matrix to disk"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_store.save_permutation_vector.assert_called_once_with(
            self.PERM_VECTOR, 1
        )

    def test_should_call_backup(self):
        "should ask the store to handle backup for us"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.mock_store.backup.assert_called_once_with(1)


class TestFiveSteps(unittest.TestCase):
    def setUp(self):
        self.N_REPS = 6
        self.MAX_STEPS = 5
        self.mock_ladder = mock.Mock(spec_set=ladder.NearestNeighborLadder)
        self.mock_ladder.compute_exchanges.return_value = list(
            reversed(range(self.N_REPS))
        )
        self.mock_adaptor = mock.Mock(adaptor.EqualAcceptanceAdaptor)
        self.runner = leader.LeaderReplicaExchangeRunner(
            self.N_REPS, self.MAX_STEPS, self.mock_ladder, self.mock_adaptor
        )
        self.mock_comm = mock.Mock(spec_set=comm.MPICommunicator)
        self.mock_comm.n_replicas = 6
        self.mock_comm.broadcast_states_to_workers.return_value = sentinel.MY_STATE_INIT

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
        self.mock_state_5 = mock.Mock()
        self.mock_state_5.positions = sentinel.pos5
        self.mock_state_5.velocities = 1.0
        self.mock_state_6 = mock.Mock()
        self.mock_state_6.positions = sentinel.pos6
        self.mock_state_6.velocities = 1.0
        self.fake_states_after_run = [
            self.mock_state_1,
            self.mock_state_2,
            self.mock_state_3,
            self.mock_state_4,
            self.mock_state_5,
            self.mock_state_6,
        ]
        self.mock_comm.gather_states_from_workers.return_value = (
            self.fake_states_after_run
        )
        self.mock_energy_matrix = mock.MagicMock()
        self.mock_comm.gather_energies_from_workers.return_value = (
            self.mock_energy_matrix
        )
        self.mock_comm.exchange_states_for_energy_calc.return_value = (
            self.fake_states_after_run
        )

        self.mock_system_runner = mock.Mock(spec=runner.ReplicaRunner)
        self.mock_system_runner.minimize_then_run.return_value = sentinel.MY_STATE
        self.mock_system_runner.run.return_value = sentinel.MY_STATE
        self.FAKE_ENERGIES_AFTER_GET_ENERGY = [
            sentinel.E1,
            sentinel.E2,
            sentinel.E3,
            sentinel.E4,
            sentinel.E5,
            sentinel.E6,
        ]
        self.mock_system_runner.get_energy.side_effect = (
            self.FAKE_ENERGIES_AFTER_GET_ENERGY * self.MAX_STEPS
        )
        self.mock_system_runner.temperature_scaler = mock.MagicMock()
        self.mock_system_runner.temperature_scaler.return_value = 1.0

        self.mock_store = mock.Mock(spec_set=vault.DataStore)
        self.mock_store.n_replicas = 6
        self.mock_store.load_states.return_value = sentinel.ALL_STATES

    def test_save_states_is_called_each_iteration(self):
        "save_states should be called once per iteration"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.assertEqual(self.mock_store.save_states.call_count, self.MAX_STEPS)

    def test_load_states_is_called_once(self):
        "load_states should only be called once"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.assertEqual(self.mock_store.load_states.call_count, 1)

    def test_set_alphas_is_called_once(self):
        "set_alphas should only be called once"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.assertEqual(self.mock_system_runner.prepare_for_timestep.call_count, 5)

    def test_minimize_then_run_is_called_once(self):
        "minimize_then_run should only be called once"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.assertEqual(self.mock_system_runner.minimize_then_run.call_count, 1)

    def test_run_is_called_four_times(self):
        "run should only be called four times"
        self.runner.run(self.mock_comm, self.mock_system_runner, self.mock_store)

        self.assertEqual(self.mock_system_runner.run.call_count, 4)
