#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld.system import montecarlo as mc
from meld.system.indexing import AtomIndex
from meld.system.state import SystemState
import meld
import unittest
from unittest import mock  # type: ignore
import numpy as np

from meld.vault import ENERGY_GROUPS  # type: ignore


class TestMonteCarloSchedulerSingle(unittest.TestCase):
    def test_should_call_single_mover_correct_number_of_times(self):
        n_trials = 10
        mock_mover = mock.Mock()
        mock_mover.trial.return_value = (mock.Mock, True)
        mock_state = mock.Mock()
        mock_runner = mock.Mock()

        scheduler = mc.MonteCarloScheduler([(mock_mover, 1.0)], n_trials)
        scheduler.update(mock_state, mock_runner)

        self.assertEqual(mock_mover.trial.call_count, n_trials)


class TestMonteCarloSchedulerTwo(unittest.TestCase):
    def setUp(self):
        self.mock_structure = mock.Mock()
        self.mock_mover1 = mock.Mock()
        self.mock_mover1.trial.return_value = (self.mock_structure, True)
        self.mock_mover2 = mock.Mock()
        self.mock_mover2.trial.return_value = (self.mock_structure, True)
        self.mock_state = mock.Mock()
        self.mock_runner = mock.Mock()

    def test_should_call_pair_of_movers_correct_number_of_times(self):
        n_trials = 100000
        weight1 = 1
        weight2 = 2

        scheduler = mc.MonteCarloScheduler(
            [(self.mock_mover1, weight1), (self.mock_mover2, weight2)], n_trials
        )
        scheduler.update(self.mock_state, self.mock_runner)

        frac1 = self.mock_mover1.trial.call_count / float(n_trials)
        expected1 = weight1 / float(weight1 + weight2)
        frac2 = self.mock_mover2.trial.call_count / float(n_trials)
        expected2 = weight2 / float(weight1 + weight2)
        self.assertAlmostEqual(frac1, expected1, 2)
        self.assertAlmostEqual(frac2, expected2, 2)

    def test_should_track_total_count_correctly(self):
        n_trials = 1000
        weight1 = 1
        weight2 = 1

        scheduler = mc.MonteCarloScheduler(
            [(self.mock_mover1, weight1), (self.mock_mover2, weight2)], n_trials
        )
        scheduler.update(self.mock_state, self.mock_runner)

        self.assertEqual(scheduler.trial_counts[0], self.mock_mover1.trial.call_count)
        self.assertEqual(scheduler.trial_counts[1], self.mock_mover2.trial.call_count)

    def test_should_track_accepted_count_correctly(self):
        n_trials = 1000
        weight1 = 1
        weight2 = 1

        scheduler = mc.MonteCarloScheduler(
            [(self.mock_mover1, weight1), (self.mock_mover2, weight2)], n_trials
        )
        scheduler.update(self.mock_state, self.mock_runner)

        # our mock always returns True, so accepted should == attempted
        self.assertEqual(
            scheduler.accepted_counts[0], self.mock_mover1.trial.call_count
        )
        self.assertEqual(
            scheduler.accepted_counts[1], self.mock_mover2.trial.call_count
        )

    def test_should_return_correct_state(self):
        n_trials = 1000
        weight1 = 1
        weight2 = 1

        scheduler = mc.MonteCarloScheduler(
            [(self.mock_mover1, weight1), (self.mock_mover2, weight2)], n_trials
        )
        result = scheduler.update(self.mock_state, self.mock_runner)

        # our mock always returns True, so accepted should == attempted
        self.assertIs(result, self.mock_structure)


class TestRotateAroundAxis(unittest.TestCase):
    def test_should_produce_correct_rotation(self):
        start = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [0.0, 0.0, -2.0],
                [1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0],
            ]
        )
        p1 = np.array([0.0, 0.0, -1.0])
        p2 = np.array([0.0, 2.0, -1.0])
        rotation_angle = 90.0

        # rotation should leave the first two points unchanged,
        # while the rest are circularly permuted due to 90 degree
        # rotation
        end = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [-1.0, 0.0, -1.0],
                [0.0, 0.0, -2.0],
                [1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        )

        result = mc._rotate_around_vector(p1, p2, rotation_angle, start)

        np.testing.assert_array_almost_equal(result, end)


class TestMetropolis(unittest.TestCase):
    def test_should_always_accept_favorable_move(self):
        current_energy = 0.0
        trial_energy = -1.0
        bias = 0.0

        result = mc._metropolis(current_energy, trial_energy, bias)

        self.assertEqual(result, True)

    def test_should_accept_unfavorable_move_with_random_less_than_metropolis_weight(
        self,
    ):
        current_energy = 0.0
        trial_energy = 1.0
        bias = 0.0

        with mock.patch("meld.system.montecarlo.random.random") as mock_random:
            mock_random.return_value = 0.35  # slightly less than exp(-1)
            result = mc._metropolis(current_energy, trial_energy, bias)

        self.assertEqual(result, True)

    def test_should_not_accept_unfavorable_move_with_random_greater_than_metropolis_weight(
        self,
    ):
        current_energy = 0.0
        trial_energy = 1.0
        bias = 0.0

        with mock.patch("meld.system.montecarlo.random.random") as mock_random:
            mock_random.return_value = 0.37  # slightly more than exp(-1)
            result = mc._metropolis(current_energy, trial_energy, bias)

        self.assertEqual(result, False)


class TestRandomTorsionMover(unittest.TestCase):
    def test_should_accept_favorable_move(self):
        # setup
        start = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [0.0, 0.0, -2.0],
                [1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0],
            ]
        )

        end = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [0.0, 0.0, -2.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        )

        state = SystemState(
            start, np.zeros_like(start), 0.0, 0.0, np.zeros(ENERGY_GROUPS), np.zeros(3)
        )
        mock_runner = mock.Mock()
        mock_runner.get_energy.return_value = -1.0

        mover = mc.RandomTorsionMover(
            AtomIndex(0), AtomIndex(1), [AtomIndex(4), AtomIndex(5)]
        )

        # exercise
        with mock.patch(
            "meld.system.montecarlo._generate_uniform_angle"
        ) as mock_gen_angle:
            mock_gen_angle.return_value = 90.0
            new_state, accepted = mover.trial(state, mock_runner)

        # assert
        self.assertEqual(new_state.energy, -1.0)
        self.assertEqual(accepted, True)
        np.testing.assert_array_almost_equal(new_state.positions, end)

    def test_should_reject_impossible_move(self):
        # setup
        start = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [0.0, 0.0, -2.0],
                [1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0],
            ]
        )

        end = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [0.0, 0.0, -2.0],
                [1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [-1.0, 0.0, -1.0],
            ]
        )

        state = SystemState(
            start, np.zeros_like(start), 0.0, 0.0, np.zeros(ENERGY_GROUPS), np.zeros(3)
        )
        mock_runner = mock.Mock()
        mock_runner.get_energy.return_value = 1000.0

        mover = mc.RandomTorsionMover(
            AtomIndex(0), AtomIndex(1), [AtomIndex(4), AtomIndex(5)]
        )

        # exercise
        with mock.patch(
            "meld.system.montecarlo._generate_uniform_angle"
        ) as mock_gen_angle:
            mock_gen_angle.return_value = 90.0
            new_state, accepted = mover.trial(state, mock_runner)

        # assert
        self.assertEqual(new_state.energy, 0.0)
        self.assertEqual(accepted, False)
        np.testing.assert_array_almost_equal(new_state.positions, end)
