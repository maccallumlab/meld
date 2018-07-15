#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from unittest import mock  #type: ignore
import numpy as np  #type: ignore
from meld.remd import ladder, adaptor


class TestLadderInputs(unittest.TestCase):
    "Ladder.compute_exchanges should accept the right kinds of inputs"

    def setUp(self):
        self.mock_adaptor = mock.Mock(spec_set=adaptor.EqualAcceptanceAdaptor)
        self.ladder = ladder.NearestNeighborLadder(n_trials=1)

    def test_accepts(self):
        "should accept a square energy array"
        N = 10
        energies = np.zeros((N, N))

        result = self.ladder.compute_exchanges(energies, self.mock_adaptor)
        self.assertTrue(result)

    def test_reject_3d(self):
        "should only accept 2d energy array"
        N = 10
        energies = np.zeros((N, N, N))

        with self.assertRaises(AssertionError):
            self.ladder.compute_exchanges(energies, self.mock_adaptor)

    def test_rejects_non_square(self):
        "should reject non-square energy arrays"
        N = 10
        M = 20
        energies = np.zeros((N, M))

        with self.assertRaises(AssertionError):
            self.ladder.compute_exchanges(energies, self.mock_adaptor)


class TestSingleTrialWithTwoReplicas(unittest.TestCase):
    "test compute_exchanges with a single trial on a two replica system"

    def setUp(self):
        self.ladder = ladder.NearestNeighborLadder(n_trials=1)
        self.mock_adaptor = mock.Mock(spec_set=adaptor.EqualAcceptanceAdaptor)

    def test_favorable(self):
        "should always accept a favorable swap"
        energy = np.array([[0, -1], [0, 0]])

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(result, [1, 0], "swap should have been accepted")

    def test_very_unfavorable(self):
        "should never accept very unfavorable swap"
        energy = np.array([[0, 100000], [0, 0]])

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(result, [0, 1], "swap should have been rejected")

    def test_marginal_unfavorable_below(self):
        "we should accept when a random variable is below the metropolis weight"
        with mock.patch("meld.remd.ladder.random.random") as mock_random:
            energy = np.array([[0, 1.0], [0, 0]])
            # random number smaller than exp(-1)
            mock_random.return_value = 0.3

            result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

            self.assertEqual(result, [1, 0], "swap should have been accepted")

    def test_marginal_unfavorable_above(self):
        "we should reject when a random variable is above the metropolis weight"
        with mock.patch("meld.remd.ladder.random.random") as mock_random:
            energy = np.array([[0, 1.0], [0, 0]])
            # random number larger than exp(-1)
            mock_random.return_value = 0.5

            result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

            self.assertEqual(result, [0, 1], "swap should have been rejected")

    def test_adaptor_update_called_accept(self):
        "we should call the adaptor when the exchange is accepted"
        energy = np.array([[0, -1], [0, 0]])

        self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.mock_adaptor.update.assert_called_once_with(0, True)

    def test_adaptor_update_called_rejected(self):
        "we should call the adaptor when the exchange is failed"
        energy = np.array([[0, 100000], [0, 0]])

        self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.mock_adaptor.update.assert_called_once_with(0, False)


class TestTwoTrialsWithTwoReplicas(unittest.TestCase):
    "test compute_exchanges with two trials on a two replica system"

    def setUp(self):
        self.ladder = ladder.NearestNeighborLadder(n_trials=2)
        self.mock_adaptor = mock.Mock(spec_set=adaptor.EqualAcceptanceAdaptor)

    def test_adatpor_called(self):
        "adaptor should be called twice"
        energy = np.array([[0, 0], [0, 0]])

        self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(
            self.mock_adaptor.update.call_count, 2, "adaptor should be updated twice"
        )
        call_1 = mock.call(0, True)
        call_2 = mock.call(0, True)
        self.mock_adaptor.update.assert_has_calls([call_1, call_2])

    def test_swap_second_try(self):
        "if the swap fails, then is accepted, we will have permuted the items"
        with mock.patch("meld.remd.ladder.random.random") as mock_random:
            energy = np.array([[0, 1], [0, 0]])
            # first fails, second accepts
            mock_random.side_effect = [0.5, 0.3]

            result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

            self.assertEqual(result, [1, 0], "we should have failed and then swapped")
            call_1 = mock.call(0, False)
            call_2 = mock.call(0, True)
            self.mock_adaptor.update.assert_has_calls([call_1, call_2])

    def test_swap_back(self):
        "if the swap succeeds, then we should swap back without calling random"
        with mock.patch("meld.remd.ladder.random.random") as mock_random:
            energy = np.array([[0, 1], [0, 0]])
            # random number below exp(-1)
            mock_random.return_value = 0.3

            result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

            self.assertEqual(
                mock_random.call_count, 1, "we only need one random number"
            )
            self.assertEqual(
                result, [0, 1], "we should have swapped and then swapped back"
            )

    def test_swap_back_calls_adaptor(self):
        "if the swap succeeds, then we should swap back and call the adaptor appropriately"
        with mock.patch("meld.remd.ladder.random.random") as mock_random:
            energy = np.array([[0, 1], [0, 0]])
            # random number below exp(-1)
            mock_random.return_value = 0.3

            self.ladder.compute_exchanges(energy, self.mock_adaptor)

            call_1 = mock.call(0, True)
            call_2 = mock.call(0, True)
            self.mock_adaptor.update.assert_has_calls([call_1, call_2])


class TestTwoTrialsWithThreeReplicas(unittest.TestCase):
    "test compute_exchanges with two trials on three replicas"

    def setUp(self):
        self.ladder = ladder.NearestNeighborLadder(n_trials=2)
        self.mock_adaptor = mock.Mock(spec_set=adaptor.EqualAcceptanceAdaptor)

    def test_low_energy_move_down(self):
        "if the highest replica has a low energy, it should move to the bottom"
        with mock.patch("meld.remd.ladder.random.choice") as mock_random_choice:
            # the lowest energy is when replica 3 is at the bottom
            # the other two replicas don't care what their indices are
            energy = np.array([[0, 0, 0], [0, 0, 0], [-2, -1, 0]])
            # first swap 1 with 2, then 0 with 1
            # this should allow replica 3 to move down
            mock_random_choice.side_effect = [1, 0]

            result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

            # order should be 2, 0, 1
            self.assertEqual(result, [2, 0, 1])

    def test_low_energy_move_down_calls_adaptor(self):
        "adaptor should be called correctly when replica 2 moves down"
        with mock.patch("meld.remd.ladder.random.choice") as mock_random_choice:
            # the lowest energy is when replica 3 is at the bottom
            # the other two replicas don't care what their indices are
            energy = np.array([[0, 0, 0], [0, 0, 0], [-2, -1, 0]])
            # first swap 1 with 2, then 0 with 1
            # this should allow replica 3 to move down
            mock_random_choice.side_effect = [1, 0]

            self.ladder.compute_exchanges(energy, self.mock_adaptor)

            call_1 = mock.call(1, True)
            call_2 = mock.call(0, True)
            self.mock_adaptor.update.assert_has_calls([call_1, call_2])


class TestFiveHundredTrialsWithThreeReplicas(unittest.TestCase):
    "test compute exchanges with 500 trials on three replicas"

    def test_low_energy_move_down(self):
        # the preferred order will be 2, 0, 1
        energy = np.array([[0, 0, -10000], [0, 0, 0], [0, 0, 0]])
        mock_adaptor = mock.Mock(spec_set=adaptor.EqualAcceptanceAdaptor)
        l = ladder.NearestNeighborLadder(n_trials=500)

        result = l.compute_exchanges(energy, mock_adaptor)

        self.assertEqual(result[0], 2, "replica 2 should be at the bottom")
