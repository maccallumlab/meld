import unittest
import math
import mock
import numpy
from meld import ladder

class TestLadder(unittest.TestCase):
    def setUp(self):
        self.mock_adaptor = mock.MagicMock()
        self.ladder = ladder.Ladder(n_iterations=1)

    def test_accepts(self):
        "should accept a square energy array"
        N = 10
        energies = numpy.zeros( (N,N) )

        self.ladder.compute_exchanges(energies, self.mock_adaptor)

    def test_reject_3d(self):
        "should only accept 2d energy array"
        N = 10
        energies = numpy.zeros( (N,N,N) )

        with self.assertRaises(AssertionError):
            self.ladder.compute_exchanges(energies, self.mock_adaptor)

    def test_rejects_non_square(self):
        "should reject non-square energy arrays"
        N = 10
        M = 20
        energies = numpy.zeros( (N,M) )

        with self.assertRaises(AssertionError):
            self.ladder.compute_exchanges(energies, self.mock_adaptor)

class TestSingleSwapTwoReps(unittest.TestCase):
    def setUp(self):
        self.ladder = ladder.Ladder(n_iterations=1)
        self.mock_adaptor = mock.MagicMock()

    def test_favorable(self):
        "should always accept a favorable swap"
        energy = numpy.array( [[0, -1], [0, 0]] )

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(result[0], 1, 'entries should be permuted by swap')
        self.assertEqual(result[1], 0, 'entries should be permuted by swap')

    def test_very_unfavorable(self):
        "should never accept very unfavorable swap"
        energy = numpy.array( [[0, 100000], [0, 0]] )

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(result[0], 0, 'entries should not be permuted by swap')
        self.assertEqual(result[1], 1, 'entries should not be permuted by swap')

    @mock.patch('meld.ladder.random.random')
    def test_marginal_unfavorable_below(self, mock_random):
        "we should accept when a random variable is below the metropolis weight"
        energy = numpy.array( [[0, 1.0], [0, 0]] )
        # random number smaller than exp(-1)
        mock_random.return_value = 0.3

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(result[0], 1, 'entries should be permuted by swap')
        self.assertEqual(result[1], 0, 'entries should be permuted by swap')

    @mock.patch('meld.ladder.random.random')
    def test_marginal_unfavorable_above(self, mock_random):
        "we should reject when a random variable is above the metropolis weight"
        energy = numpy.array( [[0, 1.0], [0, 0]] )
        # random number smaller than exp(-1)
        mock_random.return_value = 0.5

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(result[0], 0, 'entries should not be permuted by swap')
        self.assertEqual(result[1], 1, 'entries should not be permuted by swap')

    def test_adaptor_update_called_accept(self):
        "we should call the adaptor when the exchange is accepted"
        energy = numpy.array( [[0, -1], [0, 0]] )

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.mock_adaptor.update.assert_called_once_with(0, 1, True)

    def test_adaptor_update_called_rejected(self):
        "we should call the adaptor when the exchange is failed"
        energy = numpy.array( [[0, 100000], [0, 0]] )

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.mock_adaptor.update.assert_called_once_with(0, 1, False)

class TestTwoSwapTwoRep(unittest.TestCase):
    def setUp(self):
        self.ladder = ladder.Ladder(n_iterations=2)
        self.mock_adaptor = mock.MagicMock()

    def test_adatpor_called(self):
        "adaptor should be called twice"
        energy = numpy.array( [[0, 0], [0, 0]] )

        self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(self.mock_adaptor.update.call_count, 2, 'adaptor should be updated twice')

    @mock.patch('meld.ladder.random.random')
    def test_swap_second_try(self, mock_random):
        "if the swap fails, then is accepted, we will have permuted the items"
        energy = numpy.array( [[0, 1], [0, 0]] )
        # first fails, second accepts
        mock_random.side_effect = [0.5, 0.3]

        result = self.ladder.compute_exchanges(energy, self.mock_adaptor)

        self.assertEqual(result[0], 1, 'we should have swapped')
        self.assertEqual(result[1], 0, 'we should have swapped')

