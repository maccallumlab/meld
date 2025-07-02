import numpy as np  # type: ignore
import unittest
from meld.system import param_sampling


class TestDiscreteSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = param_sampling.DiscreteSampler(0, 9, 5)

    def test_should_be_valid_within_bounds(self):
        in_bound1 = self.sampler.is_valid(0)
        self.assertEqual(in_bound1, True)

        in_bound2 = self.sampler.is_valid(9)
        self.assertEqual(in_bound2, True)

    def test_should_be_invalid_outside_bounds(self):
        in_bound1 = self.sampler.is_valid(-1)
        self.assertEqual(in_bound1, False)

        in_bound2 = self.sampler.is_valid(10)
        self.assertEqual(in_bound2, False)

    def test_should_produce_uniform_distribution(self):
        counts = np.zeros(10)
        v = 0
        for i in range(100_000):
            new_v = self.sampler.sample(v)
            if self.sampler.is_valid(new_v):
                v = new_v
            counts[v] += 1
        self.assertTrue(np.all(counts > 9000))
        self.assertTrue(np.all(counts < 11_000))


class TestContinuousSampler(unittest.TestCase):
    def setUp(self):
        self.sampler = param_sampling.ContinuousSampler(0, 9, 5)

    def test_should_be_valid_within_bounds(self):
        in_bound1 = self.sampler.is_valid(0)
        self.assertEqual(in_bound1, True)

        in_bound2 = self.sampler.is_valid(9)
        self.assertEqual(in_bound2, True)

    def test_should_be_invalid_outside_bounds(self):
        in_bound1 = self.sampler.is_valid(-1)
        self.assertEqual(in_bound1, False)

        in_bound2 = self.sampler.is_valid(10)
        self.assertEqual(in_bound2, False)

    def test_should_produce_uniform_distribution(self):
        values = []
        v = 0.0
        for i in range(100_000):
            new_v = self.sampler.sample(v)
            if self.sampler.is_valid(new_v):
                v = new_v
            values.append(v)

        hist, _ = np.histogram(values, bins=10, range=(0, 9))
        self.assertTrue(np.all(hist > 9000))
        self.assertTrue(np.all(hist < 11_000))
