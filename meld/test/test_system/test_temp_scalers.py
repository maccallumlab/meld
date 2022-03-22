#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from meld.system.temperature import (
    ConstantTemperatureScaler,
    LinearTemperatureScaler,
    GeometricTemperatureScaler,
)
from openmm import unit as u  # type: ignore


class TestConstantTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.s = ConstantTemperatureScaler(300.0 * u.kelvin)

    def test_returns_constant_when_alpha_is_zero(self):
        t = self.s(0.0)
        self.assertAlmostEqual(t, 300.0)

    def test_returns_constant_when_alpha_is_one(self):
        t = self.s(1.0)
        self.assertAlmostEqual(t, 300.0)

    def test_raises_when_alpha_below_zero(self):
        with self.assertRaises(RuntimeError):
            self.s(-1)

    def test_raises_when_alpha_above_one(self):
        with self.assertRaises(RuntimeError):
            self.s(2)


class TestLinearTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.s = LinearTemperatureScaler(0.2, 0.8, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_returns_min_when_alpha_is_low(self):
        t = self.s(0)
        self.assertAlmostEqual(t, 300.0)

    def test_returns_max_when_alpha_is_high(self):
        t = self.s(1)
        self.assertAlmostEqual(t, 500.0)

    def test_returns_mid_when_alpha_is_half(self):
        t = self.s(0.5)
        self.assertAlmostEqual(t, 400.0)

    def test_raises_when_alpha_below_zero(self):
        with self.assertRaises(RuntimeError):
            self.s(-1)

    def test_raises_when_alpha_above_one(self):
        with self.assertRaises(RuntimeError):
            self.s(2)

    def test_raises_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(-0.1, 0.8, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(1.1, 0.8, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(0.0, -0.1, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(0.0, 1.1, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_alpha_min_above_alpha_max(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(1.0, 0.0, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_temp_min_is_below_zero(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(0.0, 1.0, -300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_temp_max_is_below_zero(self):
        with self.assertRaises(RuntimeError):
            LinearTemperatureScaler(0.0, 1.0, 300.0 * u.kelvin, -500.0 * u.kelvin)


class TestGeometricTemperatureScaler(unittest.TestCase):
    def setUp(self):
        self.s = GeometricTemperatureScaler(
            0.2, 0.8, 300.0 * u.kelvin, 500.0 * u.kelvin
        )

    def test_returns_min_when_alpha_is_low(self):
        t = self.s(0)
        self.assertAlmostEqual(t, 300.0)

    def test_returns_max_when_alpha_is_high(self):
        t = self.s(1)
        self.assertAlmostEqual(t, 500.0)

    def test_returns_mid_when_alpha_is_half(self):
        t = self.s(0.5)
        self.assertAlmostEqual(t, 387.298334621)

    def test_raises_when_alpha_below_zero(self):
        with self.assertRaises(RuntimeError):
            self.s(-1)

    def test_raises_when_alpha_above_one(self):
        with self.assertRaises(RuntimeError):
            self.s(2)

    def test_raises_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(-0.1, 0.8, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(1.1, 0.8, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(0.0, -0.1, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(0.0, 1.1, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_alpha_min_above_alpha_max(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(1.0, 0.0, 300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_temp_min_is_below_zero(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(0.0, 1.0, -300.0 * u.kelvin, 500.0 * u.kelvin)

    def test_raises_when_temp_max_is_below_zero(self):
        with self.assertRaises(RuntimeError):
            GeometricTemperatureScaler(0.0, 1.0, 300.0 * u.kelvin, -500.0 * u.kelvin)
