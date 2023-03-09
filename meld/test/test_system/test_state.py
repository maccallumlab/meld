#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
import numpy as np  # type: ignore
from meld.system import state
from meld.vault import ENERGY_GROUPS


class TestInitState(unittest.TestCase):
    def setUp(self):
        self.N_ATOMS = 500
        self.N_SPRINGS = 600

        self.coords = np.zeros((self.N_ATOMS, 3))
        self.vels = np.zeros((self.N_ATOMS, 3))
        self.box_vectors = np.zeros(3)
        self.lam = 0.5
        self.energy = 42.0
        self.group_energies = np.zeros(ENERGY_GROUPS)

    def test_should_raise_with_coords_not_2d(self):
        "should raise RuntimeError if coords is not 2d"
        bad_pos = np.zeros((75, 83, 52))
        with self.assertRaises(RuntimeError):
            state.SystemState(
                bad_pos,
                self.vels,
                self.lam,
                self.energy,
                self.group_energies,
                self.box_vectors,
            )

    def test_should_raise_with_coords_second_dim_not_3(self):
        "should raise RuntimeError if coords is not (n, 3)"
        bad_pos = np.zeros((75, 4))
        with self.assertRaises(RuntimeError):
            state.SystemState(
                bad_pos,
                self.vels,
                self.lam,
                self.energy,
                self.group_energies,
                self.box_vectors,
            )

    def test_should_raise_if_vels_not_match_coords(self):
        "should raise runtime error if vels is not the same shape as coords"
        bad_vels = np.zeros((42, 3))
        with self.assertRaises(RuntimeError):
            state.SystemState(
                self.coords,
                bad_vels,
                self.lam,
                self.energy,
                self.group_energies,
                self.box_vectors,
            )

    def test_lambda_must_be_between_zero_and_one(self):
        "should raise RuntimeError if lambda is outside of [0,1]"
        bad_lam = -2
        with self.assertRaises(RuntimeError):
            state.SystemState(
                self.coords,
                self.vels,
                bad_lam,
                self.energy,
                self.group_energies,
                self.box_vectors,
            )
