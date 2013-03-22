import unittest
import numpy as np
from meld.system import state


class TestInitState(unittest.TestCase):
    def setUp(self):
        self.N_ATOMS = 500
        self.N_SPRINGS = 600

        self.coords = np.zeros((self.N_ATOMS, 3))
        self.vels = np.zeros((self.N_ATOMS, 3))
        self.spring_states = np.zeros(self.N_SPRINGS)
        self.lam = 0.5
        self.energy = 42.
        self.spring_energies = np.zeros(self.N_SPRINGS)

    def test_should_raise_with_coords_not_2d(self):
        "should raise RuntimeError if coords is not 2d"
        bad_pos = np.zeros((75, 83, 52))
        with self.assertRaises(RuntimeError):
            state.SystemState(bad_pos, self.vels, self.spring_states, self.lam, self.energy,
                              self.spring_energies)

    def test_should_raise_with_coords_second_dim_not_3(self):
        "should raise RuntimeError if coords is not (n, 3)"
        bad_pos = np.zeros((75, 4))
        with self.assertRaises(RuntimeError):
            state.SystemState(bad_pos, self.vels, self.spring_states, self.lam, self.energy,
                              self.spring_energies)

    def test_should_raise_if_vels_not_match_coords(self):
        "should raise runtime error if vels is not the same shape as coords"
        bad_vels = np.zeros((42, 3))
        with self.assertRaises(RuntimeError):
            state.SystemState(self.coords, bad_vels, self.spring_states, self.lam, self.energy,
                              self.spring_energies)

    def test_lambda_must_be_between_zero_and_one(self):
        "should raise RuntimeError if lambda is outside of [0,1]"
        bad_lam = -2
        with self.assertRaises(RuntimeError):
            state.SystemState(self.coords, self.vels, self.spring_states, bad_lam, self.energy,
                              self.spring_energies)

    def test_spring_states_must_be_1d(self):
        bad_states = np.zeros((42, 42))
        with self.assertRaises(RuntimeError):
            state.SystemState(self.coords, self.vels, bad_states, self.lam, self.energy,
                              self.spring_energies)

    def test_spring_energies_must_match_spring_states(self):
        "should raise RuntimeError if spring_energies does not match spring_states"
        bad_spring_energies = np.zeros(42)
        with self.assertRaises(RuntimeError):
            state.SystemState(self.coords, self.vels, self.spring_states, self.lam, self.energy,
                              bad_spring_energies)

    def test_should_work_with_no_springs(self):
        "should work if all springs are None"
        state.SystemState(self.coords, self.vels, None, self.lam, self.energy, None)

    def test_should_raise_if_spring_states_is_none_but_spring_energies_is_not(self):
        "should raise RuntimeError if spring_states is none but spring energies is not"
        with self.assertRaises(RuntimeError):
            state.SystemState(self.coords, self.vels, None, self.lam, self.energy,
                              self.spring_energies)
