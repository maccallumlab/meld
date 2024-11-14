#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
A module to implement basic Monte Carlo moves.

These are primarily useful during initialization to
remove bad geometry.
"""

import math
import random
from typing import List, Tuple

import numpy as np  # type: ignore

from meld import interfaces
from meld.system import indexing


class Mover:
    pass


class MonteCarloScheduler:
    """
    Weighted random selection of Monte Carlo moves
    """

    def __init__(
        self, movers_with_weights: List[Tuple[Mover, float]], update_trials: int
    ):
        """
        Initialize a MonteCarloscheduler

        Args:
            movers_with_weights: a list of tuples of moves and weights
            update_trials: number of trials to perform

        Note:
           Weights do not need to be normalized.
        """
        self.update_trials = update_trials
        self._movers_with_weights = movers_with_weights

        self.trial_counts = np.zeros(len(self._movers_with_weights))
        self.accepted_counts = np.zeros(len(self._movers_with_weights))

    def update(
        self,
        starting_state: interfaces.IState,
        runner: interfaces.IRunner,
    ) -> interfaces.IState:
        """
        Perform a series of Monte Carlo moves

        Args:
            starting_state: initial state of system
            runner: runner to evaluate energies

        Returns:
            the system state after Monte Carlo trials
        """
        current_state = starting_state

        for trial in range(self.update_trials):
            # choose a mover
            mover, index = self._choose_mover()
            self.trial_counts[index] += 1

            # execute trial
            current_state, accepted = mover.trial(current_state, runner)
            if accepted:
                self.accepted_counts[index] += 1

        return current_state

    def _choose_mover(self):
        total = sum(w for c, w in self._movers_with_weights)
        r = random.uniform(0, total)
        upto = 0
        for i, (c, w) in enumerate(self._movers_with_weights):
            if upto + w >= r:
                return c, i
            upto += w
        assert False, "Should never get here"


class RandomTorsionMover(Mover):
    """
    Rotate a torsion to a random angle
    """

    def __init__(
        self,
        index1: indexing.AtomIndex,
        index2: indexing.AtomIndex,
        atom_indices: List[indexing.AtomIndex],
    ):
        """
        Initialize a RandomTorsionMover

        Args:
            index1: index of atom to rotate around
            index2: index of second atom to rotate around
            atom_indices: list of atom indices that should be rotated
        """
        assert isinstance(index1, indexing.AtomIndex)
        assert isinstance(index2, indexing.AtomIndex)
        self.index1 = index1
        self.index2 = index2
        for atom in atom_indices:
            assert isinstance(atom, indexing.AtomIndex)
        self.atom_indices = atom_indices

    def trial(
        self, state: interfaces.IState, runner: interfaces.IRunner
    ) -> Tuple[interfaces.IState, bool]:
        """
        Perform a Metropolis trial

        Args:
            starting_state: initial state of system
            runner: runner to evaluate energies

        Returns:
            the system state after Monte Carlo trials
        """
        starting_positions = state.positions.copy()
        starting_energy = state.energy

        angle = _generate_uniform_angle()
        trial_positions = starting_positions.copy()
        trial_positions[self.atom_indices, :] = _rotate_around_vector(
            starting_positions[self.index1, :],
            starting_positions[self.index2, :],
            angle,
            starting_positions[self.atom_indices, :],
        )
        state.positions = trial_positions
        trial_energy = runner.get_energy(state)

        accepted = _metropolis(starting_energy, trial_energy, 0.0)
        if accepted:
            state.energy = trial_energy
            state.positions = trial_positions
        else:
            state.energy = starting_energy
            state.positions = starting_positions

        return state, accepted


class DoubleTorsionMover(Mover):
    """
    A class to move pairs of torsions
    """

    def __init__(
        self,
        index1a: indexing.AtomIndex,
        index1b: indexing.AtomIndex,
        atom_indices1: List[indexing.AtomIndex],
        index2a: indexing.AtomIndex,
        index2b: indexing.AtomIndex,
        atom_indices2: List[indexing.AtomIndex],
    ):
        """
        Initialize a DoubleTorsionMover

        Args:
            index1a: first atom of first bond
            index1b: second atom of first bond
            atom_indices1: atoms to rotate around first bond
            index2a: first atom of second bond
            index2b: second atom of second bond
            atom_indices2 : atoms to rotate around second bond
        """
        assert isinstance(index1a, indexing.AtomIndex)
        assert isinstance(index1b, indexing.AtomIndex)
        assert isinstance(index2a, indexing.AtomIndex)
        assert isinstance(index2b, indexing.AtomIndex)
        for atom in atom_indices1:
            assert isinstance(atom, indexing.AtomIndex)
        for atom in atom_indices2:
            assert isinstance(atom, indexing.AtomIndex)
        self.index1a = index1a
        self.index1b = index1b
        self.atom_indices1 = atom_indices1
        self.index2a = index2a
        self.index2b = index2b
        self.atom_indices2 = atom_indices2

    def trial(
        self, state: interfaces.IState, runner: interfaces.IRunner
    ) -> Tuple[interfaces.IState, bool]:
        """
        Perform a Metropolis trial

        Args:
            starting_state: initial state of system
            runner: runner to evaluate energies

        Returns:
            the system state after Monte Carlo trials
        """
        starting_positions = state.positions.copy()
        starting_energy = state.energy

        angle1 = _generate_uniform_angle()
        angle2 = _generate_uniform_angle()

        trial_positions = starting_positions.copy()
        trial_positions[self.atom_indices1, :] = _rotate_around_vector(
            starting_positions[self.index1a, :],
            starting_positions[self.index1b, :],
            angle1,
            starting_positions[self.atom_indices1, :],
        )
        trial_positions[self.atom_indices2, :] = _rotate_around_vector(
            trial_positions[self.index2a, :],
            trial_positions[self.index2b, :],
            angle2,
            trial_positions[self.atom_indices2, :],
        )

        state.positions = trial_positions
        trial_energy = runner.get_energy(state)

        accepted = _metropolis(starting_energy, trial_energy, 0.0)
        if accepted:
            state.energy = trial_energy
            state.positions = trial_positions
        else:
            state.energy = starting_energy
            state.positions = starting_positions

        return state, accepted


class TranslationMover(Mover):
    """
    Translate a chain
    """

    def __init__(self, atom_indices: List[indexing.AtomIndex], move_size: float = 0.1):
        """
        Initialize a TranslationMover

        Args:
            atom_indices: atom indices to move
            move_size: standard deviation of random move in nanometers
        """
        self.atom_indices = atom_indices
        self.move_size = move_size

    def trial(
        self, state: interfaces.IState, runner: interfaces.IRunner
    ) -> Tuple[interfaces.IState, bool]:
        """
        Perform a Metropolis trial

        Args:
            starting_state: initial state of system
            runner: runner to evaluate energies

        Returns:
            the system state after Monte Carlo trials
        """
        starting_positions = state.positions.copy()
        starting_energy = state.energy

        trial_positions = starting_positions.copy()
        random_vector = np.random.normal(loc=0.0, scale=self.move_size, size=3)
        trial_positions[self.atom_indices, :] += random_vector
        state.positions = trial_positions
        trial_energy = runner.get_energy(state)

        accepted = _metropolis(starting_energy, trial_energy, bias=0.0)
        if accepted:
            state.energy = trial_energy
            state.positions = trial_positions
        else:
            state.energy = starting_energy
            state.positions = starting_positions

        return state, accepted


def _rotate_around_vector(
    p1: np.ndarray, p2: np.ndarray, angle: float, points: np.ndarray
) -> np.ndarray:
    """
    Rotate points around a vector

    Args:
        p1: first point on axis to rotate around
        p2: second point on axis to rotate around
        angle: angle in degrees
        points: array of shape (n_atoms, 3)

    Returns:
        rotated array of shape (n_atoms, 3)
    """
    direction = p2 - p1
    angle = angle / 180.0 * math.pi
    rot_mat = _rotation_matrix(angle, direction, point=p1)
    return _covert_from_homogeneous(np.dot(_convert_to_homogeneous(points), rot_mat))


def _metropolis(current_energy: float, trial_energy: float, bias: float) -> bool:
    """
    Perform a Metropolis accept/reject step.

    Args:
        current_energy: current energy in units of kT
        trial_energy: energy of trial in units of kT
        bias: negative log of ratio of forward and reverse move probabilities

    Returns:
        Indicates if step should be accepted
    """
    total_trial = trial_energy + bias
    delta = total_trial - current_energy
    if delta <= 0:
        return True
    else:
        metrop = math.exp(-delta)
        rand = random.random()
        if rand <= metrop:
            return True
        else:
            return False


def _generate_uniform_angle() -> float:
    """
    Generate a uniform angle in (0, 360]

    Returns:
        a random angle
    """
    return 360.0 * random.random()


def _rotation_matrix(angle, direction, point=None):
    """
    Adapted from transformations.py
    (C) 2006-2014 Christoph Gohlke
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = direction[:3] / np.linalg.norm(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M.T


def _convert_to_homogeneous(coords):
    homogeneous = np.ones((coords.shape[0], 4))
    homogeneous[:, :3] = coords
    return homogeneous


def _covert_from_homogeneous(coords):
    return coords[:, :3]
