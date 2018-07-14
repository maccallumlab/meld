#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import random
import numpy as np
import math


class MonteCarloScheduler():
    """
    Weighted random selection of Monte Carlo moves
    """

    def __init__(self, movers_with_weights, update_trials):
        """
        Parameters
        ----------
        movers_with_weights : [(mover, weight)]
            List of (mover, weight) tuples. Weights do not need to be normalized.
        """
        self.update_trials = update_trials
        self._movers_with_weights = movers_with_weights

        self.trial_counts = np.zeros(len(self._movers_with_weights))
        self.accepted_counts = np.zeros(len(self._movers_with_weights))

    def update(self, starting_state, runner):
        """
        Perform a series of Monte Carlo moves

        Parameters
        ----------
        starting_state : SystemState
            Initial state of system
        runner : OpenMMRunner
            Runner to evaluate energies
        n_trials : int
            Number of trials to run

        Returns
        -------
        SystemState
            The system state after Monte Carlo trials
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


class RandomTorsionMover():
    """
    Rotate a torsion to a random angle
    """

    def __init__(self, index1, index2, atom_indices):
        """
        Parameters
        ----------
        index1 : int
            Index of atom to rotate around
        index2 : int
            Index of second atom to rotate around
        atom_indices : [int]
            List of atom indices that should be rotated
        """
        self.index1 = index1
        self.index2 = index2
        self.atom_indices = atom_indices

    def trial(self, state, runner):
        """
        Perform a Metropolis trial

        Parameters
        ----------
        state: SystemState
            Initial state of the system
        runner: OpenMMRunner
            Runner to evaluate the energies

        Returns
        -------
        SystemState, boolean
            System state after trial and indiciator if trial was accepted
        """
        starting_positions = state.positions.copy()
        starting_energy = state.energy

        angle = generate_uniform_angle()
        trial_positions = starting_positions.copy()
        trial_positions[self.atom_indices, :] = rotate_around_vector(
            starting_positions[self.index1, :],
            starting_positions[self.index2, :],
            angle,
            starting_positions[self.atom_indices, :],
        )
        state.positions = trial_positions
        trial_energy = runner.get_energy(state)

        accepted = metropolis(starting_energy, trial_energy, 0.0)
        if accepted:
            state.energy = trial_energy
            state.positions = trial_positions
        else:
            state.energy = starting_energy
            state.positions = starting_positions

        return state, accepted


class DoubleTorsionMover():
    def __init__(
        self, index1a, index1b, atom_indices1, index2a, index2b, atom_indices2
    ):
        """
        Parameters
        ----------
        index1a : int
        index1b : int
        atom_indices1 : [int]
        index2a : int
        index2b : int
        atom_indices2 : [int]
        """
        self.index1a = index1a
        self.index1b = index1b
        self.atom_indices1 = atom_indices1
        self.index2a = index2a
        self.index2b = index2b
        self.atom_indices2 = atom_indices2

    def trial(self, state, runner):
        """
        Perform a Metropolis trial

        Parameters
        ----------
        state : SystemState
            Initial state of the system
        runner: OpenMMRunner
            Runner to evaluate the energies

        Returns
        -------
        SystemState, boolean
            System state after trial and indiciator if trial was accepted
        """
        starting_positions = state.positions.copy()
        starting_energy = state.energy

        angle1 = generate_uniform_angle()
        angle2 = generate_uniform_angle()

        trial_positions = starting_positions.copy()
        trial_positions[self.atom_indices1, :] = rotate_around_vector(
            starting_positions[self.index1a, :],
            starting_positions[self.index1b, :],
            angle1,
            starting_positions[self.atom_indices1, :],
        )
        trial_positions[self.atom_indices2, :] = rotate_around_vector(
            trial_positions[self.index2a, :],
            trial_positions[self.index2b, :],
            angle2,
            trial_positions[self.atom_indices2, :],
        )

        state.positions = trial_positions
        trial_energy = runner.get_energy(state)

        accepted = metropolis(starting_energy, trial_energy, 0.0)
        if accepted:
            state.energy = trial_energy
            state.positions = trial_positions
        else:
            state.energy = starting_energy
            state.positions = starting_positions

        return state, accepted


class TranslationMover():
    """
    Translate a chain
    """

    def __init__(self, atom_indices, move_size=0.1):
        """
        Parameters
        ----------
        atom_indices: [int]
            List of atoms to translate
        move_size: float
            Standard deviation of random move in nanometers
        """
        self.atom_indices = atom_indices
        self.move_size = move_size

    def trial(self, state, runner):
        """
        Perform a metropolis trial

        :param state: initial `SystemState`
        :param runner: `OpenMMRunner` to evaluate the energies
        :return: updated `SystemState`
        """
        starting_positions = state.positions.copy()
        starting_energy = state.energy

        trial_positions = starting_positions.copy()
        random_vector = np.random.normal(loc=0.0, scale=self.move_size, size=3)
        trial_positions[self.atom_indices, :] += random_vector
        state.positions = trial_positions
        trial_energy = runner.get_energy(state)

        accepted = metropolis(starting_energy, trial_energy, bias=0.0)
        if accepted:
            state.energy = trial_energy
            state.positions = trial_positions
        else:
            state.energy = starting_energy
            state.positions = starting_positions

        return state, accepted


def rotate_around_vector(p1, p2, angle, points):
    """
    Parameters
    ----------
    p1 : ndarray
        First point on axis to rotate around
    p2 : ndarray
        Second point on axis to rotate around
    angle : float
        Angle in degrees
    points : ndarray
        Numpy array of shape (n_atoms, 3)

    Returns
    -------
    ndarray
        Rotate array of shape (n_atoms, 3)
    """
    direction = p2 - p1
    angle = angle / 180. * math.pi
    rot_mat = _rotation_matrix(angle, direction, point=p1)
    return _covert_from_homogeneous(np.dot(_convert_to_homogeneous(points), rot_mat))


def metropolis(current_energy, trial_energy, bias):
    """
    Perform a Metropolis accept/reject step.

    Parameters
    ----------
    current_energy : float
        Current energy in units of kT
    trial_energy : float
        Energy of trial in units of kT
    bias : float
        Negative log of ratio of forward and reverse move probabilities

    Returns
    -------
    boolean
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


def generate_uniform_angle():
    """
    Generate a uniform angle in (0, 360]
    Returns
    -------
    float
    """
    return 360. * random.random()


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
