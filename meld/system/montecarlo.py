#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import random
import numpy as np
import math


class MonteCarloScheduler(object):
    """
    Weighted random selection of Monte Carlo moves
    """
    def __init__(self, movers_with_weights, update_trials):
        """
        :param movers_with_weights: a list of (mover, weight) tuples;
                                    weights do not need to be normalized
        """
        self.update_trials = update_trials
        self._movers_with_weights = movers_with_weights

        self.trial_counts = np.zeros(len(self._movers_with_weights))
        self.accepted_counts = np.zeros(len(self._movers_with_weights))

    def update(self, starting_state, runner):
        """
        Perform a series of Monte Carlo moves
        :param starting_state: initial `SystemState`
        :param runner: an `OpenMMRunner` to evaluate energies
        :param n_trials: int number of trials
        :return: new `SystemState` object
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
        assert False, 'Should never get here'


class RandomTorsionMover(object):
    """
    Rotate a torsion to a random angle
    """
    def __init__(self, index1, index2, atom_indices):
        """
        :param index1: index of atom to rotate around
        :param index2: index of second atom defining rotation vector
        :param atom_indices: list of atom indices that should be rotated
        """
        self.index1 = index1
        self.index2 = index2
        self.atom_indices = atom_indices

    def trial(self, state, runner):
        """
        Perform a Metropolis trial
        :param state: initial `SystemState`
        :param runner: `OpenMMRunner` to evaluate the energies
        :return: updated `SystemState`
        """
        starting_positions = state.positions.copy()
        starting_energy = state.energy

        angle = generate_uniform_angle()
        trial_positions = starting_positions.copy()
        trial_positions[self.atom_indices, :] = rotate_around_vector(
            starting_positions[self.index1, :],
            starting_positions[self.index2, :],
            angle, starting_positions[self.atom_indices, :])
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


class DoubleTorsionMover(object):
    def __init__(self, index1a, index1b, atom_indices1, index2a,
                 index2b, atom_indices2):
        self.index1a = index1a
        self.index1b = index1b
        self.atom_indices1 = atom_indices1
        self.index2a = index2a
        self.index2b = index2b
        self.atom_indices2 = atom_indices2

    def trial(self, state, runner):
        """
        Perform a Metropolis trial
        :param state: initial `SystemState`
        :param runner: `OpenMMRunner` to evaluate the energies
        :return: updated `SystemState`
        """
        starting_positions = state.positions.copy()
        starting_energy = state.energy

        angle1 = generate_uniform_angle()
        angle2 = generate_uniform_angle()

        trial_positions = starting_positions.copy()
        trial_positions[self.atom_indices1, :] = rotate_around_vector(
            starting_positions[self.index1a, :],
            starting_positions[self.index1b, :],
            angle1, starting_positions[self.atom_indices1, :])
        trial_positions[self.atom_indices2, :] = rotate_around_vector(
            trial_positions[self.index2a, :],
            trial_positions[self.index2b, :],
            angle2, trial_positions[self.atom_indices2, :])

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


def rotate_around_vector(p1, p2, angle, points):
    """
    :param p1: first point on axis to rotate around
    :param p2: second point on axis to rotate around
    :param angle: angle in degrees
    :param points: numpy array of shape (n_atoms, 3)
    :return: rotated numpy array of shape (n_atoms, 3)
    """
    direction = p2 - p1
    angle = angle / 180. * math.pi
    rot_mat = _rotation_matrix(angle, direction, point=p1)
    return _covert_from_homogeneous(
        np.dot(_convert_to_homogeneous(points), rot_mat))


def metropolis(current_energy, trial_energy, bias):
    """
    Perform a Metropolis accept/reject step.

    :param current_energy: current energy in units of kT
    :param trial_energy: energy of trial in units of kT
    :param bias: negative log of ratio of forward and reverse
                 move probabilities
    :return: boolean indicating to accept or reject the step
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
    :return: float angle
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
    R += np.array([[ 0.0,          -direction[2],  direction[1]],
                   [ direction[2],  0.0,          -direction[0]],
                   [-direction[1],  direction[0],  0.0]])
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
