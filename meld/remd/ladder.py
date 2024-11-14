#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to handle replica exchange swaps
"""

import logging
import math
import random
from typing import List

import numpy as np  # type: ignore

from meld import util
from meld.remd import adaptor

logger = logging.getLogger(__name__)


class NearestNeighborLadder:
    """
    Class to compute replica exchange swaps between neighboring replicas.
    """

    def __init__(self, n_trials: int) -> None:
        """
        Initialize a NearestNeighborLadder

        Args:
            n_trials: number of swaps to attempt
        """
        self.n_trials = n_trials

    @util.log_timing(logger)
    def compute_exchanges(
        self, energies: np.ndarray, adaptor: adaptor.Adaptor
    ) -> List[int]:
        """
        Compute the exchanges from a given energy matrix.

        Args:
            energies: numpy array of energies (see below for details)
            adaptor: replica exchange adaptor that is updated every attempted swap
        Returns:
            a permutation vector (see below for details)

        The energy matrix should be an n_rep x n_rep numpy array. All energies
        are in dimensionless units (unit of kT). Each column represents a
        particular structure, while each row represents a particular
        combination of temperature and Hamiltonian. So, ``energies[i,j]`` is
        the energy of structure j with temperature and hamiltonian i. The
        diagonal ``energies[i,i]`` are the energies that were actually
        simulated.

        This method will attempt ``self.n_trials`` swaps between randomly
        chosen pairs of adjacent replicas. It will return a permutation vector
        that describes which index each structure should be at after swapping.
        So, if the output was ``[2, 0, 1]``, it would mean that replica 0
        should now be at index 2, replica 1 should now be at index 0, and
        replica 2 should now be at index 1. Output of ``[0, 1, 2]`` would
        mean no change.

        The adaptor object is called once for each attempted exchange with the
        indices of the attempted swap and the success or failure of the swap.
        """
        assert len(energies.shape) == 2
        assert energies.shape[0] == energies.shape[1]

        n_replicas = energies.shape[0]
        permutation_vector = list(range(n_replicas))

        choices = range(n_replicas - 1)
        for iteration in range(self.n_trials):
            i = random.choice(choices)
            j = i + 1
            self._do_trial(i, j, permutation_vector, energies, adaptor)

        return permutation_vector

    def _do_trial(
        self,
        i: int,
        j: int,
        permutation_vector: List[int],
        energies: np.ndarray,
        adaptor: adaptor.Adaptor,
    ) -> None:
        """Perform a replica exchange trial"""
        delta = energies[i, i] - energies[j, i] + energies[j, j] - energies[i, j]
        accepted = False

        if delta >= 0:
            accepted = True
        else:
            metrop = math.exp(delta)
            rand = random.random()
            if rand < metrop:
                accepted = True

        if accepted:
            self._swap_permutation(i, j, permutation_vector)
            self._swap_energies(i, j, energies)
            adaptor.update(i, True)
        else:
            adaptor.update(i, False)

    @staticmethod
    def _swap_permutation(i: int, j: int, permutation_vector: List[int]) -> None:
        """Swap two elements of the permutation matrix"""
        permutation_vector[i], permutation_vector[j] = (
            permutation_vector[j],
            permutation_vector[i],
        )

    @staticmethod
    def _swap_energies(i: int, j: int, energies: np.ndarray) -> None:
        """Swap two columns of the energy matrix"""
        energies[:, [i, j]] = energies[:, [j, i]]
