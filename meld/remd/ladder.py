import random
import math


class NearestNeighborLadder(object):
    '''
    Class to compute replica exchange swaps between neighboring replicas.
    '''
    def __init__(self, n_trials):
        '''
        Initialize a Ladder object.

        Parameters:
            n_trials -- total number of replica exchange swaps to attempt

        '''
        self.n_trials = n_trials

    def compute_exchanges(self, energies, adaptor):
        '''
        Compute the exchanges given an energy matrix.

        Parameters:
            energies -- numpy array of energies (see below for details)
            adaptor -- replica exchange adaptor that is updated every attempted swap
        Returns:
            a permutation vector (see below for details)

        The energy matrix should be an n_rep x n_rep numpy array. All energies are in dimenstinless units (unit of kT).
        Each row represents a particular structure, while each replica represents a particular combination of
        temperature and Hamiltonian. So, energies[i,j] is the energy of structure i with temperature and hamiltonian j.
        The diagnonal energies[i,i] is the energies that were actually simulated.

        This method will attempt self.n_trials swaps between randomly chosen pairs of adjacent replicas. It will
        return a permutation vector that describes which index each structure should be at after swapping. So, if the
        output was [2, 0, 1], it would mean that replica 0 should now be at index 2, replica 1 should now be at index 0,
        and replica 2 should now be at index 1. Output of [0, 1, 2] would mean no change.

        The adaptor object is called once for each attempted exchange with the indices of the attempted swap and the
        success or failure of the swap.

        '''
        assert len(energies.shape) == 2
        assert energies.shape[0] == energies.shape[1]

        n_replicas = energies.shape[0]
        permutation_matrix = range(n_replicas)

        choices = range(n_replicas - 1)
        for iteration in range(self.n_trials):
            i = random.choice(choices)
            j = i + 1
            self._do_trial(i, j, permutation_matrix, energies, adaptor)

        return permutation_matrix

    def _do_trial(self, i, j, permutation_matrix, energies, adaptor):
        '''Perform a replica exchange trial'''
        delta = energies[i, i] - energies[i, j] + energies[j, j] - energies[j, i]
        accepted = False

        if delta >= 0:
            accepted = True
        else:
            metrop = math.exp(delta)
            rand = random.random()
            if rand < metrop:
                accepted = True

        if accepted:
            self._swap_permutation(i, j, permutation_matrix)
            self._swap_energies(i, j, energies)
            adaptor.update(i, True)
        else:
            adaptor.update(i, False)

    @staticmethod
    def _swap_permutation(i, j, permutation_matrix):
        '''Swap two elements of the permutation matrix'''
        permutation_matrix[i], permutation_matrix[j] = permutation_matrix[j], permutation_matrix[i]

    @staticmethod
    def _swap_energies(i, j, energies):
        '''Swap two rows of the energy matrix'''
        energies[[i, j], :] = energies[[j, i], :]
