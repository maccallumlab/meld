import random
import math


class Ladder(object):
    def __init__(self, n_iterations):
        self.n_iterations = n_iterations

    def compute_exchanges(self, energies, adaptor):
        assert len(energies.shape) == 2
        assert energies.shape[0] == energies.shape[1]

        n_replicas = energies.shape[0]
        permutation_matrix = range(n_replicas)

        for iteration in range(self.n_iterations):
            delta = energies[0,0] - energies[0,1] + energies[1,1] - energies[1,0]
            accepted = False

            if delta >= 0:
                accepted = True
            else:
                metrop = math.exp(delta)
                rand = random.random()
                if rand < metrop:
                    accepted = True

            if accepted:
                permutation_matrix = self._swap(permutation_matrix)
                adaptor.update(0, 1, True)
            else:
                adaptor.update(0, 1, False)

        return permutation_matrix

    @staticmethod
    def _swap(permutation_matrix):
        return [ permutation_matrix[1], permutation_matrix[0] ]

