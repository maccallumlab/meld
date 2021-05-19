#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import numpy as np  # type: ignore
from meld import comm


N_ATOMS = 5000
N_REPLICAS = 4


def generate_energies(index):
    return index * np.ones(N_REPLICAS)


def main():
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS)
    c.initialize()

    energies = generate_energies(c.rank)

    if c.is_leader():
        all_energies = c.gather_energies_from_workers(energies)
        assert all_energies[0, 0] == 0.
        assert all_energies[0, 1] == 0.
        assert all_energies[0, 2] == 0.
        assert all_energies[0, 3] == 0.

        assert all_energies[1, 1] == 1.
        assert all_energies[2, 2] == 2.
        assert all_energies[3, 3] == 3.

    else:
        c.send_energies_to_leader(energies)


if __name__ == "__main__":
    main()
