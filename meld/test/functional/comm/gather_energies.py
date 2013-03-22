import numpy as np
from meld import comm


N_ATOMS = 500
N_REPLICAS = 4
N_SPRINGS = 100


def generate_energies(index):
    return index * np.ones(N_REPLICAS)


def main():
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS, N_SPRINGS)
    c.initialize()

    energies = generate_energies(c.rank)

    if c.is_master():
        all_energies = c.gather_energies_from_slaves(energies)
        assert all_energies[0, 0] == 0.
        assert all_energies[0, 1] == 0.
        assert all_energies[0, 2] == 0.
        assert all_energies[0, 3] == 0.

        assert all_energies[1, 1] == 1.
        assert all_energies[2, 2] == 2.
        assert all_energies[3, 3] == 3.

    else:
        c.send_energies_to_master(energies)


if __name__ == '__main__':
    main()
