#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import comm

N_ATOMS = 500
N_REPLICAS = 4


def main():
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS)
    c.initialize()

    if c.is_leader():
        alphas = [0., 0.1, 0.2, 0.3]
        c.broadcast_alphas_to_workers(alphas)
        try:
            assert alphas[0] == 0.
        except AssertionError:
            print(f"Expected 0.0, but got {alphas[0]}")
            raise

    else:
        alpha = c.receive_alpha_from_leader()
        if c.rank == 1:
            try:
                assert alpha == 0.1
            except AssertionError:
                print(f"Expected 0.1, but got {alpha}")
                raise
        elif c.rank == 2:
            try:
                assert alpha == 0.2
            except AssertionError:
                print(f"Expected 0.2, but got {alpha}")
                raise
        elif c.rank == 3:
            try:
                assert alpha == 0.3
            except:
                print(f"Expected 0.3, but got {alpha}")
                raise


if __name__ == "__main__":
    main()
