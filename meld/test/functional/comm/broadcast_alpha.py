from meld import comm

N_ATOMS = 500
N_REPLICAS = 4


def main():
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS)
    c.initialize()

    if c.is_master():
        alphas = [0., 0.1, 0.2, 0.3]
        c.broadcast_alphas_to_slaves(alphas)

    else:
        alpha = c.receive_alpha_from_master()
        if c.rank == 1:
            assert alpha == 0.1
        elif c.rank == 2:
            assert alpha == 0.2
        elif c.rank == 3:
            assert alpha == 0.3


if __name__ == '__main__':
    main()
