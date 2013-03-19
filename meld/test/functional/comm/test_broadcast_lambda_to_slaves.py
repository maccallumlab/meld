from meld import comm

N_ATOMS = 500
N_REPLICAS = 4
N_SPRINGS = 100


def main():
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS, N_SPRINGS)
    c.initialize()

    if c.is_master():
        print __file__
        lambdas = [0., 0.1, 0.2, 0.3]
        c.broadcast_lambdas_to_slaves(lambdas)

    else:
        lam = c.recieve_lambda_from_master()
        if c.rank == 1:
            assert lam == 0.1
        elif c.rank == 2:
            assert lam == 0.2
        elif c.rank == 3:
            assert lam == 0.3

    print '\tSuccess'


if __name__ == '__main__':
    main()
