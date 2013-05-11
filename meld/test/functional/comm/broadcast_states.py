import numpy as np
from meld import comm
from meld.system.state import SystemState

N_ATOMS = 500
N_REPLICAS = 4
N_SPRINGS = 100


def generate_state(index):
    coords = index * np.ones((N_ATOMS, 3))
    vels = index * np.ones((N_ATOMS, 3))
    alpha = float(index) / 10.
    energy = float(index)

    return SystemState(coords, vels, alpha, energy)


def check_state(state, index):
    assert state.positions[0, 0] == index
    assert state.velocities[0, 0] == index
    assert state.alpha == index / 10.
    assert state.energy == index


def main():
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS, N_SPRINGS)
    c.initialize()

    if c.is_master():
        states = [generate_state(index) for index in range(4)]
        my_state = c.broadcast_states_to_slaves(states)
        check_state(my_state, 0)

    else:
        my_state = c.recieve_state_from_master()
        check_state(my_state, c.rank)


if __name__ == '__main__':
    main()
