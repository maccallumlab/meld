import numpy as np
from meld import comm
from meld.system.state import SystemState

N_ATOMS = 500
N_REPLICAS = 4
N_SPRINGS = 100


def generate_state(index):
    coords = index * np.ones((N_ATOMS, 3))
    vels = index * np.ones((N_ATOMS, 3))
    spring_states = index * np.ones(N_SPRINGS)
    lam = float(index) / 10.
    energy = float(index)
    spring_energies = index * np.ones(N_SPRINGS)

    return SystemState(coords, vels, spring_states, lam, energy, spring_energies)


def check_state(state, index):
    assert state.coords[0, 0] == index
    assert state.vels[0, 0] == index
    assert state.spring_states[0] == index
    assert state.lam == index / 10.
    assert state.energy == index
    assert state.spring_energies[0] == index


def main():
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS, N_SPRINGS)
    c.initialize()

    if c.is_master():
        print __file__
        states = [generate_state(index) for index in range(4)]
        my_state = c.broadcast_states_to_slaves(states)
        check_state(my_state, 0)

    else:
        my_state = c.recieve_state_from_master()
        check_state(my_state, c.rank)

    print '\tSuccess'


if __name__ == '__main__':
    main()
