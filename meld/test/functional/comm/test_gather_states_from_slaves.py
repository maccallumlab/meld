import numpy
from meld import comm
from meld.system.state import SystemState

N_ATOMS = 500
N_REPLICAS = 4
N_SPRINGS = 100


def generate_state(index):
    coords = index * numpy.ones((N_ATOMS, 3))
    vels = index * numpy.ones((N_ATOMS, 3))
    spring_states = index * numpy.ones(N_SPRINGS)
    lam = float(index) / 10.
    energy = float(index)
    spring_energies = index * numpy.ones(N_SPRINGS)

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

    state = generate_state(c.rank)

    if c.is_master():
        print __file__
        all_states = c.gather_states_from_slaves(state)
        check_state(all_states[0], 0)
        check_state(all_states[1], 1)
        check_state(all_states[2], 2)
        check_state(all_states[3], 3)

    else:
        c.send_state_to_master(state)

    print '\tSuccess'


if __name__ == '__main__':
    main()
