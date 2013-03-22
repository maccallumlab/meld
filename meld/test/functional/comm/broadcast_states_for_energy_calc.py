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
    assert state.positions[0, 0] == index
    assert state.velocities[0, 0] == index
    assert state.spring_states[0] == index
    assert state.lam == index / 10.
    assert state.energy == index
    assert state.spring_energies[0] == index


def main():
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS, N_SPRINGS)
    c.initialize()

    if c.is_master():
        states = [generate_state(index) for index in range(4)]
        c.broadcast_states_for_energy_calc_to_slaves(states)

    else:
        all_states = c.recieve_states_for_energy_calc_from_master()
        check_state(all_states[0], 0)
        check_state(all_states[1], 1)
        check_state(all_states[2], 2)
        check_state(all_states[3], 3)


if __name__ == '__main__':
    main()
