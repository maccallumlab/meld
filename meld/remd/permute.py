import math
from typing import List

from meld.interfaces import IRunner, IState


def permute_states(
    permutation_matrix: List[int], states: List[IState], runner: IRunner, timestep: int
) -> List[IState]:
    # We don't swap the energies or group energies, as we will
    # be recomputing them below.
    old_coords = [s.positions for s in states]
    old_velocities = [s.velocities for s in states]
    old_box_vectors = [s.box_vector for s in states]
    old_params = [s.parameters for s in states]
    old_mappings = [s.mappings for s in states]
    old_alignments = [s.rdc_alignments for s in states]

    assert runner.temperature_scaler is not None
    temperatures = [runner.temperature_scaler(s.alpha) for s in states]

    for i, index in enumerate(permutation_matrix):
        states[i].positions = old_coords[index]
        states[i].velocities = (
            math.sqrt(temperatures[i] / temperatures[index]) * old_velocities[index]
        )
        states[i].box_vector = old_box_vectors[index]
        states[i].parameters = old_params[index]
        states[i].mappings = old_mappings[index]
        states[i].rdc_alignments = old_alignments[index]

    for state in states:
        runner.prepare_for_timestep(state, state.alpha, timestep)
        energy = runner.get_energy(state)
        group_energies = runner.get_group_energies(state)
        state.energy = energy
        state.group_energies = group_energies

    return states
