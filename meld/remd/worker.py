#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
A module for replica exchange workers
"""

from meld import interfaces


class WorkerReplicaExchangeRunner:
    """
    This class coordinates running replica exchange on the workers.
    """

    def __init__(self, step: int, max_steps: int):
        """
        Initialize a WorkerReplicaExchangeRunner

        Args:
            step: current step
            max_steps: number of steps to run
        """
        self._step = step
        self._max_steps = max_steps

    @property
    def step(self) -> int:
        """current step"""
        return self._step

    @property
    def max_steps(self) -> int:
        """number of steps to run"""
        return self._max_steps

    def run(
        self, communicator: interfaces.ICommunicator, system_runner: interfaces.IRunner
    ) -> None:
        """
        Continue running worker jobs until done.

        Args:
            communicator: a communicator object for talking to the leader
            system_runner: a system runner object for actually running the simulations
        """
        # we always minimize when we first start, either on the first
        # stage or the first stage after a restart
        minimize = True
        while self._step <= self._max_steps:
            # update simulation conditions
            state = communicator.receive_state_from_leader()
            new_alpha = communicator.receive_alpha_from_leader()

            state.alpha = new_alpha

            system_runner.prepare_for_timestep(state, new_alpha, self._step)

            # do one round of simulation
            if minimize:
                state = system_runner.minimize_then_run(state)
                minimize = False  # we don't need to minimize again
            else:
                state = system_runner.run(state)

            # compute energies
            states = communicator.exchange_states_for_energy_calc(state)

            energies = []
            for state in states:
                energy = system_runner.get_energy(state)
                energies.append(energy)
            communicator.send_energies_to_leader(energies)

            self._step += 1
