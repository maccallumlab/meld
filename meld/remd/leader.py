#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module for replica exchange leader
"""

from meld import interfaces
from meld import vault
from meld.remd import worker
from meld.remd import ladder
from meld.remd import adaptor

import logging
import math
import numpy as np
from typing import List, Sequence


logger = logging.getLogger(__name__)


class LeaderReplicaExchangeRunner:
    """
    Class to coordinate running of replica exchange

    This class doesn't really know much about the calculation that
    is happening, but it's the glue that holds everything together.
    """

    @property
    def n_replicas(self) -> int:
        """number of replicas"""
        return self._n_replicas

    @property
    def alphas(self) -> List[float]:
        """current values of alpha"""
        return self._alphas

    @property
    def step(self) -> int:
        """current step"""
        return self._step

    @property
    def max_steps(self) -> int:
        """number of steps to run"""
        return self._max_steps

    _alphas: List[float]

    def __init__(
        self,
        n_replicas: int,
        max_steps: int,
        ladder: ladder.NearestNeighborLadder,
        adaptor: adaptor.Adaptor,
    ) -> None:
        """
        Initialize a LeaderReplicaExchangeRunner

        Args:
            n_replicas: number of replicas
            max_steps: maximum number of steps to run
            ladder: ladder object to handle exchanges
            adaptor: adaptor object to handle alphas adaptation
        """
        self._n_replicas = n_replicas
        self._max_steps = max_steps
        self._step = 1
        self.ladder = ladder
        self.adaptor = adaptor
        self._setup_alphas()

    def to_worker(self) -> worker.WorkerReplicaExchangeRunner:
        """
        Convert leader to worker
        """
        return worker.WorkerReplicaExchangeRunner(self.step, self.max_steps)

    def run(
        self,
        communicator: interfaces.ICommunicator,
        system_runner: interfaces.IRunner,
        store: vault.DataStore,
    ):
        """
        Run replica exchange until finished

        Args:
            communicator: a communicator object to talk with workers
            system_runner: a interfaces.IRunner object to run the simulations
            store: a store object to handle storing data to disk
        """
        logger.info("Beginning replica exchange")
        # check to make sure n_replicas matches
        assert self._n_replicas == communicator.n_replicas
        assert self._n_replicas == store.n_replicas

        # load previous state from the store
        states = store.load_states(stage=self.step - 1)

        # we always minimize when we first start, either on the first
        # stage or the first stage after a restart
        minimize = True

        while self._step <= self._max_steps:
            logger.info(
                "Running replica exchange step %d of %d.", self._step, self._max_steps
            )

            # communicate state
            my_state = communicator.broadcast_states_to_workers(states)

            # update alphas
            system_runner.prepare_for_timestep(my_state, 0.0, self._step)
            self._alphas = self.adaptor.adapt(self._alphas, self._step)
            communicator.broadcast_alphas_to_workers(self._alphas)

            # do one step
            if minimize:
                logger.info("First step, minimizing and then running.")
                my_state = system_runner.minimize_then_run(my_state)
                minimize = False  # we don't need to minimize again
            else:
                logger.info("Running molecular dynamics.")
                my_state = system_runner.run(my_state)

            # gather all of the states
            states = communicator.exchange_states_for_energy_calc(my_state)

            # compute our energy for each state
            my_energies = self._compute_energies(states, system_runner)
            energies = communicator.gather_energies_from_workers(my_energies)

            # ask the ladder how to permute things
            permutation_vector = self.ladder.compute_exchanges(energies, self.adaptor)
            states = self._permute_states(permutation_vector, states, system_runner)

            # store everything
            store.save_states(states, self.step)
            store.append_traj(states[0], self.step)
            store.save_alphas(np.array(self._alphas), self.step)
            store.save_permutation_vector(permutation_vector, self.step)
            store.save_energy_matrix(energies, self.step)
            store.save_acceptance_probabilities(
                self.adaptor.get_acceptance_probabilities(), self.step
            )
            store.save_data_store()

            # on to the next step!
            self._step += 1
            store.save_remd_runner(self)
            store.backup(self.step - 1)
        logger.info(
            "Finished %d steps of replica exchange successfully.", self._max_steps
        )

    #
    # private helper methods
    #

    @staticmethod
    def _compute_energies(
        states: Sequence[interfaces.IState], system_runner: interfaces.IRunner
    ) -> List[float]:
        my_energies = []
        for state in states:
            my_energies.append(system_runner.get_energy(state))
        return my_energies

    @staticmethod
    def _permute_states(
        permutation_matrix: List[int],
        states: Sequence[interfaces.IState],
        system_runner: interfaces.IRunner,
    ) -> Sequence[interfaces.IState]:
        old_coords = [s.positions for s in states]
        old_velocities = [s.velocities for s in states]
        old_box_vectors = [s.box_vector for s in states]
        old_energy = [s.energy for s in states]
        old_params = [s.parameters for s in states]
        old_mappings = [s.mappings for s in states]
        assert system_runner.temperature_scaler is not None
        temperatures = [system_runner.temperature_scaler(s.alpha) for s in states]

        for i, index in enumerate(permutation_matrix):
            states[i].positions = old_coords[index]
            states[i].velocities = (
                math.sqrt(temperatures[i] / temperatures[index]) * old_velocities[index]
            )
            states[i].box_vector = old_box_vectors[index]
            states[i].energy = old_energy[index]
            states[i].parameters = old_params[index]
            states[i].mappings = old_mappings[index]
        return states

    def _setup_alphas(self) -> None:
        delta = 1.0 / (self._n_replicas - 1.0)
        self._alphas = [i * delta for i in range(self._n_replicas)]
