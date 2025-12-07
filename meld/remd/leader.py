#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module for replica exchange leader
"""

import logging
from typing import List, Sequence

import numpy as np

from meld import interfaces, vault
from meld.remd import adaptor, ladder, worker
from meld.remd.permute import permute_states
from meld.system import gameld

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
        # Check that the number of replicas is divisible by the number of workers.
        if communicator.n_replicas % communicator.n_workers:
            raise ValueError(
                "The number of replicas must be divisible by the number of workers."
            )
        # check to make sure n_replicas matches
        assert self._n_replicas == communicator.n_replicas
        assert self._n_replicas == store.n_replicas

        logger.info("Beginning replica exchange")

        # load previous state from the store
        all_states = store.load_states(stage=self.step - 1)

        # we always minimize when we first start, either on the first
        # stage or the first stage after a restart
        minimize = True

        while self._step <= self._max_steps:
            logger.info(
                "Running replica exchange step %d of %d.", self._step, self._max_steps
            )

            # communicate state
            leader_states = communicator.distribute_states_to_workers(all_states)

            # update alphas
            self._alphas = self.adaptor.adapt(self._alphas, self._step)
            my_alphas = communicator.distribute_alphas_to_workers(self._alphas)

            for i, (state, alpha) in enumerate(zip(leader_states, my_alphas)):
                state.alpha = alpha

                logger.info("Running Hamiltonian %d of %d", i + 1, len(leader_states))
                system_runner.prepare_for_timestep(state, alpha, self._step)

                # do one step
                if minimize:
                    logger.info("First step, minimizing and then running.")
                    state = system_runner.minimize_then_run(state)
                else:
                    logger.info("Running molecular dynamics.")
                    state = system_runner.run(state)

                leader_states[i] = state

            minimize = False  # we don't need to minimize again

            # Gather and distribute all of the states
            all_states = communicator.gather_states_from_workers(leader_states)
            communicator.broadcast_all_states_to_workers(all_states)

            # compute our energy for each state
            leader_energies = self._compute_energies(
                leader_states, all_states, system_runner
            )
            energies = communicator.gather_energies_from_workers(leader_energies)

            # ask the ladder how to permute things
            permutation_vector = self.ladder.compute_exchanges(energies, self.adaptor)
            all_states = permute_states(
                permutation_vector, all_states, system_runner, self.step
            )

            if system_runner._options.enable_gamd == True:  # type: ignore
                # if it's time, change thresholds
                leader: bool = True
                gameld.change_thresholds(self.step, system_runner, communicator, leader)

            # store everything
            store.save_states(all_states, self.step)
            store.append_traj(all_states[0], self.step)
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

    def _compute_energies(
        self,
        hamiltonian_states: Sequence[interfaces.IState],
        all_states: Sequence[interfaces.IState],
        system_runner: interfaces.IRunner,
    ) -> np.ndarray:
        energies = []
        for hamiltonian in hamiltonian_states:
            hamiltonian_energies = []
            for state in all_states:
                system_runner.prepare_for_timestep(state, hamiltonian.alpha, self._step)
                energy = system_runner.get_energy(state)
                hamiltonian_energies.append(energy)
            energies.append(hamiltonian_energies)

        return np.array(energies)

    def _setup_alphas(self) -> None:
        delta = 1.0 / (self._n_replicas - 1.0)
        self._alphas = [i * delta for i in range(self._n_replicas)]
