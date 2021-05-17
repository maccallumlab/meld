#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
A module for running a replica exchange simulation on a single worker
"""

import numpy as np  # type: ignore
import logging
import math
from .ladder import NearestNeighborLadder
from .adaptor import Adaptor


logger = logging.getLogger(__name__)


class MultiplexReplicaExchangeRunner:
    """
    Class to coordinate running of replica exchange on a single worker
    """

    @property
    def n_replicas(self):
        """number of replicas"""
        return self._n_replicas

    @property
    def alphas(self):
        """current alpha values"""
        return self._alphas

    @property
    def step(self):
        """current step"""
        return self._step

    @property
    def max_steps(self):
        """number of steps to run"""
        return self._max_steps

    def __init__(
        self,
        n_replicas: int,
        max_steps: int,
        ladder: NearestNeighborLadder,
        adaptor: Adaptor,
        step: int,
    ):
        """
        Initialize a MultiplexReplicaExchangeRunner

        Args:
            n_replicas: number of replicas
            max_steps: maximum number of steps to run
            ladder: Ladder object to handle exchanges
            adaptor: Adaptor object to handle alphas adaptation
            step: current step
        """
        self._n_replicas = n_replicas
        self._max_steps = max_steps
        self._step = step
        self.ladder = ladder
        self.adaptor = adaptor

        self._alphas = None
        self._setup_alphas()

    def run(self, system_runner, store):
        """
        Run replica exchange until finished

        Args:
            system_runner: a replica runner to run the simulations
            store: a store object to handle storing data to disk
        """
        logger.info("Beginning replica exchange")
        # check to make sure n_replicas matches
        assert self._n_replicas == store.n_replicas

        # load previous state from the store
        states = store.load_states(stage=self.step - 1)

        while self._step <= self._max_steps:
            logger.info(
                "Running replica exchange step %d of %d.", self._step, self._max_steps
            )
            # update alphas
            self._alphas = self.adaptor.adapt(self._alphas, self._step)

            for state_index in range(self._n_replicas):
                states[state_index].alpha = self._alphas[state_index]
                system_runner.prepare_for_timestep(
                    self._alphas[state_index], self._step
                )

                if self._step == 1:
                    logger.info("First step, minimizing and then running.")
                    states[state_index] = system_runner.minimize_then_run(
                        states[state_index]
                    )
                else:
                    logger.info("Running molecular dynamics.")
                    states[state_index] = system_runner.run(states[state_index])

            energies = []
            for state_index in range(self._n_replicas):
                system_runner.prepare_for_timestep(
                    self._alphas[state_index], self._step
                )
                # compute our energy for each state
                my_energies = self._compute_energies(states, system_runner)
                energies.append(my_energies)
            energies = np.array(energies)

            # ask the ladder how to permute things
            permutation_vector = self.ladder.compute_exchanges(energies, self.adaptor)
            states = self._permute_states(permutation_vector, states, system_runner)

            # store everything
            store.save_states(states, self.step)
            store.append_traj(states[0], self.step)
            store.save_alphas(self._alphas, self.step)
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

    @staticmethod
    def _compute_energies(states, system_runner):
        my_energies = []
        for state in states:
            my_energies.append(system_runner.get_energy(state))
        return my_energies

    @staticmethod
    def _permute_states(permutation_matrix, states, system_runner):
        old_coords = [s.positions for s in states]
        old_velocities = [s.velocities for s in states]
        old_box_vectors = [s.box_vector for s in states]
        old_energy = [s.energy for s in states]
        temperatures = [system_runner.temperature_scaler(s.alpha) for s in states]
        for i, index in enumerate(permutation_matrix):
            states[i].positions = old_coords[index]
            states[i].box_vector = old_box_vectors[index]
            states[i].velocities = (
                math.sqrt(temperatures[i] / temperatures[index]) * old_velocities[index]
            )
            states[i].energy = old_energy[index]
        return states

    def _setup_alphas(self):
        delta = 1.0 / (self._n_replicas - 1.0)
        self._alphas = [i * delta for i in range(self._n_replicas)]
