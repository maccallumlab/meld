#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld.vault import DataStore
from meld.system.state import SystemState
from meld.remd import follower
from meld.remd.ladder import NearestNeighborLadder
from meld.remd.adaptor import Adaptor
from meld.remd.reseed import NullReseeder
from meld.system.runner import ReplicaRunner
from meld.comm import MPICommunicator
import logging
import math
from typing import List, Union


logger = logging.getLogger(__name__)


class LeaderReplicaExchangeRunner:
    """
    Class to coordinate running of replica exchange

    This class doesn't really know much about the calculation that
    is happening, but it's the glue that holds everything together.

    :param n_replicas: number of replicas
    :param max_steps: maximum number of steps to run
    :param ladder: Ladder object to handle exchanges
    :param adaptor: Adaptor object to handle alphas adaptation

    """

    #
    # read only properties
    #

    @property
    def n_replicas(self) -> int:
        return self._n_replicas

    @property
    def alphas(self) -> List[float]:
        return self._alphas

    @property
    def step(self) -> int:
        return self._step

    @property
    def max_steps(self) -> int:
        return self._max_steps

    #
    # public methods
    #

    _alphas: List[float]

    def __init__(
        self,
        n_replicas: int,
        max_steps: int,
        ladder: NearestNeighborLadder,
        adaptor: Adaptor,
    ) -> None:
        self._n_replicas = n_replicas
        self._max_steps = max_steps
        self._step = 1
        self.ladder = ladder
        self.adaptor = adaptor
        self._setup_alphas()
        self.reseeder = NullReseeder()

    def to_follower(self) -> follower.FollowerReplicaExchangeRunner:
        """
        Return a FollowerReplicaExchangeRunner based on self.

        """
        return follower.FollowerReplicaExchangeRunner.from_leader(self)

    def run(
        self,
        communicator: MPICommunicator,
        system_runner: ReplicaRunner,
        store: DataStore,
    ):
        """
        Run replica exchange until finished

        :param communicator: A communicator object to talk with followers
        :param system_runner: a ReplicaRunner object to run the simulations
        :param store: a Store object to handle storing data to disk

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
            my_state = communicator.broadcast_states_to_followers(states)

            # update alphas
            system_runner.prepare_for_timestep(my_state, 0.0, self._step)
            self._alphas = self.adaptor.adapt(self._alphas, self._step)
            communicator.broadcast_alphas_to_followers(self._alphas)

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
            energies = communicator.gather_energies_from_followers(my_energies)

            # ask the ladder how to permute things
            permutation_vector = self.ladder.compute_exchanges(energies, self.adaptor)
            states = self._permute_states(permutation_vector, states, system_runner)

            # perform reseeding if it is time
            self.reseeder.reseed(self.step, states, store)

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

    #
    # private helper methods
    #

    @staticmethod
    def _compute_energies(
        states: List[SystemState], system_runner: ReplicaRunner
    ) -> List[float]:
        my_energies = []
        for state in states:
            my_energies.append(system_runner.get_energy(state))
        return my_energies

    @staticmethod
    def _permute_states(
        permutation_matrix: List[int],
        states: List[SystemState],
        system_runner: ReplicaRunner,
    ) -> List[SystemState]:
        old_coords = [s.positions for s in states]
        old_velocities = [s.velocities for s in states]
        old_box_vectors = [s.box_vector for s in states]
        old_energy = [s.energy for s in states]
        old_params = [s.parameters for s in states]
        temperatures = [system_runner.temperature_scaler(s.alpha) for s in states]

        for i, index in enumerate(permutation_matrix):
            states[i].positions = old_coords[index]
            states[i].velocities = (
                math.sqrt(temperatures[i] / temperatures[index]) * old_velocities[index]
            )
            states[i].box_vector = old_box_vectors[index]
            states[i].energy = old_energy[index]
            states[i].parameters = old_params[index]
        return states

    def _setup_alphas(self) -> None:
        delta = 1.0 / (self._n_replicas - 1.0)
        self._alphas = [i * delta for i in range(self._n_replicas)]
