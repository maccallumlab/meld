#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
This module implements a number of interfaces that help to minimize coupling
and eliminate circular imports
"""

from __future__ import annotations
from typing import Sequence, Optional
from abc import ABC, abstractmethod
import numpy as np


class ICommunicator(ABC):
    """
    Interface for communication between leader and workers
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize and start MPI
        """
        pass

    @abstractmethod
    def is_leader(self) -> bool:
        """
        Is this the leader node?

        Returns:
            :const:`True` if we are the leader, otherwise :const:`False`
        """
        pass

    @abstractmethod
    def barrier(self) -> None:
        """
        Wait until all workers reach this point
        """
        pass

    @abstractmethod
    def broadcast_alphas_to_workers(self, alphas: Sequence[float]) -> None:
        """
        Send the alpha values to the workers.

        Args:
            alphas: a list of alpha values, one for each replica.

        .. note::
           The leader's alpha value should be included in :code:`alphas`.
           The leader's node will always be at :code:`alpha=0.0`.
        """
        pass

    @abstractmethod
    def receive_alpha_from_leader(self) -> float:
        """
        Receive alpha value from leader node.

        Returns:
            value for alpha in ``[0,1]``
        """
        pass

    @abstractmethod
    def broadcast_states_to_workers(self, states: Sequence[IState]) -> IState:
        """
        Send a state to each worker.

        Args:
            states: a list of states.

        Returns:
            the state to run on the leader node

        .. note::
           The list of states should include the state for the leader node.
        """
        pass

    @abstractmethod
    def receive_state_from_leader(self) -> IState:
        """
        Get state to run for this step

        Returns:
            the state to run for this step
        """
        pass

    @abstractmethod
    def gather_states_from_workers(self, state_on_leader: IState) -> Sequence[IState]:
        """
        Receive states from all workers

        Args:
            state_on_leader: the state on the leader after simulating

        Returns:
            A list of states, one from each replica.
        """
        pass

    @abstractmethod
    def send_state_to_leader(self, state: IState) -> None:
        """
        Send state to leader

        Args:
            state: State to send to leader.
        """
        pass

    @abstractmethod
    def broadcast_states_for_energy_calc_to_workers(
        self, states: Sequence[IState]
    ) -> None:
        """
        Broadcast states to all workers.

        Send all results from this step to every worker so that we can
        calculate the energies and do replica exchange.

        Args:
            states: a list of states
        """
        pass

    @abstractmethod
    def exchange_states_for_energy_calc(self, state: IState) -> Sequence[IState]:
        """
        Exchange states between all processes.

        Args:
            state: the state for this node

        Returns:
            a list of states from all nodes
        """
        pass

    @abstractmethod
    def receive_states_for_energy_calc_from_leader(self) -> Sequence[IState]:
        """
        Receive all states from leader.

        Returns:
            a list of states to calculate the energy of
        """
        pass

    @abstractmethod
    def gather_energies_from_workers(
        self, energies_on_leader: Sequence[float]
    ) -> np.ndarray:
        """
        Receive a list of energies from each worker.

        Args:
            energies_on_leader: a list of energies from the leader

        Returns:
            a square matrix of every state on every replica to be used for replica exchange
        """
        pass

    @abstractmethod
    def send_energies_to_leader(self, energies: Sequence[float]) -> None:
        """
        Send a list of energies to the leader.

        Args:
            energies: a list of energies to send to the leader
        """
        pass

    @abstractmethod
    def negotiate_device_id(self) -> int:
        """
        Negotiate CUDA device id

        Returns:
            the cuda device id to use

        """
        pass

    @property
    @abstractmethod
    def n_replicas(self) -> int:
        """number of replicas"""
        pass

    @property
    @abstractmethod
    def n_atoms(self) -> int:
        """number of atoms"""
        pass

    @property
    @abstractmethod
    def rank(self) -> int:
        """rank of this worker"""
        pass


class IState(ABC):
    """
    Interface for SystemState
    """

    positions: np.ndarray
    velocities: np.ndarray
    alpha: float
    energy: float
    box_vector: np.ndarray


class IRunner(ABC):
    """
    Interface for replica runners
    """

    temperature_scaler: Optional[ITemperatureScaler]

    @abstractmethod
    def minimize_then_run(self, state: IState) -> IState:
        pass

    @abstractmethod
    def run(self, state: IState) -> IState:
        pass

    @abstractmethod
    def get_energy(self, state: IState) -> float:
        pass

    @abstractmethod
    def prepare_for_timestep(self, alpha: float, timestep: int):
        pass


class ITemperatureScaler(ABC):
    """
    Interface for temperature scalers
    """

    @abstractmethod
    def __call__(self, alpha: float) -> float:
        pass
