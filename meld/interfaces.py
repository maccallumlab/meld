#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
This module implements a number of interfaces that help to minimize coupling
and eliminate circular imports
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Sequence

import numpy as np
import openmm as mm  # type: ignore
from openmm import app  # type: ignore

from meld.system import (
    density,
    indexing,
    mapping,
    param_sampling,
    pdb_writer,
    restraints,
    temperature,
)


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
    def distribute_alphas_to_workers(self, all_alphas: List[float]) -> List[float]:
        """
        Distribute alphas to workers

        Args:
            all_alphas: the alpha values to be distributed

        Returns:
            the block of alpha values for the leader
        """
        pass

    @abstractmethod
    def receive_alphas_from_leader(self) -> List[float]:
        """
        Receive a block of alphas from leader.

        Returns:
            the block of alpha values for this worker
        """
        pass

    @abstractmethod
    def distribute_states_to_workers(
        self, all_states: Sequence[IState]
    ) -> List[IState]:
        """
        Distribute a block of states to each worker.

        Args:
            all_states: states to be distributed

        Returns:
            the block of states to run on the leader node
        """
        pass

    @abstractmethod
    def receive_states_from_leader(self) -> List[IState]:
        """
        Get the block of states to run for this step

        Returns:
            the block of states to run for this step
        """
        pass

    @abstractmethod
    def gather_states_from_workers(self, state_on_leader: List[IState]) -> List[IState]:
        """
        Receive states from all workers

        Args:
            states_on_leader: the block of states on the leader after simulating
        Returns:
            A list of states, one from each replica.
        """
        pass

    @abstractmethod
    def send_states_to_leader(self, block: Sequence[IState]) -> None:
        """
        Send a block of states to the leader

        Args:
            block: block of states to send to the leader.
        """
        pass

    @abstractmethod
    def broadcast_all_states_to_workers(self, states: Sequence[IState]) -> None:
        """
        Broadcast all states to all workers.

        Args:
            states: a list of states
        """
        pass

    @abstractmethod
    def receive_all_states_from_leader(self) -> Sequence[IState]:
        """
        Receive all states from leader.

        Returns:
            a list of states to calculate the energy of
        """
        pass

    @abstractmethod
    def gather_energies_from_workers(
        self, energies_on_leader: np.ndarray
    ) -> np.ndarray:
        """
        Receive energies from each worker.

        Args:
            energies_on_leader: the energies from the leader

        Returns:
            a square matrix of every state on every replica to be used for replica exchange

        Note:
            Each row of the output matrix represents a different Hamiltonian. Each column
            represents a different state. Each worker will compute multiple rows of the
            output matrix.
        """
        pass

    @abstractmethod
    def send_energies_to_leader(self, energies: np.ndarray) -> None:
        """
        Send a block of energies to the leader.

        Args:
            energies: block of energies to send to the leader
        Note:
            Each row represents a different Hamiltonian. Each column represents a
            different state.
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

    @abstractmethod
    def distribute_thresholds_to_workers(
        self, all_thresholds: List[float]
    ) -> List[float]:
        """
        Distribute thresholds to workers

        Args:
            all_thresholds: the threshold values to be distributed

        Returns:
            the block of threshold values for the leader
        """
        pass

    @abstractmethod
    def receive_thresholds_from_leader(self) -> List[float]:
        """
        Receive a threshold from leader.

        Returns:
            the block of threshold values for this worker
        """
        pass

    @abstractmethod
    def gather_thresholds_from_workers(
        self, thresholds_on_leader: List[List[float]]
    ) -> List[List[float]]:
        """
        Receive threshold from each worker.

        Args:
            thresholds_on_leader: the threshold from the leader

        Returns:
            threshold
        """
        pass

    @abstractmethod
    def send_thresholds_to_leader(self, thresholds: List[List[float]]) -> None:
        """
        Send a block of thresholds to the leader.

        Args:
            thresholds: block of thresholds to send to the leader
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
    def n_workers(self) -> int:
        """number of workers"""
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
    group_energies: np.ndarray
    box_vector: np.ndarray
    parameters: param_sampling.ParameterState
    mappings: np.ndarray
    rdc_alignments: np.ndarray


class IRunner(ABC):
    """
    Interface for replica runners
    """

    temperature_scaler: Optional[temperature.TemperatureScaler]

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
    def get_group_energies(self, state: IState) -> np.ndarray:
        pass

    @abstractmethod
    def prepare_for_timestep(self, state: IState, alpha: float, timestep: int):
        pass


class ExtraBondParam(NamedTuple):
    i: int
    j: int
    length: float
    force_constant: float


class ExtraAngleParam(NamedTuple):
    i: int
    j: int
    k: int
    angle: float
    force_constant: float


class ExtraTorsParam(NamedTuple):
    i: int
    j: int
    k: int
    l: int
    phase: float
    energy: float
    multiplicity: int


class ISystem(ABC):
    """
    An interface for MELD systems
    """

    restraints: restraints.RestraintManager
    index: indexing.Indexer
    temperature_scaler: Optional[temperature.TemperatureScaler]
    param_sampler: param_sampling.ParameterManager
    mapper: mapping.PeakMapManager
    density: density.DensityManager
    builder_info: dict

    extra_bonds: List[ExtraBondParam]
    extra_restricted_angles: List[ExtraAngleParam]
    extra_torsions: List[ExtraTorsParam]

    @property
    @abstractmethod
    def num_alignments(self) -> int:
        """
        Number of rdc alignments
        """
        pass

    @property
    @abstractmethod
    def omm_system(self) -> mm.System:
        """
        Get the openmm system
        """
        pass

    @property
    @abstractmethod
    def topology(self) -> app.Topology:
        """
        Get the openmm topology
        """
        pass

    @property
    @abstractmethod
    def integrator(self) -> mm.LangevinIntegrator:
        """
        Get the integrator
        """
        pass

    @property
    @abstractmethod
    def barostat(self) -> mm.MonteCarloBarostat:
        """
        Get the barostat
        """
        pass

    @property
    @abstractmethod
    def solvation(self) -> str:
        """
        Get the solvation model
        """
        pass

    @property
    @abstractmethod
    def n_atoms(self) -> int:
        """
        number of atoms
        """
        pass

    @property
    @abstractmethod
    def template_coordinates(self) -> np.ndarray:
        """
        Get the template coordinates
        """
        pass

    @property
    @abstractmethod
    def template_velocities(self) -> np.ndarray:
        """
        Get the template velocities
        """
        pass

    @property
    @abstractmethod
    def template_box_vectors(self) -> Optional[np.ndarray]:
        """
        Get the template box vectors
        """
        pass

    @property
    @abstractmethod
    def atom_names(self) -> List[str]:
        """
        names for each atom
        """
        pass

    @property
    @abstractmethod
    def residue_numbers(self) -> List[int]:
        """
        residue numbers for each atom
        """
        pass

    @property
    @abstractmethod
    def residue_names(self) -> List[str]:
        """
        residue names for each atom
        """
        pass

    @abstractmethod
    def get_state_template(self) -> IState:
        """
        Get a template state for this system
        """
        pass

    @abstractmethod
    def get_pdb_writer(self) -> pdb_writer.PDBWriter:
        """
        Get the PDBWriter
        """
        pass

    @abstractmethod
    def add_extra_bond(
        self,
        i: indexing.AtomIndex,
        j: indexing.AtomIndex,
        length: float,
        force_constant: float,
    ) -> None:
        """
        Add an extra bond to the system

        Args:
            i: first atom in bond
            j: second atom in bond
            length: length of bond, in nm
            force_constant: strength of bond in kJ/mol/nm^2
        """
        pass

    @abstractmethod
    def add_extra_angle(
        self,
        i: indexing.AtomIndex,
        j: indexing.AtomIndex,
        k: indexing.AtomIndex,
        angle: float,
        force_constant: float,
    ) -> None:
        """
        Add an extra angle to the system

        Args:
            i: first atom in angle
            j: second atom in angle
            k: third atom in angle
            angle: equilibrium angle in degrees
            force_constant: strength of angle in kJ/mol/deg^2
        """
        pass

    @abstractmethod
    def add_extra_torsion(
        self,
        i: indexing.AtomIndex,
        j: indexing.AtomIndex,
        k: indexing.AtomIndex,
        l: indexing.AtomIndex,
        phase: float,
        energy: float,
        multiplicity: int,
    ) -> None:
        """
        Add an extra torsion to the system

        Args:
            i: first atom in torsion
            j: second atom in torsion
            k: third atom in torsion
            l: fourth atom in angle
            phase: phase angle in degrees
            energy: energy in kJ/mol
            multiplicity: periodicity of torsion
        """
        pass
