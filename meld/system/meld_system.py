#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
A module to define MELD Systems
"""

import openmm as mm  # type: ignore
from openmm import app

from meld import interfaces
from meld.vault import ENERGY_GROUPS
from meld.system import restraints
from meld.system import pdb_writer
from meld.system import indexing
from meld.system import temperature
from meld.system import param_sampling
from meld.system import state
from meld.system import mapping
from meld.system import density

import numpy as np  # type: ignore
from typing import List, Optional, Any


class System(interfaces.ISystem):
    """
    A class representing a MELD system.
    """

    restraints: restraints.RestraintManager
    """The object to handle managing restraints for the system"""

    density: density.DensityManager
    """The object to handle managing densities for the system"""

    index: indexing.Indexer
    """The object to lookup atom and residue indices """

    temperature_scaler: Optional[temperature.TemperatureScaler]
    """The temperature scaler for the system"""

    param_sampler: param_sampling.ParameterManager
    """The sampler for parameters"""

    mapper: mapping.PeakMapManager
    """The peak map manager"""

    _solvation: str
    _openmm_system: mm.System
    _openmm_topology: app.Topology
    _integrator: mm.LangevinIntegrator
    _barostat: Optional[mm.MonteCarloBarostat]
    _template_coordinates: np.ndarray
    _template_velocities: np.ndarray
    _template_box_vectors: Optional[np.ndarray]
    _n_atoms: int

    _atom_names: List[str]
    _residue_names: List[str]
    _residue_numbers: List[int]

    def __init__(
        self,
        solvation: str,
        openmm_system: mm.System,
        openmm_topology: app.Topology,
        integrator: mm.LangevinIntegrator,
        barostat: Optional[mm.MonteCarloBarostat],
        template_coordinates: np.ndarray,
        template_velocities: np.ndarray,
        template_box_vectors: Optional[np.ndarray],
        builder_info: dict,
    ):
        """
        Initialize a MELD system

        Args:
            solvation: the solvation model to use
            openmm_system: the OpenMM system
            openmm_topology: the OpenMM topology
            integrator: the OpenMM integrator
            barostat: the OpenMM barostat
            template_coordinates: the coordinates of the template
            template_velocities: the velocities of the template
            template_box_vectors: the box vectors of the template
            builder_info: a dictionary of extra information from the builder/patchers
        """
        self._solvation = solvation
        self._openmm_system = openmm_system
        self._openmm_topology = openmm_topology
        self._integrator = integrator
        self._barostat = barostat
        self._template_coordinates = template_coordinates
        self._n_atoms = self._template_coordinates.shape[0]
        self._template_velocities = template_velocities
        self._template_box_vectors = template_box_vectors
        self.builder_info = builder_info
        self.restraints = restraints.RestraintManager(self)
        self.density = density.DensityManager()
        self.param_sampler = param_sampling.ParameterManager()
        self.mapper = mapping.PeakMapManager()

        self.extra_bonds = []
        self.extra_restricted_angles = []
        self.extra_torsions = []

        self.temperature_scaler = None

        self._setup_indexing()

    @property
    def num_alignments(self) -> int:
        return self.builder_info.get("num_alignments", 0)

    @property
    def solvation(self):
        return self._solvation

    @property
    def omm_system(self):
        return self._openmm_system

    @property
    def topology(self):
        return self._openmm_topology

    @property
    def integrator(self):
        return self._integrator

    @property
    def barostat(self):
        return self._barostat

    @property
    def n_atoms(self) -> int:
        """
        number of atoms
        """
        return self._n_atoms

    @property
    def template_coordinates(self) -> np.ndarray:
        """
        Get the template coordinates
        """
        return self._template_coordinates

    @property
    def template_velocities(self) -> np.ndarray:
        """
        Get the template velocities
        """
        return self._template_velocities

    @property
    def template_box_vectors(self) -> Optional[np.ndarray]:
        """
        Get the template box vectors
        """
        return self._template_box_vectors

    @property
    def atom_names(self) -> List[str]:
        """
        names for each atom
        """
        return self._atom_names

    @property
    def residue_numbers(self) -> List[int]:
        """
        residue numbers for each atom
        """
        return self._residue_numbers

    @property
    def residue_names(self) -> List[str]:
        """
        residue names for each atom
        """
        return self._residue_names

    def get_state_template(self) -> state.SystemState:
        """
        Get a template SystemState.
        """
        pos = self._template_coordinates.copy()
        vel = self._template_velocities.copy()
        alpha = 0.0
        energy = 0.0
        group_energies = np.zeros(ENERGY_GROUPS)

        box_vectors = self._template_box_vectors
        if box_vectors is None:
            box_vectors = np.array([0.0, 0.0, 0.0])

        params = self.param_sampler.get_initial_state()
        mappings = self.mapper.get_initial_state()

        if self.num_alignments == 0:
            alignments = None
        else:
            alignments = np.zeros(self.num_alignments * 5)

        return state.SystemState(
            pos,
            vel,
            alpha,
            energy,
            group_energies,
            box_vectors,
            params,
            mappings,
            alignments,
        )

    def get_pdb_writer(self) -> pdb_writer.PDBWriter:
        """
        Get the PDBWriter
        """
        return pdb_writer.PDBWriter(
            list(range(1, len(self._atom_names) + 1)),
            self._atom_names,
            self._residue_numbers,
            self._residue_names,
        )

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
        assert isinstance(i, indexing.AtomIndex)
        assert isinstance(j, indexing.AtomIndex)
        self.extra_bonds.append(
            interfaces.ExtraBondParam(
                i=int(i), j=int(j), length=length, force_constant=force_constant
            )
        )

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
        assert isinstance(i, indexing.AtomIndex)
        assert isinstance(j, indexing.AtomIndex)
        assert isinstance(k, indexing.AtomIndex)
        self.extra_restricted_angles.append(
            interfaces.ExtraAngleParam(
                i=int(i), j=int(j), k=int(k), angle=angle, force_constant=force_constant
            )
        )

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
        assert isinstance(i, indexing.AtomIndex)
        assert isinstance(j, indexing.AtomIndex)
        assert isinstance(k, indexing.AtomIndex)
        assert isinstance(l, indexing.AtomIndex)
        self.extra_torsions.append(
            interfaces.ExtraTorsParam(
                i=int(i),
                j=int(j),
                k=int(k),
                l=int(l),
                phase=phase,
                energy=energy,
                multiplicity=multiplicity,
            )
        )

    def _setup_indexing(self):
        self.index = indexing.setup_indexing(self._openmm_topology)

        self._atom_names = [atom.name for atom in self._openmm_topology.atoms()]
        assert len(self._atom_names) == self._n_atoms

        self._residue_numbers = [
            atom.residue.index for atom in self._openmm_topology.atoms()
        ]
        assert len(self._residue_numbers) == self._n_atoms

        self._residue_names = [
            atom.residue.name for atom in self._openmm_topology.atoms()
        ]
        assert len(self._residue_names) == self._n_atoms
