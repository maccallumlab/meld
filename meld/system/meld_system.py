#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
A module to define MELD Systems
"""

from meld import interfaces
from meld.system import amber
from meld.system import restraints
from meld.system import pdb_writer
from meld.system import indexing
from meld.system import temperature
from meld.system import param_sampling
from meld.system import state
from meld.system import mapping

import numpy as np  # type: ignore
from typing import List, Optional, Any


class System(interfaces.ISystem):
    """
    A class representing a MELD system.
    """

    restraints: restraints.RestraintManager
    """ The object to handle managing restraints for the system """

    index: indexing.Indexer
    """The object to lookup atom and residue indices """

    temperature_scaler: Optional[temperature.TemperatureScaler]
    """The temperature scaler for the system"""

    param_sampler: param_sampling.ParameterManager
    """The sampler for parameters"""

    _coordinates: np.ndarray
    _box_vectors: np.ndarray
    _n_atoms: int

    _atom_names: List[str]
    _residue_names: List[str]
    _residue_numbers: List[int]
    _atom_index: Any

    def __init__(self, openmm_system, openmm_topology, integrator, barostat, init_coordinates):
        """
        Initialize a MELD system

        Args:
            top_string: topology of system from tleap
            mdcrd_string: coordinates of system from tleap
            indexer: an Indexer object to handle indexing
        """
        self._openmm_system = openmm_system
        self._openmm_topology = openmm_topology
        self._integrator = integrator
        self._barostat = barostat
        self._init_coordinates = init_coordinates
        self.restraints = restraints.RestraintManager(self)
        self.param_sampler = param_sampling.ParameterManager()
        self.mapper = mapping.PeakMapManager()

        self.extra_bonds = []
        self.extra_restricted_angles = []
        self.extra_torsions = []

        self.temperature_scaler = None
        self._setup_coords()

        # TODO: need to re-write indexing to use openmm topology
        self._setup_indexing()

    @property
    def n_atoms(self) -> int:
        """
        number of atoms
        """
        # TODO: need to set this up based on openmm_topology
        return self._n_atoms

    @property
    def init_coordinates(self) -> np.ndarray:
        """
        initial coordinates of system
        """
        return self._init_coordinates

    @property
    def atom_names(self) -> List[str]:
        """
        names for each atom
        """
        # TODO: this needs to be setup
        return self._atom_names

    @property
    def residue_numbers(self) -> List[int]:
        """
        residue numbers for each atom
        """
        # TODO: this needs to be setup
        return self._residue_numbers

    @property
    def residue_names(self) -> List[str]:
        """
        residue names for each atom
        """
        # TODO: this needs to be setup
        return self._residue_names

    def get_state_template(self):
        pos = self._init_coordinates.copy()
        vel = np.zeros_like(pos)
        alpha = 0.0
        energy = 0.0
        box_vectors = self._box_vectors
        if box_vectors is None:
            box_vectors = np.array([0.0, 0.0, 0.0])
        params = self.param_sampler.get_initial_state()
        mappings = self.mapper.get_initial_state()
        return state.SystemState(pos, vel, alpha, energy, box_vectors, params, mappings)

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
        # TODO: This needs to be re-written
        reader = amber.ParmTopReader(self._top_string)

        self._atom_names = reader.get_atom_names()
        assert len(self._atom_names) == self._n_atoms

        self._residue_numbers = reader.get_residue_numbers()
        assert len(self._residue_numbers) == self._n_atoms

        self._residue_names = reader.get_residue_names()
        assert len(self._residue_names) == self._n_atoms

    def _setup_coords(self):
        # TODO: This needs to be re-written
        reader = amber.CrdReader(self._mdcrd_string)
        self._coordinates = reader.get_coordinates()
        self._box_vectors = reader.get_box_vectors()
        self._n_atoms = self._coordinates.shape[0]

# TODO: This needs to be re-written
# The patchers will work on OpenMM system, topology, coords, and box vectors.
# Each patcher will run in sequence, modifying things before everything
# is evenentually sent to the MELD System constructor.
def _load_amber_system(top_filename, crd_filename, chains, patchers=None):
    # Load in top and crd files output by leap
    with open(top_filename, "rt") as topfile:
        top = topfile.read()
    with open(crd_filename) as crdfile:
        crd = crdfile.read()

    # Allow patchers to modify top and crd strings
    if patchers is None:
        patchers = []
    for patcher in patchers:
        top, crd = patcher.patch(top, crd)

    # Setup indexing
    indexer = indexing._setup_indexing(
        chains, amber.ParmTopReader(top), amber.CrdReader(crd)
    )

    # Create the system
    system = System(top, crd, indexer)

    # Allow the patchers to modify the system
    for patcher in patchers:
        patcher.finalize(system)

    return system
