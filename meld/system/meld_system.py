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

import numpy as np  # type: ignore
from typing import NamedTuple, List, Optional, Any, Set, Tuple, Dict


class _ExtraBondParam(NamedTuple):
    i: int
    j: int
    length: float
    force_constant: float


class _ExtraAngleParam(NamedTuple):
    i: int
    j: int
    k: int
    angle: float
    force_constant: float


class _ExtraTorsParam(NamedTuple):
    i: int
    j: int
    k: int
    l: int
    phase: float
    energy: float
    multiplicity: int


class System(interfaces.ISystem):
    """
    A class representing a MELD system.
    """

    _top_string: str
    _mdcrd_string: str
    restraints: restraints.RestraintManager
    _indexer: indexing.Indexer

    temperature_scaler: Optional[temperature.TemperatureScaler]
    _coordinates: np.ndarray
    _box_vectors: np.ndarray
    _n_atoms: int

    _atom_names: List[str]
    _residue_names: List[str]
    _residue_numbers: List[int]
    _atom_index: Any

    def __init__(self, top_string: str, mdcrd_string: str, indexer: indexing.Indexer):
        """
        Initialize a MELD system

        Args:
            top_string: topology of system from tleap
            mdcrd_string: coordinates of system from tleap
            indexer: an Indexer object to handle indexing
        """
        self._top_string = top_string
        self._mdcrd_string = mdcrd_string
        self.restraints = restraints.RestraintManager(self)
        self._indexer = indexer

        self.temperature_scaler = None
        self._setup_coords()

        self._setup_indexing()

        self.extra_bonds: List[_ExtraBondParam] = []
        self.extra_restricted_angles: List[_ExtraAngleParam] = []
        self.extra_torsions: List[_ExtraTorsParam] = []

    def atom_index(
        self,
        resid: int,
        atom_name: str,
        expected_resname: Optional[str] = None,
        chainid: Optional[int] = None,
        one_based: bool = False,
    ) -> indexing.AtomIndex:
        """
        Find the :class:`indexing.AtomIndex`

        The indexing can be either absolute (if `chainid` is `None`),
        or relative to a chain (if `chainid` is set).

        Both `resid` and `chainid` are one-based if `one_based` is `True`,
        or both are zero-based if `one_based=False` (the default).

        If `expected_resname` is specified, error checking will be performed to
        ensure that the returned atom has the expected residue name. Note
        that the residue names are those after processing with `tleap`,
        so some residue names may not match their value in an input pdb file.

        Args:
            resid: the residue index to lookup
            atom_name: the name of the atom to lookup
            expected_resname: the expected residue name, usually three all-caps characters,
                e.g. "ALA".
            chainid: the chain id to lookup
            one_based: use one-based indexing

        Returns:
            zero-based absolute atom index
        """
        return self._indexer.atom_index(
            resid, atom_name, expected_resname, chainid, one_based
        )

    def residue_index(
        self,
        resid: int,
        expected_resname: Optional[str] = None,
        chainid: Optional[int] = None,
        one_based: bool = False,
    ) -> indexing.ResidueIndex:
        """
        Find the :class:`indexing.ResidueIndex`

        The indexing can be either absolute (if `chainid` is `None`),
        or relative to a chain (if `chainid` is set).

        Both `resid` and `chainid` are one-based if `one_based` is `True`,
        or both are zero-based if `one_based=False` (the default).

        If `expected_resname` is specified, error checking will be performed to
        ensure that the returned atom has the expected residue name. Note
        that the residue names are those after processing with `tleap`,
        so some residue names may not match their value in an input pdb file.

        Args:
            resid: the residue index to lookup
            expected_resname: the expected residue name, usually three all-caps characters,
                e.g. "ALA".
            chainid: the chain id to lookup
            one_based: use one-based indexing

        Returns:
            zero-based absolute residue index
        """
        return self._indexer.residue_index(resid, expected_resname, chainid, one_based)

    @property
    def top_string(self) -> str:
        """
        tleap topology string for the system
        """
        return self._top_string

    @property
    def n_atoms(self) -> int:
        """
        number of atoms
        """
        return self._n_atoms

    @property
    def coordinates(self) -> np.ndarray:
        """
        coordinates of system
        """
        return self._coordinates

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
            _ExtraBondParam(
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
            _ExtraAngleParam(
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
            _ExtraTorsParam(
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
        reader = amber.ParmTopReader(self._top_string)

        self._atom_names = reader.get_atom_names()
        assert len(self._atom_names) == self._n_atoms

        self._residue_numbers = reader.get_residue_numbers()
        assert len(self._residue_numbers) == self._n_atoms

        self._residue_names = reader.get_residue_names()
        assert len(self._residue_names) == self._n_atoms

    def _setup_coords(self):
        reader = amber.CrdReader(self._mdcrd_string)
        self._coordinates = reader.get_coordinates()
        self._box_vectors = reader.get_box_vectors()
        self._n_atoms = self._coordinates.shape[0]


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
