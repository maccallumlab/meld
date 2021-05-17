#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
A module to define MELD Systems
"""

from .restraints import RestraintManager
from ..pdb_writer import PDBWriter
from .indexing import _setup_indexing, AtomIndex, ResidueIndex, Indexer
from .temperature import TemperatureScaler
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


class System:
    """
    A class representing a MELD system.
    """

    _top_string: str
    _mdcrd_string: str
    restraints: RestraintManager
    _indexer: Indexer

    temperature_scaler: Optional[TemperatureScaler]
    _coordinates: np.ndarray
    _box_vectors: np.ndarray
    _n_atoms: int

    _atom_names: List[str]
    _residue_names: List[str]
    _residue_numbers: List[int]
    _atom_index: Any

    def __init__(self, top_string: str, mdcrd_string: str, indexer: Indexer):
        """
        Initialize a MELD system

        Args:
            top_string: topology of system from tleap
            mdcrd_string: coordinates of system from tleap
            indexer: an Indexer object to handle indexing
        """
        self._top_string = top_string
        self._mdcrd_string = mdcrd_string
        self.restraints = RestraintManager(self)
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
    ) -> AtomIndex:
        """
        Find the :class:`AtomIndex`

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
    ) -> ResidueIndex:
        """
        Find the :class:`ResidueIndex`

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

    def get_pdb_writer(self) -> PDBWriter:
        """
        Get the PDBWriter
        """
        return PDBWriter(
            list(range(1, len(self._atom_names) + 1)),
            self._atom_names,
            self._residue_numbers,
            self._residue_names,
        )

    def add_extra_bond(
        self, i: AtomIndex, j: AtomIndex, length: float, force_constant: float
    ) -> None:
        """
        Add an extra bond to the system

        Args:
            i: first atom in bond
            j: second atom in bond
            length: length of bond, in nm
            force_constant: strength of bond in kJ/mol/nm^2
        """
        assert isinstance(i, AtomIndex)
        assert isinstance(j, AtomIndex)
        self.extra_bonds.append(
            _ExtraBondParam(
                i=int(i), j=int(j), length=length, force_constant=force_constant
            )
        )

    def add_extra_angle(
        self,
        i: AtomIndex,
        j: AtomIndex,
        k: AtomIndex,
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
        assert isinstance(i, AtomIndex)
        assert isinstance(j, AtomIndex)
        assert isinstance(k, AtomIndex)
        self.extra_restricted_angles.append(
            _ExtraAngleParam(
                i=int(i), j=int(j), k=int(k), angle=angle, force_constant=force_constant
            )
        )

    def add_extra_torsion(
        self,
        i: AtomIndex,
        j: AtomIndex,
        k: AtomIndex,
        l: AtomIndex,
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
        assert isinstance(i, AtomIndex)
        assert isinstance(j, AtomIndex)
        assert isinstance(k, AtomIndex)
        assert isinstance(l, AtomIndex)
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
        reader = ParmTopReader(self._top_string)

        self._atom_names = reader.get_atom_names()
        assert len(self._atom_names) == self._n_atoms

        self._residue_numbers = reader.get_residue_numbers()
        assert len(self._residue_numbers) == self._n_atoms

        self._residue_names = reader.get_residue_names()
        assert len(self._residue_names) == self._n_atoms

    def _setup_coords(self):
        reader = CrdReader(self._mdcrd_string)
        self._coordinates = reader.get_coordinates()
        self._box_vectors = reader.get_box_vectors()
        self._n_atoms = self._coordinates.shape[0]


class CrdReader:
    """
    Class to read in coordinates from mdcrd string
    """

    _coords: np.ndarray
    _box_vectors: np.ndarray

    def __init__(self, crd_string: str):
        """
        Initialize CrdReader

        Args:
            crd_string: mdcrd string from tleap
        """
        self.crd_string = crd_string
        self._read()

    def get_coordinates(self) -> np.ndarray:
        """
        Get the coordiantes

        Returns:
            coordinates, shape(n_atoms, 3)
        """
        return self._coords

    def get_box_vectors(self) -> np.ndarray:
        """
        Get the box_vectors

        Returns:
            box vectors, shape(3, 3)
        """
        return self._box_vectors

    def _read(self):
        def split_len(seq, length):
            return [seq[i : i + length] for i in range(0, len(seq), length)]

        lines = self.crd_string.splitlines()
        n_atoms = int(lines[1].split()[0])
        coords = []
        box_vectors = None

        for line in lines[2:]:
            cols = split_len(line, 12)
            cols = [float(c) for c in cols]
            coords.extend(cols)

        # check for box vectors
        if len(coords) == 3 * n_atoms + 6:
            coords, box_vectors = coords[:-6], coords[-6:]
            for bv in box_vectors[-3:]:
                if not bv == 90.0:
                    raise RuntimeError("box angle != 90.0 degrees")
            box_vectors = np.array(box_vectors[:-3])
        elif not len(coords) == 3 * n_atoms:
            raise RuntimeError("len(coords) != 3 * n_atoms")

        coords = np.array(coords)
        coords = coords.reshape((n_atoms, 3))
        self._coords = coords
        self._box_vectors = box_vectors


class ParmTopReader:
    """
    Read in information from parmtop file
    """

    def __init__(self, top_string: str):
        """
        Initialize ParmTopReader

        Args:
            top_string: topology string from tleap
        """
        self._top_string = top_string

    def get_atom_names(self) -> List[str]:
        """
        Get the atom names

        Returns:
            the name for each atom
        """
        return self._get_parameter_block("%FLAG ATOM_NAME", chunksize=4)

    def get_residue_names(self) -> List[str]:
        """
        Get the residue names

        Returns:
            the residue name for each atom
        """
        res_names = self._get_parameter_block("%FLAG RESIDUE_LABEL", chunksize=4)
        res_numbers = self.get_residue_numbers()
        return [res_names[i - 1] for i in res_numbers]

    def get_residue_numbers(self) -> List[int]:
        """
        Get the residue numbers

        Returns:
            the residue number for each atom
        """
        n_atoms = int(self._get_parameter_block("%FLAG POINTERS", chunksize=8)[0])
        res_pointers_str = self._get_parameter_block(
            "%FLAG RESIDUE_POINTER", chunksize=8
        )
        res_pointers = [int(p) for p in res_pointers_str]
        res_pointers.append(n_atoms + 1)
        residue_numbers: List[int] = []
        for res_number, (start, end) in enumerate(
            zip(res_pointers[:-1], res_pointers[1:])
        ):
            residue_numbers.extend([res_number + 1] * (int(end) - int(start)))
        return residue_numbers

    def _get_parameter_block(self, flag: str, chunksize: int) -> List[str]:
        lines = self._top_string.splitlines()

        # find the line with our flag
        index_start = [i for (i, line) in enumerate(lines) if line.startswith(flag)][
            0
        ] + 2

        # find the index of the next flag
        index_end = [
            i for (i, line) in enumerate(lines[index_start:]) if line and line[0] == "%"
        ][0] + index_start

        # do something useful with the data
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i : i + n]

        data = []
        for line in lines[index_start:index_end]:
            for chunk in chunks(line, chunksize):
                data.append(chunk.strip())
        return data

    def get_bonds(self) -> Set[Tuple[int, int]]:
        """
        Get the bonded atoms from topology

        Returns:
            The set of bonds from the topology

        .. note::
           the indices are zero-based
        """
        # the amber bonds section contains a triple of integers for each bond:
        # i, j, type_index. We need i, j, but will end up ignoring type_index
        bond_item_str = self._get_parameter_block(
            "%FLAG BONDS_WITHOUT_HYDROGEN", chunksize=8
        )
        bond_item_str += self._get_parameter_block(
            "%FLAG BONDS_INC_HYDROGEN", chunksize=8
        )
        # the bonds section of the amber file is indexed by coordinate
        # to get the atom index we divide by three and add one
        bond_items = [int(item) // 3 + 1 for item in bond_item_str]

        bonds = set()
        # take the items 3 at a time, ignoring the type_index
        for i, j, _ in zip(bond_items[::3], bond_items[1::3], bond_items[2::3]):
            # Add both orders to make life easy for callers.
            # Amber is 1-based, but we are 0-based.
            bonds.add((i - 1, j - 1))
            bonds.add((j - 1, i - 1))
        return bonds

    def get_atom_map(self) -> Dict[Tuple[int, str], int]:
        """
        Get the atom map

        Returns:
            the mapping from (resid, atom_name) to atom_index

        .. note::
           both resid and atom_index are zero-based
        """
        residue_numbers = [r - 1 for r in self.get_residue_numbers()]
        atom_names = self.get_atom_names()
        atom_numbers = range(len(atom_names))
        return {
            (res_num, atom_name): atom_index
            for res_num, atom_name, atom_index in zip(
                residue_numbers, atom_names, atom_numbers
            )
        }


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
    indexer = _setup_indexing(chains, ParmTopReader(top), CrdReader(crd))

    # Create the system
    system = System(top, crd, indexer)

    # Allow the patchers to modify the system
    for patcher in patchers:
        patcher.finalize(system)

    return system
