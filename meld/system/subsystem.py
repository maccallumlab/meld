#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to build SubSystems from sequence or PDB file
"""

from meld.system import indexing
import parmed  # type: ignore

import numpy as np  # type: ignore
import math
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import NamedTuple, List


class _SubSystem(ABC):
    """
    Base class for other SubSystem classes.

    Provides functionality for translation/rotation and adding H-bonds.
    """

    def __init__(self):
        self._translation_vector = np.zeros(3)
        self._rotatation_matrix = np.eye(3)
        self._disulfide_list = []
        self._general_bond = []
        self._prep_files = []
        self._frcmod_files = []
        self._lib_files = []
        self._info = []

    @abstractmethod
    def prepare_for_tleap(self, mol_id: str):
        """
        Prepare any inputs needed for tleap

        Args:
            mol_id: identifier for this moleule

        This runs in a temporary directory where tleap
        will be run.
        """
        pass

    @abstractmethod
    def generate_tleap_input(self, mol_id: str) -> List[str]:
        """
        Returns a list of tleap commands to run.

        Args:
            mol_id: identifier for this moleule

        Returns:
            a list of telap commands
        """
        pass

    def set_translation(self, translation_vector: np.ndarray):
        """
        Set the translation vector.

        Args:
            translation_vector: in nanometers

        .. note::
           Translation happens after rotation.
        """
        self._translation_vector = np.array(translation_vector)

    def set_rotation(self, rotation_axis: np.ndarray, theta: float):
        """
        Set the rotation.

        Args:
            rotation_axis: in nanometers
            theta: angle of rotation in degrees

        .. note::
           Rotation happens after translation.
        """
        theta = theta * 180 / math.pi
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        a = np.cos(theta / 2.0)
        b, c, d = -rotation_axis * np.sin(theta / 2.0)
        self._rotatation_matrix = np.array(
            [
                [
                    a * a + b * b - c * c - d * d,
                    2 * (b * c - a * d),
                    2 * (b * d + a * c),
                ],
                [
                    2 * (b * c + a * d),
                    a * a + c * c - b * b - d * d,
                    2 * (c * d - a * b),
                ],
                [
                    2 * (b * d - a * c),
                    2 * (c * d + a * b),
                    a * a + d * d - b * b - c * c,
                ],
            ]
        )

    def add_bond(
        self,
        res_index_i: indexing.ResidueIndex,
        res_index_j: indexing.ResidueIndex,
        atom_name_i: str,
        atom_name_j: str,
        bond_type: str,
    ):
        """
        Add a general bond.

        Args:
            res_index_i: index of residue i
            res_index_j: index of residue j
            atom_name_i: name of i
            atom_name_j: name of j
            bond_type:   type of bond ["S", "D", "T"...]
        """
        assert isinstance(res_index_i, indexing.ResidueIndex)
        assert isinstance(res_index_j, indexing.ResidueIndex)
        self._general_bond.append(
            (int(res_index_i), int(res_index_j), atom_name_i, atom_name_j, bond_type)
        )

    def add_disulfide(self, res_index_i, res_index_j):
        """
        Add a disulfide bond.

        Args:
            res_index_i: index of residue i
            res_index_j: index of residue j
        """
        assert isinstance(res_index_i, indexing.ResidueIndex)
        assert isinstance(res_index_j, indexing.ResidueIndex)
        self._disulfide_list.append((int(res_index_i), int(res_index_j)))

    def add_prep_file(self, fname: str):
        """
        Add a prep file.

        This will be needed when using residues that
        are not defined in the general amber force field

        Args:
            fname: filename of prep file
        """
        self._prep_files.append(fname)

    def add_frcmod_file(self, fname: str):
        """
        Add a frcmod file.

        This will be needed when using residues that
        are not defined in the general amber force field

        Args:
            fname: name of frcmod file
        """
        self._frcmod_files.append(fname)

    def add_lib_file(self, fname: str):
        """
        Add a lib file.

        This will be needed when using residues that
        are not defined in the general amber force field

        Args:
            fname: name of lib file
        """
        self._lib_files.append(fname)

    def _gen_translation_string(self, mol_id):
        return """translate {mol_id} {{ {x} {y} {z} }}""".format(
            mol_id=mol_id,
            x=self._translation_vector[0],
            y=self._translation_vector[1],
            z=self._translation_vector[2],
        )

    def _gen_rotation_string(self, mol_id):
        return ""

    def _gen_bond_string(self, mol_id):
        bond_strings = []
        for i, j, a, b, t in self._general_bond:
            d = f'bond {mol_id}.{i+1}.{a} {mol_id}.{j+1}.{b} "{t}"'
            bond_strings.append(d)
        return bond_strings

    def _gen_disulfide_string(self, mol_id):
        disulfide_strings = []
        for i, j in self._disulfide_list:
            d = f"bond {mol_id}.{i+1}.SG {mol_id}.{j+1}.SG"
            disulfide_strings.append(d)
        return disulfide_strings

    def _gen_read_prep_string(self):
        prep_string = []
        for p in self._prep_files:
            prep_string.append(f"loadAmberPrep {p}")
        return prep_string

    def _gen_read_frcmod_string(self):
        frcmod_string = []
        for p in self._frcmod_files:
            frcmod_string.append(f"loadAmberParams {p}")
        return frcmod_string

    def _gen_read_lib_string(self):
        lib_string = []
        for p in self._lib_files:
            lib_string.append(f"loadoff {p}")
        return lib_string


class SubSystemFromSequence(_SubSystem):
    """
    Class to create a sub-system from sequence.

    This class will create a sub-system from sequence. This class is
    pretty dumb and relies on AmberTools to do all of the heavy lifting.

    The sequence is specified in Amber/Leap format. There are special NRES and
    CRES variants for the N- and C-termini. Different protonation states are
    also available via different residue names. E.g. ASH
    for neutral ASP.
    """

    def __init__(self, sequence: str):
        """
        Initialize a SubSystemFromSequence

        Args:
            sequence: the sequence to build
        """
        super(SubSystemFromSequence, self).__init__()
        self._sequence = sequence
        sequence_len = len(sequence.split(" "))
        chain_info = indexing._ChainInfo({i: i for i in range(sequence_len)})
        self._info = indexing._SubSystemInfo(sequence_len, [chain_info])

    def prepare_for_tleap(self, mol_id):
        # we don't need to do anything
        pass

    def generate_tleap_input(self, mol_id):
        leap_cmds = []
        leap_cmds.append("source leaprc.gaff")
        leap_cmds.extend(self._gen_read_frcmod_string())
        leap_cmds.extend(self._gen_read_prep_string())
        leap_cmds.extend(self._gen_read_lib_string())
        leap_cmds.append(f"{mol_id} = sequence {{ {self._sequence} }}")
        leap_cmds.extend(self._gen_disulfide_string(mol_id))
        leap_cmds.extend(self._gen_bond_string(mol_id))
        leap_cmds.append(self._gen_rotation_string(mol_id))
        leap_cmds.append(self._gen_translation_string(mol_id))
        return leap_cmds


class SubSystemFromPdbFile(_SubSystem):
    """
    Create a new susbsystem from a pdb file.

    This class is dumb and relies on AmberTools for the heavy lifting.

    :param pdb_path: string path to the pdb file

    .. note::
        no processing happens to this pdb file. It must be understandable by
        tleap and atoms/residues may need to be added/deleted/renamed. These
        manipulations should happen to the file before MELD is invoked.

    """

    def __init__(self, pdb_path: str):
        """
        Initialize a SubSystemFromPdbFile

        Args:
            pdb_path: path to pdb file
        """
        super(SubSystemFromPdbFile, self).__init__()
        with open(pdb_path) as pdb_file:
            self._pdb_contents = pdb_file.read()

        # figure out chains
        pdb = parmed.load_file(pdb_path)
        n_residues = len(pdb.residues)

        # get list of chainids
        chainids = []
        chain_to_res = defaultdict(list)
        for i, residue in enumerate(pdb.residues):
            chainids.append(residue.chain)
            chain_to_res[residue.chain].append(i)
        chainid_set = set(chainids)

        # loop over the chainids in alphabetical order
        chains = []
        for chainid in sorted(chainid_set):
            chain = indexing._ChainInfo({i: j for i, j in enumerate(chain_to_res[chainid])})
            chains.append(chain)
        self._info = indexing._SubSystemInfo(n_residues, chains)

    def prepare_for_tleap(self, mol_id):
        # copy the contents of the pdb file into the current working directory
        pdb_path = f"{mol_id}.pdb"
        with open(pdb_path, "w") as pdb_file:
            pdb_file.write(self._pdb_contents)

    def generate_tleap_input(self, mol_id):
        leap_cmds = []
        leap_cmds.append("source leaprc.gaff")
        leap_cmds.extend(self._gen_read_frcmod_string())
        leap_cmds.extend(self._gen_read_prep_string())
        leap_cmds.extend(self._gen_read_lib_string())
        leap_cmds.append(f"{mol_id} = loadPdb {mol_id}.pdb")
        leap_cmds.extend(self._gen_bond_string(mol_id))
        leap_cmds.extend(self._gen_disulfide_string(mol_id))
        leap_cmds.append(self._gen_rotation_string(mol_id))
        leap_cmds.append(self._gen_translation_string(mol_id))
        return leap_cmds
