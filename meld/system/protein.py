#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import numpy as np  # type: ignore
import math
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import NamedTuple
from .indexing import ChainInfo, SubSystemInfo
import parmed  # type: ignore


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
    def prepare_for_tleap(self, mol_id):
        """
        Prepare any inputs needed for tleap

        This runs in a temporary directory where tleap
        will be run.
        """
        pass

    @abstractmethod
    def generate_tleap_input(self, mol_id):
        """
        Returns a list of tleap commands to run.
        """
        pass

    def set_translation(self, translation_vector):
        """
        Set the translation vector.

        :param translation_vector: ``numpy.array(3)`` in nanometers

        Translation happens after rotation.

        """
        self._translation_vector = np.array(translation_vector)

    def set_rotation(self, rotation_axis, theta):
        """
        Set the rotation.

        :param rotation_axis: ``numpy.array(3)`` in nanometers
        :param theta: angle of rotation in degrees

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

    def add_bond(self, res_index_i, res_index_j, atom_name_i, atom_name_j, bond_type):
        """
        Add a general bond.

        :param res_index_i: zero-based index of residue i
        :param res_index_j: zero-based index of residue j
        :param atom_name_i: string name of i
        :param atom_name_j: string name of j
        :param bond_type:   string specifying the "S", "D","T"... bond

        .. note::
            indexing starts from zero and the residue numbering from the
            PDB file is ignored.

        """
        self._general_bond.append(
            (res_index_i, res_index_j, atom_name_i, atom_name_j, bond_type)
        )

    def add_disulfide(self, res_index_i, res_index_j):
        """
        Add a disulfide bond.

        :param res_index_i: zero-based index of residue i
        :param res_index_j: zero-based index of residue j

        .. note::
            indexing starts from zero and the residue numbering from the
            PDB file is ignored. When loading from a PDB or creating a
            sequence, residue name must be CYX, not CYS.

        """
        self._disulfide_list.append((res_index_i, res_index_j))

    def add_prep_file(self, fname):
        """
        Add a prep file.
        This will be needed when using residues that
        are not defined in the general amber force field
        """
        self._prep_files.append(fname)

    def add_frcmod_file(self, fname):
        """
        Add a frcmod file.
        This will be needed when using residues that
        are not defined in the general amber force field
        """
        self._frcmod_files.append(fname)

    def add_lib_file(self, fname):
        """
        Add a lib file.
        This will be needed when using residues that
        are not defined in the general amber force field
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

    :param sequence: sequence create

    The sequence is specified in Amber/Leap format. There are special NRES and
    CRES variants for the N- and C-termini. Different protonation states are
    also available via different residue names. E.g. ASH
    for neutral ASP.

    """

    def __init__(self, sequence):
        super(SubSystemFromSequence, self).__init__()
        self._sequence = sequence
        sequence_len = len(sequence.split(" "))
        chain_info = ChainInfo({i: i for i in range(sequence_len)})
        self._info = SubSystemInfo(sequence_len, [chain_info])

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

    def __init__(self, pdb_path):
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
        chainids = set(chainids)

        # loop over the chainids in alphabetical order
        chains = []
        for chainid in sorted(chainids):
            chain = ChainInfo({i: j for i, j in enumerate(chain_to_res[chainid])})
            chains.append(chain)
        self._info = SubSystemInfo(n_residues, chains)

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
