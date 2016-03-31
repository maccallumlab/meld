#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import numpy as np
import math


class ProteinBase(object):
    '''
    Base class for other Protein classes.

    Provides functionality for translation/rotation and adding H-bonds.

    '''
    def __init__(self):
        self._translation_vector = np.zeros(3)
        self._rotatation_matrix = np.eye(3)
        self._disulfide_list = []
        self._general_bond = []
        self._prep_files = []
        self._frcmod_files = []
        self._lib_files = []

    def set_translation(self, translation_vector):
        '''
        Set the translation vector.

        :param translation_vector: ``numpy.array(3)`` in nanometers

        Translation happens after rotation.

        '''
        self._translation_vector = np.array(translation_vector)

    def set_rotation(self, rotation_axis, theta):
        '''
        Set the rotation.

        :param rotation_axis: ``numpy.array(3)`` in nanometers
        :param theta: angle of rotation in degrees

        Rotation happens after translation.

        '''
        theta = theta * 180 / math.pi
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        a = np.cos(theta / 2.)
        b, c, d = -rotation_axis * np.sin(theta / 2.)
        self._rotatation_matrix = np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                                           [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                                           [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

    def add_bond(self, res_index_i, res_index_j, atom_name_i, atom_name_j, bond_type):
        '''
        Add a general bond.

        :param res_index_i: one-based index of residue i
        :param res_index_j: one-based index of residue j
        :param atom_name_i: string name of i
        :param atom_name_j: string name of j 
        :param bond_type:   string specifying the "S", "D","T"... bond

        .. note::
            indexing starts from one and the residue numbering from the PDB file is ignored. 

        '''
        self._general_bond.append((res_index_i, res_index_j,atom_name_i,atom_name_j,bond_type))

    def add_disulfide(self, res_index_i, res_index_j):
        '''
        Add a disulfide bond.

        :param res_index_i: one-based index of residue i
        :param res_index_j: one-based index of residue j

        .. note::
            indexing starts from one and the residue numbering from the PDB file is ignored. When loading
            from a PDB or creating a sequence, residue name must be CYX, not CYS.

        '''
        self._disulfide_list.append((res_index_i, res_index_j))

    def add_prep_file(self,fname):
        '''
        Add a prep file.
        This will be needed when using residues that
        are not defined in the general amber force field
        '''
        self._prep_files.append(fname)

    def add_frcmod_file(self,fname):
        '''
        Add a frcmod file.
        This will be needed when using residues that
        are not defined in the general amber force field
        '''
        self._frcmod_files.append(fname)

    def add_lib_file(self,fname):
        '''
        Add a lib file.
        This will be needed when using residues that
        are not defined in the general amber force field
        '''
        self._lib_files.append(fname)

    def _gen_translation_string(self, mol_id):
        return '''translate {mol_id} {{ {x} {y} {z} }}'''.format(mol_id=mol_id,
                                                                 x=self._translation_vector[0],
                                                                 y=self._translation_vector[1],
                                                                 z=self._translation_vector[2])

    def _gen_rotation_string(self, mol_id):
        return ''

    def _gen_bond_string(self,mol_id):
        bond_strings = []
        for i,j,a,b,t in self._general_bond:
            d = 'bond {mol_id}.{i}.{a} {mol_id}.{j}.{b} "{t}"'.format(mol_id=mol_id, i=i, j=j, a=a, b=b, t=t)
            bond_strings.append(d)
        return bond_strings

    def _gen_disulfide_string(self, mol_id):
        disulfide_strings = []
        for i, j in self._disulfide_list:
            d = 'bond {mol_id}.{i}.SG {mol_id}.{j}.SG'.format(mol_id=mol_id, i=i, j=j)
            disulfide_strings.append(d)
        return disulfide_strings

    def _gen_read_prep_string(self):
        prep_string = []
        for p in self._prep_files:
            prep_string.append('loadAmberPrep {}'.format(p))
        return prep_string

    def _gen_read_frcmod_string(self):
        frcmod_string = []
        for p in self._frcmod_files:
            frcmod_string.append('loadAmberParams {}'.format(p))
        return frcmod_string

    def _gen_read_lib_string(self):
        lib_string = []
        for p in self._lib_files:
            lib_string.append('loadoff {}'.format(p))
        return lib_string


class ProteinMoleculeFromSequence(ProteinBase):
    '''
    Class to create a protein from sequence. This class will create a protein molecule from sequence. This class is pretty dumb and relies on AmberTools
    to do all of the heavy lifting.

    :param sequence: sequence of the protein to create

    The sequence is specified in Amber/Leap format. There are special NRES and CRES variants for the N-
    and C-termini. Different protonation states are also available via different residue names. E.g. ASH
    for neutral ASP.

    '''
    def __init__(self, sequence):
        super(ProteinMoleculeFromSequence, self).__init__()
        self._sequence = sequence

    def prepare_for_tleap(self, mol_id):
        # we don't need to do anything
        pass

    def generate_tleap_input(self, mol_id):
        leap_cmds = []
        leap_cmds.append('source leaprc.gaff')
        leap_cmds.extend(self._gen_read_frcmod_string())
        leap_cmds.extend(self._gen_read_prep_string())
        leap_cmds.extend(self._gen_read_lib_string())
        leap_cmds.append('{mol_id} = sequence {{ {seq} }}'.format(mol_id=mol_id, seq=self._sequence))
        leap_cmds.extend(self._gen_disulfide_string(mol_id))
        leap_cmds.extend(self._gen_bond_string(mol_id))
        leap_cmds.append(self._gen_rotation_string(mol_id))
        leap_cmds.append(self._gen_translation_string(mol_id))
        return leap_cmds


class ProteinMoleculeFromPdbFile(ProteinBase):
    '''
    Create a new protein molecule from a pdb file.
    This class is dumb and relies on AmberTools for the heavy lifting.

    :param pdb_path: string path to the pdb file

    .. note::
        no processing happens to this pdb file. It must be understandable by tleap and atoms/residues may
        need to be added/deleted/renamed. These manipulations should happen to the file before MELD is invoked.

    '''
    def __init__(self, pdb_path):
        super(ProteinMoleculeFromPdbFile, self).__init__()
        with open(pdb_path) as pdb_file:
            self._pdb_contents = pdb_file.read()

    def prepare_for_tleap(self, mol_id):
        # copy the contents of the pdb file into the current working directory
        pdb_path = '{mol_id}.pdb'.format(mol_id=mol_id)
        with open(pdb_path, 'w') as pdb_file:
            pdb_file.write(self._pdb_contents)

    def generate_tleap_input(self, mol_id):
        leap_cmds = []
        leap_cmds.append('source leaprc.gaff')
        leap_cmds.extend(self._gen_read_frcmod_string())
        leap_cmds.extend(self._gen_read_prep_string())
        leap_cmds.extend(self._gen_read_lib_string())
        leap_cmds.append('{mol_id} = loadPdb {mol_id}.pdb'.format(mol_id=mol_id))
        leap_cmds.extend(self._gen_bond_string(mol_id))
        leap_cmds.extend(self._gen_disulfide_string(mol_id))
        leap_cmds.append(self._gen_rotation_string(mol_id))
        leap_cmds.append(self._gen_translation_string(mol_id))
        #print leap_cmds
        return leap_cmds
