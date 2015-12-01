#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from collections import OrderedDict, namedtuple
import os
import math

from simtk import openmm
import numpy as np

from meld.system.system import ParmTopReader


CMAPResidue = namedtuple('CMAPResidue', 'res_num res_name index_N index_CA index_C')

#Termini residues that act as a cap and have no amap term
capped = ['ACE','NHE','OHE', 'NME', 'GLP','DUM','NAG','DIF','BER','GUM','KNI','PU5','AMP','0E9']


class CMAPAdder(object):
    _map_index = {
        'GLY': 0,
        'PRO': 1,
        'ALA': 2,
        'CYS': 3,
        'CYX': 3,
        'ASP': 3,
        'ASH': 3,
        'GLU': 3,
        'GLH': 3,
        'PHE': 3,
        'HIS': 3,
        'HIE': 3,
        'HID': 3,
        'HIP': 3,
        'ILE': 3,
        'LYS': 3,
        'LYN': 3,
        'MET': 3,
        'ASN': 3,
        'GLN': 3,
        'SER': 3,
        'THR': 3,
        'VAL': 3,
        'TRP': 3,
        'TYR': 3,
        'LEU': 3,
        'ARG': 3
    }

    def __init__(self, top_string, alpha_bias=1.0, beta_bias=1.0, ccap=False, ncap=False):
        """
        Initialize a new CMAPAdder object

        :param top_string: an Amber new-style topology in string form
        :param alpha_bias: strength of alpha correction, default=1.0
        :param beta_bias: strength of beta correction, default=1.0
        """
        self._top_string = top_string
        self._alpha_bias = alpha_bias
        self._beta_bias = beta_bias
        self._ccap = ccap
        self._ncap = ncap
        reader = ParmTopReader(self._top_string)
        self._bonds = reader.get_bonds()
        self._residue_numbers = reader.get_residue_numbers()
        self._residue_names = reader.get_residue_names()
        self._atom_map = reader.get_atom_map()
        self._ala_map = None
        self._gly_map = None
        self._pro_map = None
        self._gen_map = None
        self._load_maps()

    def add_to_openmm(self, openmm_system):
        """
        Add CMAPTorsionForce to openmm system.

        :param openmm_system:  System object to receive the force
        """
        cmap_force = openmm.CMAPTorsionForce()
        cmap_force.addMap(self._gly_map.shape[0], self._gly_map.flatten())
        cmap_force.addMap(self._pro_map.shape[0], self._pro_map.flatten())
        cmap_force.addMap(self._ala_map.shape[0], self._ala_map.flatten())
        cmap_force.addMap(self._gen_map.shape[0], self._gen_map.flatten())

        # loop over all of the contiguous chains of amino acids
        for chain in self._iterate_cmap_chains():
            # loop over the interior residues
            n_res = len(chain)
            for i in range(1, n_res-1):
                map_index = self._map_index[chain[i].res_name]
                # subtract one from all of these to get zero-based indexing, as in openmm
                c_prev = chain[i - 1].index_C - 1
                n = chain[i].index_N - 1
                ca = chain[i].index_CA - 1
                c = chain[i].index_C - 1
                n_next = chain[i+1].index_N - 1
                print "CMAP term:",i,map_index
                cmap_force.addTorsion(map_index, c_prev, n, ca, c, n, ca, c, n_next)
        openmm_system.addForce(cmap_force)

    def _iterate_cmap_chains(self):
        """
        Yield a series of chains of amino acid residues that are bonded together.

        :return: a generator that will yield lists of CMAPResidue
        """
        # use an ordered dict to remember num, name pairs in order, while removing duplicates
        residues = OrderedDict((num, name) for (num, name) in zip(self._residue_numbers, self._residue_names))
        print residues
        new_res = []
        for r in residues.items():
            num,name = r
            if name not in capped:
                new_res.append(r)
        residues = OrderedDict(new_res)
        print residues
        # now turn the ordered dict into a list of CMAPResidues
        residues = [self._to_cmap_residue(num, name) for (num, name) in residues.items()]
        print residues

        # is each residue i connected to it's predecessor, i-1?
        connected = self._compute_connected(residues)

        # now we iterate until we've handled all residues
        while connected:
            chain = [residues.pop(0)]             # we always take the first residue
            connected.pop(0)

            # if there are other residues connected, take them too
            while connected and connected[0]:
                chain.append(residues.pop(0))
                connected.pop(0)

            # we've taken a single connected chain, so yield it
            # then loop back to the beginning
            print 'CHAIN:',chain
            yield chain

    def _compute_connected(self, residues):
        """
        Return a list of boolean values indicating if each residue is connected to its predecessor.

        :param residues: a list of CMAPResidue objects
        :return: a list of boolean values indicating if residue i is bonded to i-1
        """
        def has_c_n_bond(res_i, res_j):
            """Return True if there is a bond between C of res_i and N of res_j, otherwise False."""
            if (res_i.index_C, res_j.index_N) in self._bonds:
                return True
            else:
                return False

        # zip to together consecutive residues and see if they are bonded
        connected = [has_c_n_bond(i, j) for (i, j) in zip(residues[0:], residues[1:])]
        # the first element has no element to the left, so it's not connected
        connected = [False] + connected
        return connected

    def _to_cmap_residue(self, num, name):
        """
        Turn a residue number and name into a CMAPResidue object

        :param num: residue number
        :param name: residue name
        :return: CMAPResidue
        """
        n = self._atom_map[(num, 'N')]
        ca = self._atom_map[(num, 'CA')]
        c = self._atom_map[(num, 'C')]
        res = CMAPResidue(res_num=num, res_name=name, index_N=n, index_CA=ca, index_C=c)
        return res

    def _load_map(self, stem):
        basedir = os.path.join(os.path.dirname(__file__), 'maps')
        alpha = np.loadtxt(os.path.join(basedir, '{}_alpha.txt'.format(stem))) * self._alpha_bias
        beta = np.loadtxt(os.path.join(basedir, '{}_beta.txt'.format(stem))) * self._beta_bias
        total = alpha + beta
        assert total.shape[0] == total.shape[1]
        n = int(math.ceil(total.shape[0] / 2.0))
        total = np.roll(total, -n, axis=0)
        total = np.roll(total, -n, axis=1)
        total = np.flipud(total)
        return total

    def _load_maps(self):
        """Load the maps from disk and apply the alpha and beta biases."""
        self._gly_map = self._load_map('gly')
        self._pro_map = self._load_map('pro')
        self._ala_map = self._load_map('ala')
        self._gen_map = self._load_map('gen')
