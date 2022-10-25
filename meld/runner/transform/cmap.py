#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Add AMAP correction for GB models
"""

from meld import interfaces
from meld.system import amber
from meld.runner import transform
from meld.system import options
from simtk import openmm as mm  # type: ignore
from openmm import app  # type: ignore

from collections import OrderedDict
import os
import math
import numpy as np  # type: ignore
from typing import NamedTuple, List, Set, Tuple, Dict, Iterator


class CMAPResidue(NamedTuple):
    res_num: int
    res_name: str
    index_N: int
    index_CA: int
    index_C: int


# Termini residues that act as a cap and have no amap term
capped = [
    "ACE",
    "NHE",
    "OHE",
    "NME",
    "GLP",
    "DUM",
    "NAG",
    "DIF",
    "BER",
    "GUM",
    "KNI",
    "PU5",
    "AMP",
    "0E9",
]

residue_to_map = {
    "GLY": 0,
    "PRO": 1,
    "ALA": 2,
    "CYS": 3,
    "CYX": 3,
    "ASP": 3,
    "ASH": 3,
    "GLU": 3,
    "GLH": 3,
    "PHE": 3,
    "HIS": 3,
    "HIE": 3,
    "HID": 3,
    "HIP": 3,
    "ILE": 3,
    "LYS": 3,
    "LYN": 3,
    "MET": 3,
    "ASN": 3,
    "GLN": 3,
    "SER": 3,
    "THR": 3,
    "VAL": 3,
    "TRP": 3,
    "TYR": 3,
    "LEU": 3,
    "ARG": 3,
}


class CMAPTransformer(transform.TransformerBase):
    _alpha_bias: float
    _beta_bias: float
    _ccap: bool
    _ncap: bool
    _bonds: Set[Tuple[int, int]]
    _residue_numbers: List[int]
    _residue_names: List[str]
    _atom_map: Dict[Tuple[int, str], int]
    _ala_map: np.ndarray
    _gly_map: np.ndarray
    _pro_map: np.ndarray
    _gen_map: np.ndarray

    def __init__(
        self,
        options: options.RunOptions,
        top_string: str,
        ccap: bool = False,
        ncap: bool = False,
    ) -> None:
        """
        Initialize a new CMAPTransformer

        Args:
            options: run options
            top_string: an Amber new-style topology in string form
        """
        self._alpha_bias = options.amap_alpha_bias
        self._beta_bias = options.amap_beta_bias
        self._active = options.use_amap
        self._top_string = top_string
        self._ccap = ccap
        self._ncap = ncap
        if self._active:
            reader = amber.ParmTopReader(self._top_string)
            self._bonds = reader.get_bonds()
            self._residue_numbers = [r - 1 for r in reader.get_residue_numbers()]
            self._residue_names = reader.get_residue_names()
            self._atom_map = reader.get_atom_map()
            self._load_maps()

    def add_interactions(
        self, state: interfaces.IState, openmm_system: mm.System, topology: app.Topology
    ) -> mm.System:
        if not self._active:
            return openmm_system

        cmap_force = mm.CMAPTorsionForce()
        cmap_force.addMap(self._gly_map.shape[0], self._gly_map.flatten())
        cmap_force.addMap(self._pro_map.shape[0], self._pro_map.flatten())
        cmap_force.addMap(self._ala_map.shape[0], self._ala_map.flatten())
        cmap_force.addMap(self._gen_map.shape[0], self._gen_map.flatten())

        # loop over all of the contiguous chains of amino acids
        for chain in self._iterate_cmap_chains():
            # loop over the interior residues
            n_res = len(chain)
            for i in range(1, n_res - 1):
                map_index = residue_to_map[chain[i].res_name]
                c_prev = chain[i - 1].index_C
                n = chain[i].index_N
                ca = chain[i].index_CA
                c = chain[i].index_C
                n_next = chain[i + 1].index_N
                cmap_force.addTorsion(map_index, c_prev, n, ca, c, n, ca, c, n_next)
        openmm_system.addForce(cmap_force)
        return openmm_system

    def _iterate_cmap_chains(self) -> Iterator[List[CMAPResidue]]:
        """
        Yield a series of chains of amino acid residues that are bonded
        together.

        Returns:
            a generator over chains
        """
        # use an ordered dict to remember num, name pairs in order, while
        # removing duplicates
        residues = OrderedDict(
            (num, name)
            for (num, name) in zip(self._residue_numbers, self._residue_names)
        )
        new_res = []
        for r in residues.items():
            num, name = r
            if name not in capped:
                new_res.append(r)

        ordered_residues = OrderedDict(new_res)

        # now turn the ordered dict into a list of CMAPResidues
        cmap_residues = [
            self._to_cmap_residue(num, name)
            for (num, name) in ordered_residues.items()
            if name in residue_to_map.keys()
        ]

        # is each residue i connected to it's predecessor, i-1?
        connected = self._compute_connected(cmap_residues)

        # now we iterate until we've handled all residues
        while connected:
            chain = [cmap_residues.pop(0)]  # we always take the first residue
            connected.pop(0)

            # if there are other residues connected, take them too
            while connected and connected[0]:
                chain.append(cmap_residues.pop(0))
                connected.pop(0)

            # we've taken a single connected chain, so yield it
            # then loop back to the beginning
            yield chain

    def _compute_connected(self, residues: List[CMAPResidue]) -> List[bool]:
        """
        Return a list of boolean values indicating if each residue is
        connected to its predecessor.

        Args:
            residues: the list of residues

        Returns:
            the list of sequential connectivities
        """

        def has_c_n_bond(res_i, res_j):
            """Return True if there is a bond between C of res_i and N of
            res_j, otherwise False.
            """
            if (res_i.index_C, res_j.index_N) in self._bonds:
                return True
            else:
                return False

        # zip to together consecutive residues and see if they are bonded
        connected = [has_c_n_bond(i, j) for (i, j) in zip(residues[0:], residues[1:])]
        # the first element has no element to the left, so it's not connected
        connected = [False] + connected
        return connected

    def _to_cmap_residue(self, num: int, name: str) -> CMAPResidue:
        """
        Turn a residue number and name into a CMAPResidue object

        Args:
            num: residue number
            name: residue name
        Returns:
            the CMAPResidue
        """
        n = self._atom_map[(num, "N")]
        ca = self._atom_map[(num, "CA")]
        c = self._atom_map[(num, "C")]
        res = CMAPResidue(res_num=num, res_name=name, index_N=n, index_CA=ca, index_C=c)
        return res

    def _load_map(self, stem: str) -> np.ndarray:
        basedir = os.path.join(os.path.dirname(__file__), "maps")
        alpha = (
            np.loadtxt(os.path.join(basedir, f"{stem}_alpha.txt")) * self._alpha_bias
        )
        beta = np.loadtxt(os.path.join(basedir, f"{stem}_beta.txt")) * self._beta_bias
        total = alpha + beta
        assert total.shape[0] == total.shape[1]
        n = int(math.ceil(total.shape[0] / 2.0))
        total = np.roll(total, -n, axis=0)
        total = np.roll(total, -n, axis=1)
        total = np.flipud(total)
        return total

    def _load_maps(self) -> None:
        """Load the maps from disk and apply the alpha and beta biases."""
        self._gly_map = self._load_map("gly")
        self._pro_map = self._load_map("pro")
        self._ala_map = self._load_map("ala")
        self._gen_map = self._load_map("gen")
