#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Add AMAP correction for GB models
"""

import math
import os
from typing import Iterator, List, NamedTuple

import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import app  # type: ignore


class CMAPResidue(NamedTuple):
    res_num: int
    res_name: str
    N: app.Atom
    CA: app.Atom
    C: app.Atom
    residue: app.Residue


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


def add_amap(
    system: mm.System,
    topology: app.Topology,
    alpha_bias: float,
    beta_bias: float,
):
    gly_map = _load_map("gly", alpha_bias, beta_bias)
    pro_map = _load_map("pro", alpha_bias, beta_bias)
    ala_map = _load_map("ala", alpha_bias, beta_bias)
    gen_map = _load_map("gen", alpha_bias, beta_bias)

    cmap_force = mm.CMAPTorsionForce()
    cmap_force.addMap(gly_map.shape[0], gly_map.flatten())
    cmap_force.addMap(pro_map.shape[0], pro_map.flatten())
    cmap_force.addMap(ala_map.shape[0], ala_map.flatten())
    cmap_force.addMap(gen_map.shape[0], gen_map.flatten())

    for chain in _iterate_cmap_chains(topology):
        n_res = len(chain)

        for i in range(1, n_res - 1):
            map_index = residue_to_map[chain[i].res_name]
            c_prev = chain[i - 1].C.index
            n = chain[i].N.index
            ca = chain[i].CA.index
            c = chain[i].C.index
            n_next = chain[i + 1].N.index
            cmap_force.addTorsion(map_index, c_prev, n, ca, c, n, ca, c, n_next)
    system.addForce(cmap_force)


def _load_map(stem: str, alpha_bias: float, beta_bias: float) -> np.ndarray:
    basedir = os.path.join(os.path.dirname(__file__), "maps")
    alpha = np.loadtxt(os.path.join(basedir, f"{stem}_alpha.txt")) * alpha_bias
    beta = np.loadtxt(os.path.join(basedir, f"{stem}_beta.txt")) * beta_bias
    total = alpha + beta
    assert total.shape[0] == total.shape[1]
    n = int(math.ceil(total.shape[0] / 2.0))
    total = np.roll(total, -n, axis=0)
    total = np.roll(total, -n, axis=1)
    total = np.flipud(total)
    return total


def _iterate_cmap_chains(topology: app.Topology) -> Iterator[List[CMAPResidue]]:
    """
    Yield a series of chains of amino acid residues that are bonded
    together.

    Returns:
        a generator over chains
    """
    residues = _get_cmap_residues(topology)

    # is each residue i connected to it's predecessor, i-1?
    connected = _compute_connected(residues)

    # now we iterate until we've handled all residues
    while connected:
        chain = [residues.pop(0)]  # we always take the first residue
        connected.pop(0)

        # if there are other residues connected, take them too
        while connected and connected[0]:
            chain.append(residues.pop(0))
            connected.pop(0)

        # we've taken a single connected chain, so yield it
        # then loop back to the beginning
        yield chain


def _compute_connected(residues: List[CMAPResidue]) -> List[bool]:
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
        bonds = list(res_i.residue.bonds())
        if app.topology.Bond(res_i.C, res_j.N) in bonds:
            return True
        elif app.topology.Bond(res_j.N, res_i.C) in bonds:
            return True
        else:
            return False

    # zip to together consecutive residues and see if they are bonded
    connected = [has_c_n_bond(i, j) for (i, j) in zip(residues[0:], residues[1:])]
    # the first element has no element to the left, so it's not connected
    connected = [False] + connected
    return connected


def _get_cmap_residues(topology: app.Topology) -> List[CMAPResidue]:
    cmap_residues = []
    for residue in topology.residues():
        if residue.name in residue_to_map.keys():
            atom_map = {atom.name: atom for atom in residue.atoms()}
            n = atom_map["N"]
            ca = atom_map["CA"]
            c = atom_map["C"]
            res = CMAPResidue(
                res_num=residue.index,
                res_name=residue.name,
                N=n,
                CA=ca,
                C=c,
                residue=residue,
            )
            cmap_residues.append(res)
    return cmap_residues
