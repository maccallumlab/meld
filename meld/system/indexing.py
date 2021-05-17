"""
Look up atom and residue indices.

The :class:`Indexer` class is the main object for performing
indexing look ups.

An :class:`AtomIndex` is a zero-based absolute atom index.
A :class:`ResidueIndex` is a zero-based absolute residue index.

"""

from typing import NamedTuple, List, Dict, Tuple, Optional


class _ChainInfo(NamedTuple):
    residues: Dict[int, int]
    "maps from index_within_chain to global_index"


class _SubSystemInfo(NamedTuple):
    n_residues: int
    "number of residues in this subsystem"
    chains: List[_ChainInfo]
    "a list of chains in the subsystem"


#
# We create AtomIndex and ResidueIndex classes that just
# wrap int. This allows us to check elsewhere in the code
# that we're getting the right type of index.
class AtomIndex(int):
    """
    Zero-based absolute atom index.

    Usually, you should get this through `system.atom_index`,
    but if you are _sure_ you know what you are doing,
    you can construct directly via `AtomIndex(index)`.
    """

    pass


class ResidueIndex(int):
    """
    Zero-based absolute residue index.

    Usually, you should get this through `system.residue_index`,
    but if you are _sure_ you know what you are doing,
    you can construct directly via `ResidueIndex(index)`.
    """

    pass


#
# Dictionary to cannonicalize residue names
#
cannonicalize = {
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "ASH": "ASP",
    "GLH": "GLU",
    "LYN": "LYS",
    "CYX": "CYS",
}


class Indexer:
    """
    An object for performing atom and residue index lookups.
    """

    def __init__(
        self,
        abs_atom_index: Dict[Tuple[int, str], int],
        rel_residue_index: Dict[Tuple[int, int], int],
        residue_names: List[str],
        abs_resid_to_resname: Dict[int, str],
    ):
        """
        Initialize an Indexer object.

        Args:
            abs_atom_index: maps (resid, atom_name) to atomid
            rel_residue_index: maps (chainid, rel_resid) to resid
            residue_names: residue name for each atom
            abs_resid_to_resname: maps resid to resname
        """
        self.abs_atom_index = abs_atom_index
        self.rel_residue_index = rel_residue_index
        self.residue_names = residue_names
        self.abs_resid_to_resname = abs_resid_to_resname

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
        if chainid is None:
            if one_based:
                res_index = resid - 1
            else:
                res_index = resid
        else:
            if one_based:
                res_index = self.rel_residue_index[(chainid - 1, resid - 1)]
            else:
                res_index = self.rel_residue_index[(chainid, resid)]

        if expected_resname is not None:
            actual_resname = self.abs_resid_to_resname[res_index]
            if actual_resname in cannonicalize:
                actual_resname = cannonicalize[actual_resname]
            if actual_resname != expected_resname:
                raise KeyError(
                    f"expected_resname={expected_resname}, but found res_name={actual_resname}."
                )

        return ResidueIndex(res_index)

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
        resindex = self.residue_index(resid, expected_resname, chainid, one_based)
        return AtomIndex(self.abs_atom_index[resindex, atom_name])


def _setup_indexing(chains, top, crd):
    n_atoms = crd.get_coordinates().shape[0]

    atom_names = top.get_atom_names()
    assert len(atom_names) == n_atoms

    residue_names = top.get_residue_names()
    assert len(residue_names) == n_atoms

    residue_numbers = [r - 1 for r in top.get_residue_numbers()]
    assert len(residue_numbers) == n_atoms

    atom_numbers = list(range(n_atoms))

    # First setup absolute indexing.
    # This maps from (abs_resid, name): atom_index
    # Everything is zero-based
    abs_atom_index = {
        (res_num, atom_name): atom_index
        for res_num, atom_name, atom_index in zip(
            residue_numbers, atom_names, atom_numbers
        )
    }

    # Addtional residues may have been added after chains was
    # calculated, e.g. through the RdcPatcher or through tleap
    # adding explicit solvent and ions. If present, we will add
    # these extra residues to a final group.
    max_chain_resid = max(max(chain.residues.values()) for chain in chains)
    max_resid = max(residue_numbers)
    if max_resid > max_chain_resid:
        resids = list(range(max_chain_resid + 1, max_resid + 1))
        last_chain = _ChainInfo({i: j for i, j in enumerate(resids)})
        chains.append(last_chain)

    rel_residue_index = {}
    for i, chain in enumerate(chains):
        for rel_index, abs_index in chain.residues.items():
            rel_residue_index[(i, rel_index)] = abs_index

    # Setup mapping of resid to resname
    resid_to_resname = {}
    for resid, resname in zip(residue_numbers, residue_names):
        if resid in resid_to_resname:
            if resid_to_resname[resid] != resname:
                raise RuntimeError("Inconsistient residue names")
        else:
            resid_to_resname[resid] = resname

    return Indexer(abs_atom_index, rel_residue_index, residue_names, resid_to_resname)
