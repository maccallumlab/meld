from typing import NamedTuple, List, Dict


class ChainInfo(NamedTuple):
    residues: Dict[int, int]


class SubSystemInfo(NamedTuple):
    n_residues: int
    chains: List[ChainInfo]


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
    def __init__(
        self, abs_atom_index, rel_residue_index, residue_names, abs_resid_to_resname
    ):
        self.abs_atom_index = abs_atom_index
        self.rel_residue_index = rel_residue_index
        self.residue_names = residue_names
        self.abs_resid_to_resname = abs_resid_to_resname

    def residue_index(
        self, resid, expected_resname=None, chainid=None, one_based=False
    ):
        """
        Find the ResidueIndex

        The indexing can be either absolute (if `chainid` is `None`),
        or relative to a chain (if `chainid` is set).

        Both `resid` and `chainid` are one-based if `one_based` is `True`,
        or both are zero-based if `one_based=False` (the default).

        If `expected_resname` is specified, error checking will be performed to
        ensure that the returned atom has the expected residue name. Note
        that the residue names are those after processing with `tleap`,
        so some residue names may not match their value in an input pdb file.

        Parameters
        ----------
        resid : int
        expected_resname: str
            The expected residue name, usually three characters in all caps. E.g. "ALA".
        chainid : None or int
        one_based: bool

        Returns
        -------
        ResidueIndex
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
        self, resid, atom_name, expected_resname=None, chainid=None, one_based=False
    ):
        """
        Find the AtomIndex

        The indexing can be either absolute (if `chainid` is `None`),
        or relative to a chain (if `chainid` is set).

        Both `resid` and `chainid` are one-based if `one_based` is `True`,
        or both are zero-based if `one_based=False` (the default).

        If `expected_resname` is specified, error checking will be performed to
        ensure that the returned atom has the expected residue name. Note
        that the residue names are those after processing with `tleap`,
        so some residue names may not match their value in an input pdb file.

        Parameters
        ----------
        resid : int
        atom_name : str
        expected_resname: str
            The expected residue name, usually three characters in all caps. E.g. "ALA".
        chainid : None or int
        one_based: bool

        Returns
        -------
        AtomIndex
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
        last_chain = ChainInfo({i: j for i, j in enumerate(resids)})
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
