from typing import NamedTuple


class ChainInfo(NamedTuple):
    # The 0-based start index for this chain
    start: int
    # One past the end of this chain
    end: int


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


class AtomIndexer:
    """
    Find AtomIndex for a given resid, name, and (optinally) chain.
    """

    def __init__(self, abs_atom_index, rel_atom_index, res_names):
        self.abs_atom_index = abs_atom_index
        self.rel_atom_index = rel_atom_index
        self.res_names = res_names

    def __call__(
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
        if chainid is None:
            if one_based:
                atom_index = AtomIndex(self.abs_atom_index[(resid - 1, atom_name)])
            else:
                atom_index = AtomIndex(self.abs_atom_index[(resid, atom_name)])
        else:
            if one_based:
                atom_index = AtomIndex(
                    self.rel_atom_index[(chainid - 1, resid - 1, atom_name)]
                )
            else:
                atom_index = AtomIndex(self.rel_atom_index[(chainid, resid, atom_name)])

        # Ensure that the resname matches what is expected
        if expected_resname is not None:
            actual_resname = self.res_names[int(atom_index)]
            if expected_resname != actual_resname:
                raise KeyError(
                    f"expected_resname={expected_resname}, but found res_name={actual_resname}."
                )

        return atom_index


class ResidueIndexer:
    """
    Find ResidueIndex for a given resid and (optinally) chain.
    """

    def __init__(self, rel_residue_index, resid_to_resname):
        self.rel_residue_index = rel_residue_index
        self.res_id_to_resname = resid_to_resname

    def __call__(self, resid, expected_resname=None, chainid=None, one_based=False):
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
                res_index = ResidueIndex(resid - 1)
            else:
                res_index = ResidueIndex(resid)
        else:
            if one_based:
                res_index = ResidueIndex(
                    self.rel_residue_index[(chainid - 1, resid - 1)]
                )
            else:
                res_index = ResidueIndex(self.rel_residue_index[(chainid, resid)])

        if expected_resname is not None:
            actual_resname = self.res_id_to_resname[int(res_index)]
            if actual_resname in cannonicalize:
                actual_resname = cannonicalize[actual_resname]
            if actual_resname != expected_resname:
                raise KeyError(
                    f"expected_resname={expected_resname}, but found res_name={actual_resname}."
                )

        return res_index


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
    max_chain_resid = max(chain.end for chain in chains)  # max resid + 1
    max_resid = max(residue_numbers)
    if max_resid >= max_chain_resid:
        last_chain = ChainInfo(max_chain_resid, max_resid + 1)
        chains.append(last_chain)

    # Setup mapping between abs_resid and chainid
    abs_resid_to_chainid = {}
    for i, chain in enumerate(chains):
        for j in range(chain.start, chain.end):
            if j in abs_resid_to_chainid:
                raise RuntimeError()
            abs_resid_to_chainid[j] = i

    # Setup mapping between abs_resid and offset
    abs_resid_to_offset = {}
    for chain in chains:
        for i in range(chain.start, chain.end):
            if i in abs_resid_to_offset:
                raise RuntimeError()
            abs_resid_to_offset[i] = chain.start

    # Now setup relative indexing.
    # This maps from (chain_id, rel_resid, name): atom_index
    rel_atom_index = {}
    for res_num, atom_name, atom_index in zip(
        residue_numbers, atom_names, atom_numbers
    ):
        chainid = abs_resid_to_chainid[res_num]
        offset = abs_resid_to_offset[res_num]
        rel_atom_index[(chainid, res_num - offset, atom_name)] = atom_index

    # Setup our atom indexer based on absolute and relative indexing.
    atom_indexer = AtomIndexer(abs_atom_index, rel_atom_index, residue_names)

    # Setup mapping of resid to resname
    resid_to_resname = {}
    for resid, resname in zip(residue_numbers, residue_names):
        if resid in resid_to_resname:
            if resid_to_resname[resid] != resname:
                raise RuntimeError("Inconsistient residue names")
        else:
            resid_to_resname[resid] = resname

    # Setup relative indexing for resids
    unique_res = set(residue_numbers)
    rel_residue_index = {}
    for res_num in unique_res:
        chainid = abs_resid_to_chainid[res_num]
        offset = abs_resid_to_offset[res_num]
        rel_residue_index[(chainid, res_num - offset)] = res_num

    residue_indexer = ResidueIndexer(rel_residue_index, resid_to_resname)

    return atom_indexer, residue_indexer
