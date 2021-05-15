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


class AtomIndexer:
    """
    Find AtomIndex for a given resid, name, and (optinally) chain.
    """

    def __init__(self, abs_atom_index, rel_atom_index):
        self.abs_atom_index = abs_atom_index
        self.rel_atom_index = rel_atom_index

    def __call__(self, resid, atom_name, chainid=None, one_based=False):
        """
        Find the AtomIndex

        The indexing can be either absolute (if `chainid` is `None`),
        or relative to a chain (if `chainid` is set).

        Both `resid` and `chainid` are one-based if `one_based` is `True`,
        or both are zero-based if `one_based=False` (the default).

        Parameters
        ----------
        resid : int
        atom_name : str
        chainid : None or int
        one_based: bool

        Returns
        -------
        AtomIndex
        """
        if chainid is None:
            if one_based:
                return AtomIndex(self.abs_atom_index[(resid - 1, atom_name)])
            else:
                return AtomIndex(self.abs_atom_index[(resid, atom_name)])
        else:
            if one_based:
                return AtomIndex(
                    self.rel_atom_index[(chainid - 1, resid - 1, atom_name)]
                )
            else:
                return AtomIndex(self.rel_atom_index[(chainid, resid, atom_name)])


class ResidueIndexer:
    """
    Find ResidueIndex for a given resid and (optinally) chain.
    """
    def __init__(self, rel_residue_index):
        self.rel_residue_index = rel_residue_index

    def __call__(self, resid, chainid=None, one_based=False):
        """
        Find the ResidueIndex

        The indexing can be either absolute (if `chainid` is `None`),
        or relative to a chain (if `chainid` is set).

        Both `resid` and `chainid` are one-based if `one_based` is `True`,
        or both are zero-based if `one_based=False` (the default).

        Parameters
        ----------
        resid : int
        chainid : None or int
        one_based: bool

        Returns
        -------
        ResidueIndex
        """
        if chainid is None:
            if one_based:
                return ResidueIndex(resid - 1)
            else:
                return ResidueIndex(resid)
        else:
            if one_based:
                return ResidueIndex(self.rel_residue_index[(chainid - 1, resid - 1)])
            else:
                return ResidueIndex(self.rel_residue_index[(chainid, resid)])


def _setup_indexing(chains, top, crd):
    n_atoms = crd.get_coordinates().shape[0]

    atom_names = top.get_atom_names()
    assert len(atom_names) == n_atoms

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
    atom_indexer = AtomIndexer(abs_atom_index, rel_atom_index)

    # Setup relative indexing for resids
    unique_res = set(residue_numbers)
    rel_residue_index = {}
    for res_num in unique_res:
        chainid = abs_resid_to_chainid[res_num]
        offset = abs_resid_to_offset[res_num]
        rel_residue_index[(chainid, res_num - offset)] = res_num

    residue_indexer = ResidueIndexer(rel_residue_index)

    return atom_indexer, residue_indexer
