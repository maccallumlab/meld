"""
A module to read Amber parmtop and crd files
"""

from typing import Dict, List, Set, Tuple

import numpy as np


class CrdReader:
    """
    Class to read in coordinates from mdcrd string
    """

    _coords: np.ndarray
    _box_vectors: np.ndarray

    def __init__(self, crd_string: str):
        """
        Initialize CrdReader

        Args:
            crd_string: mdcrd string from tleap
        """
        self.crd_string = crd_string
        self._read()

    def get_coordinates(self) -> np.ndarray:
        """
        Get the coordiantes

        Returns:
            coordinates, shape(n_atoms, 3)
        """
        return self._coords

    def get_box_vectors(self) -> np.ndarray:
        """
        Get the box_vectors

        Returns:
            box vectors, shape(3, 3)
        """
        return self._box_vectors

    def _read(self):
        def split_len(seq, length):
            return [seq[i : i + length] for i in range(0, len(seq), length)]

        lines = self.crd_string.splitlines()
        n_atoms = int(lines[1].split()[0])
        coords = []
        box_vectors = None

        for line in lines[2:]:
            cols = split_len(line, 12)
            cols = [float(c) for c in cols]
            coords.extend(cols)

        # check for box vectors
        if len(coords) == 3 * n_atoms + 6:
            coords, box_vectors = coords[:-6], coords[-6:]
            for bv in box_vectors[-3:]:
                if not bv == 90.0:
                    raise RuntimeError("box angle != 90.0 degrees")
            box_vectors = np.array(box_vectors[:-3])
        elif not len(coords) == 3 * n_atoms:
            raise RuntimeError("len(coords) != 3 * n_atoms")

        coords = np.array(coords)
        coords = coords.reshape((n_atoms, 3)) / 10.0  # angstrom -> nm
        self._coords = coords

        if box_vectors is not None:
            self._box_vectors = box_vectors / 10.0
        else:
            self._box_vectors = None


class ParmTopReader:
    """
    Read in information from parmtop file
    """

    def __init__(self, top_string: str):
        """
        Initialize ParmTopReader

        Args:
            top_string: topology string from tleap
        """
        self._top_string = top_string

    def get_atom_names(self) -> List[str]:
        """
        Get the atom names

        Returns:
            the name for each atom
        """
        return self._get_parameter_block("%FLAG ATOM_NAME", chunksize=4)

    def get_residue_names(self) -> List[str]:
        """
        Get the residue names

        Returns:
            the residue name for each atom
        """
        res_names = self._get_parameter_block("%FLAG RESIDUE_LABEL", chunksize=4)
        res_numbers = self.get_residue_numbers()
        return [res_names[i - 1] for i in res_numbers]

    def get_residue_numbers(self) -> List[int]:
        """
        Get the residue numbers

        Returns:
            the residue number for each atom
        """
        n_atoms = int(self._get_parameter_block("%FLAG POINTERS", chunksize=8)[0])
        res_pointers_str = self._get_parameter_block(
            "%FLAG RESIDUE_POINTER", chunksize=8
        )
        res_pointers = [int(p) for p in res_pointers_str]
        res_pointers.append(n_atoms + 1)
        residue_numbers: List[int] = []
        for res_number, (start, end) in enumerate(
            zip(res_pointers[:-1], res_pointers[1:])
        ):
            residue_numbers.extend([res_number + 1] * (int(end) - int(start)))
        return residue_numbers

    def _get_parameter_block(self, flag: str, chunksize: int) -> List[str]:
        lines = self._top_string.splitlines()

        # find the line with our flag
        index_start = [i for (i, line) in enumerate(lines) if line.startswith(flag)][
            0
        ] + 2

        # find the index of the next flag
        index_end = [
            i for (i, line) in enumerate(lines[index_start:]) if line and line[0] == "%"
        ][0] + index_start

        # do something useful with the data
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i : i + n]

        data = []
        for line in lines[index_start:index_end]:
            for chunk in chunks(line, chunksize):
                data.append(chunk.strip())
        return data

    def get_bonds(self) -> Set[Tuple[int, int]]:
        """
        Get the bonded atoms from topology

        Returns:
            The set of bonds from the topology

        Note:
           the indices are zero-based
        """
        # the amber bonds section contains a triple of integers for each bond:
        # i, j, type_index. We need i, j, but will end up ignoring type_index
        bond_item_str = self._get_parameter_block(
            "%FLAG BONDS_WITHOUT_HYDROGEN", chunksize=8
        )
        bond_item_str += self._get_parameter_block(
            "%FLAG BONDS_INC_HYDROGEN", chunksize=8
        )
        # the bonds section of the amber file is indexed by coordinate
        # to get the atom index we divide by three and add one
        bond_items = [int(item) // 3 + 1 for item in bond_item_str]

        bonds = set()
        # take the items 3 at a time, ignoring the type_index
        for i, j, _ in zip(bond_items[::3], bond_items[1::3], bond_items[2::3]):
            # Add both orders to make life easy for callers.
            # Amber is 1-based, but we are 0-based.
            bonds.add((i - 1, j - 1))
            bonds.add((j - 1, i - 1))
        return bonds

    def get_atom_map(self) -> Dict[Tuple[int, str], int]:
        """
        Get the atom map

        Returns:
            the mapping from (resid, atom_name) to atom_index

        Note:
           both resid and atom_index are zero-based
        """
        residue_numbers = [r - 1 for r in self.get_residue_numbers()]
        atom_names = self.get_atom_names()
        atom_numbers = range(len(atom_names))
        return {
            (res_num, atom_name): atom_index
            for res_num, atom_name, atom_index in zip(
                residue_numbers, atom_names, atom_numbers
            )
        }
