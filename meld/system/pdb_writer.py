#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module for PDB output
"""

from typing import List

import numpy as np  # type: ignore

header = "REMARK stage {stage}"
footer = "TER\nEND\n\n"
template = (
    "ATOM  {atom_number:>5d} {atom_name:4s} {residue_name:>3s} "
    + "{residue_number:5d}    {x:8.3f}{y:8.3f}{z:8.3f}"
)


class PDBWriter:
    """
    Convert states into PDB format
    """

    def __init__(
        self,
        atom_numbers: List[int],
        atom_names: List[str],
        residue_numbers: List[int],
        residue_names: List[str],
    ) -> None:
        """
        Initialize a PDBWriter

        Args:
            atom_numbers: number for each atom
            atom_names: name for each atom
            residue_numbers: residue number for each atom
            residue_names: residue name for each atom
        """
        self._atom_numbers = atom_numbers
        self._n_atoms = len(atom_numbers)
        self.header = header
        self.footer = footer
        self.template = template

        assert len(atom_names) == self._n_atoms
        self._atom_names = atom_names
        self._atom_names = [
            "".join([" ", atom_name]) if len(atom_name) < 4 else atom_name
            for atom_name in self._atom_names
        ]

        assert len(residue_numbers) == self._n_atoms
        self._residue_numbers = residue_numbers

        assert len(residue_names) == self._n_atoms
        self._residue_names = residue_names

    def get_pdb_string(self, coordinates: np.ndarray, stage: int) -> str:
        """
        Get pdb representation for coordinates

        Args:
            coordinates: n_atoms x 3 array
            stage: stage number to record in pdb

        Returns:
            string representation in pdb format
        """
        assert coordinates.shape[0] == self._n_atoms
        assert coordinates.shape[1] == 3

        zipper = zip(
            self._atom_numbers,
            self._atom_names,
            self._residue_numbers,
            self._residue_names,
            range(coordinates.shape[0]),
        )
        lines = [
            self.template.format(
                atom_number=atom_num,
                atom_name=atom_name,
                residue_name=res_name,
                residue_number=res_num,
                x=coordinates[i, 0] * 10.0,  # nm -> angstrom
                y=coordinates[i, 1] * 10.0,
                z=coordinates[i, 2] * 10.0,
            )
            for atom_num, atom_name, res_num, res_name, i in zipper
        ]
        lines.insert(0, self.header.format(stage=stage))
        lines.append(self.footer)
        return "\n".join(lines)
