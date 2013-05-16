class PDBWriter(object):
    header = 'REMARK stage {stage}'
    footer = 'TER\nEND\n\n'
    template = 'ATOM  {atom_number:>5d} {atom_name:4s} {residue_name:>3s} {residue_number:5d}    {x:8.3f}{y:8.3f}{z:8.3f}'

    def __init__(self, atom_numbers, atom_names, residue_numbers, residue_names):
        self._atom_numbers = atom_numbers
        self._n_atoms = len(atom_numbers)

        assert len(atom_names) == self._n_atoms
        self._atom_names = atom_names

        assert len(residue_numbers) == self._n_atoms
        self._residue_numbers = residue_numbers

        assert len(residue_names) == self._n_atoms
        self._residue_names = residue_names

    def get_pdb_string(self, coordinates, stage):
        assert coordinates.shape[0] == self._n_atoms
        assert coordinates.shape[1] == 3

        lines = []
        lines.append(self.header.format(stage=stage))
        for atom_num, atom_name, res_num, res_name, i in zip(self._atom_numbers, self._atom_names,
                                                             self._residue_numbers, self._residue_names,
                                                             range(coordinates.shape[0])):
            if len(atom_name) < 4:
                atom_name = ''.join([' ', atom_name])
            lines.append(self.template.format(atom_number=atom_num,
                                              atom_name=atom_name,
                                              residue_name=res_name,
                                              residue_number=res_num,
                                              x=coordinates[i, 0],
                                              y=coordinates[i, 1],
                                              z=coordinates[i, 2]))
        lines.append(self.footer)
        return '\n'.join(lines)
