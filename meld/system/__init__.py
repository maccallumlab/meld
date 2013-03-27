import numpy as np


class System(object):
    def __init__(self, top_string, mdcrd_string):
        self._top_string = top_string
        self._mdcrd_string = mdcrd_string

        self._coordinates = None
        self._n_atoms = None
        self._setup_coords()

        self._atom_names = None
        self._residue_names = None
        self._residue_numbers = None
        self._atom_index = None
        self._setup_indexing()

    @property
    def n_atoms(self):
        return self._n_atoms

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def atom_names(self):
        return self._atom_names

    @property
    def residue_numbers(self):
        return self._residue_numbers

    @property
    def residue_names(self):
        return self._residue_names

    def index_of_atom(self, residue_number, atom_name):
        return self._atom_index[(residue_number, atom_name)]

    def _setup_indexing(self):
        reader = ParmTopReader(self._top_string)

        self._atom_names = reader.get_atom_names()
        assert len(self._atom_names) == self._n_atoms

        self._residue_numbers = reader.get_residue_numbers()
        assert len(self._residue_numbers) == self._n_atoms

        self._residue_names = reader.get_residue_names()
        assert len(self._residue_names) == self._n_atoms

        self._atom_index = {}
        for res_index, atom_name, atom_index in zip(self._residue_numbers, self._atom_names, range(self.n_atoms)):
            self._atom_index[(res_index, atom_name)] = atom_index + 1

    def _setup_coords(self):
        self._coordinates = CrdReader(self._mdcrd_string).get_coordinates()
        self._n_atoms = self._coordinates.shape[0]


class CrdReader(object):
    def __init__(self, crd_string):
        self.crd_string = crd_string

    def get_coordinates(self):
        lines = self.crd_string.splitlines()
        n_atoms = int(lines[1].split()[0])
        coords = []
        for line in lines[2:]:
            cols = line.split()
            cols = [float(c) for c in cols]
            coords.extend(cols)
        assert len(coords) == 3 * n_atoms
        coords = np.array(coords)
        coords = coords.reshape((n_atoms, 3))
        return coords


class ParmTopReader(object):
    def __init__(self, top_string):
        self._top_string = top_string

    def get_atom_names(self):
        return self._get_parameter_block('%FLAG ATOM_NAME')

    def get_residue_names(self):
        res_names = self._get_parameter_block(('%FLAG RESIDUE_LABEL'))
        res_numbers = self.get_residue_numbers()
        return [res_names[i-1] for i in res_numbers]

    def get_residue_numbers(self):
        n_atoms = int(self._get_parameter_block('%FLAG POINTERS')[0])
        res_pointers = self._get_parameter_block('%FLAG RESIDUE_POINTER')
        res_pointers = [int(p) for p in res_pointers]
        res_pointers.append(n_atoms+1)
        residue_numbers = []
        for res_number, (start, end) in enumerate(zip(res_pointers[:-1], res_pointers[1:])):
            residue_numbers.extend([res_number + 1] * (end - start))
        return residue_numbers

    def _get_parameter_block(self, flag):
        data = []
        lines = self._top_string.splitlines()

        # find the line with our flag
        index_start = [i for (i, line) in enumerate(lines) if line.startswith(flag)][0] + 2

        # find the index of the next flag
        index_end = [i for (i, line) in enumerate(lines[index_start:]) if line and line[0] == '%'][0] + index_start

        # do something useful with the data
        data = []
        for line in lines[index_start:index_end]:
            data.extend(line.split())
        return data
