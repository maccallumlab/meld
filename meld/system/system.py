import numpy as np


class ConstantTemperatureScaler(object):
    def __init__(self, temperature):
        self._temperature = temperature

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError('0 <= alpha <= 1. alpha={}'.format(alpha))
        return self._temperature


class LinearTemperatureScaler(object):
    def __init__(self, alpha_min, alpha_max, temperature_min, temperature_max):
        if alpha_min < 0 or alpha_min > 1:
            raise RuntimeError('0 <= alpha_min <=1')
        if alpha_max < 0 or alpha_max > 1:
            raise RuntimeError('0 <= alpha_max <=1')
        if alpha_min >= alpha_max:
            raise RuntimeError('alpha_min must be < alpha_max')
        if temperature_min <= 0 or temperature_max <= 0:
            raise RuntimeError('temperatures must be positive')

        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._temperature_min = float(temperature_min)
        self._temperature_max = float(temperature_max)
        self._delta_alpha = self._alpha_max - self._alpha_min
        self._delta_temp = self._temperature_max - self._temperature_min

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError('0 <= alpha <=1 1')
        if alpha <= self._alpha_min:
            return self._temperature_min
        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            return self._temperature_min + frac * self._delta_temp
        else:
            return self._temperature_max


class System(object):
    def __init__(self, top_string, mdcrd_string):
        self._top_string = top_string
        self._mdcrd_string = mdcrd_string

        self.temperature_scaler = None
        self._coordinates = None
        self._n_atoms = None
        self._setup_coords()

        self._atom_names = None
        self._residue_names = None
        self._residue_numbers = None
        self._atom_index = None
        self._setup_indexing()

    @property
    def top_string(self):
        return self._top_string

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
        try:
            return self._atom_index[(residue_number, atom_name)]
        except KeyError:
            print 'Could not find atom index for residue_number={} and atom name={}.'.format(
                residue_number, atom_name)
            raise

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
        return [res_names[i - 1] for i in res_numbers]

    def get_residue_numbers(self):
        n_atoms = int(self._get_parameter_block('%FLAG POINTERS')[0])
        res_pointers = self._get_parameter_block('%FLAG RESIDUE_POINTER')
        res_pointers = [int(p) for p in res_pointers]
        res_pointers.append(n_atoms + 1)
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
