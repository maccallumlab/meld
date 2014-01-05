import math

import numpy as np

from .restraints import RestraintManager
from meld.pdb_writer import PDBWriter


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


class GeometricTemperatureScaler(object):
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

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError('0 <= alpha <=1 1')
        if alpha <= self._alpha_min:
            return self._temperature_min
        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            delta = math.log(self._temperature_max) - math.log(self._temperature_min)
            return math.exp(delta * frac + math.log(self._temperature_min))
        else:
            return self._temperature_max


class System(object):
    def __init__(self, top_string, mdcrd_string):
        self._top_string = top_string
        self._mdcrd_string = mdcrd_string
        self.restraints = RestraintManager(self)

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

    def get_pdb_writer(self):
        return PDBWriter(range(1, len(self._atom_names) + 1),
                         self._atom_names, self._residue_numbers, self._residue_names)

    def _setup_indexing(self):
        reader = ParmTopReader(self._top_string)

        self._atom_names = reader.get_atom_names()
        assert len(self._atom_names) == self._n_atoms

        self._residue_numbers = reader.get_residue_numbers()
        assert len(self._residue_numbers) == self._n_atoms

        self._residue_names = reader.get_residue_names()
        assert len(self._residue_names) == self._n_atoms

        self._atom_index = reader.get_atom_map()

    def _setup_coords(self):
        self._coordinates = CrdReader(self._mdcrd_string).get_coordinates()
        self._n_atoms = self._coordinates.shape[0]


class CrdReader(object):
    def __init__(self, crd_string):
        self.crd_string = crd_string

    def get_coordinates(self):

        def split_len(seq, length):
            return [seq[i:i+length] for i in range(0, len(seq), length)]

        lines = self.crd_string.splitlines()
        n_atoms = int(lines[1].split()[0])
        coords = []
        for line in lines[2:]:
            cols = split_len(line, 12)
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
        return self.get_parameter_block('%FLAG ATOM_NAME', chunksize=4)

    def get_residue_names(self):
        res_names = self.get_parameter_block('%FLAG RESIDUE_LABEL', chunksize=4)
        res_numbers = self.get_residue_numbers()
        return [res_names[i - 1] for i in res_numbers]

    def get_residue_numbers(self):
        n_atoms = int(self.get_parameter_block('%FLAG POINTERS', chunksize=8)[0])
        res_pointers = self.get_parameter_block('%FLAG RESIDUE_POINTER', chunksize=8)
        res_pointers = [int(p) for p in res_pointers]
        res_pointers.append(n_atoms + 1)
        residue_numbers = []
        for res_number, (start, end) in enumerate(zip(res_pointers[:-1], res_pointers[1:])):
            residue_numbers.extend([res_number + 1] * (end - start))
        return residue_numbers

    def get_parameter_block(self, flag, chunksize):
        lines = self._top_string.splitlines()

        # find the line with our flag
        index_start = [i for (i, line) in enumerate(lines) if line.startswith(flag)][0] + 2

        # find the index of the next flag
        index_end = [i for (i, line) in enumerate(lines[index_start:]) if line and line[0] == '%'][0] + index_start

        # do something useful with the data
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in xrange(0, len(l), n):
                yield l[i:i+n]
        data = []
        for line in lines[index_start:index_end]:
            for chunk in chunks(line, chunksize):
                data.append(chunk.strip())
        return data

    def get_bonds(self):
        # the amber bonds section contains a triple of integers for each bond:
        # i, j, type_index. We need i, j, but will end up ignoring type_index
        bond_items = self.get_parameter_block('%FLAG BONDS_WITHOUT_HYDROGEN', chunksize=8)
        bond_items += self.get_parameter_block('%FLAG BONDS_INC_HYDROGEN', chunksize=8)
        # the bonds section of the amber file is indexed by coordinate
        # to get the atom index we divide by three and add one
        bond_items = [int(item) / 3 + 1 for item in bond_items]

        bonds = set()
        # take the items 3 at a time, ignoring the type_index
        for i, j, _ in zip(bond_items[::3], bond_items[1::3], bond_items[2::3]):
            # add both orders to make life easy for callers
            bonds.add((i, j))
            bonds.add((j, i))
        return bonds

    def get_atom_map(self):
        residue_numbers = self.get_residue_numbers()
        atom_names = self.get_atom_names()
        atom_numbers = range(1, len(atom_names) + 1)
        return {(res_num, atom_name): atom_index for res_num, atom_name, atom_index in
                zip(residue_numbers, atom_names, atom_numbers)}


class RunOptions(object):
    def __init__(self):
        self._runner = 'openmm'
        self._timesteps = 5000
        self._minimize_Steps = 1000
        self._implicit_solvent_model = 'gbNeck2'
        self._cutoff = None
        self._use_big_timestep = False
        self._use_amap = False
        self._amap_alpha_bias = 1.0
        self._amap_beta_bias = 1.0
        self._softcore = False
        self._sc_alpha_min = 0.0
        self._sc_alpha_max_coulomb = 0.3
        self._sc_alpha_max_lennard_jones = 1.0
        self._remove_com = True

    @property
    def remove_com(self):
        return self._remove_com
    @remove_com.setter
    def remove_com(self, new_value):
        self._remove_com = bool(new_value)

    @property
    def softcore(self):
        return self._softcore

    @softcore.setter
    def softcore(self, new_value):
        self._softcore = bool(new_value)
        self._check_sc()

    @property
    def sc_alpha_min(self):
        return self._sc_alpha_min

    @sc_alpha_min.setter
    def sc_alpha_min(self, new_value):
        self._sc_alpha_min = float(new_value)
        self._check_sc()

    @property
    def sc_alpha_max_coulomb(self):
        return self._sc_alpha_max_coulomb

    @sc_alpha_max_coulomb.setter
    def sc_alpha_max_coulomb(self, new_value):
        self._sc_alpha_max_coulomb = float(new_value)
        self._check_sc()

    @property
    def sc_alpha_max_lennard_jones(self):
        return self._sc_alpha_max_lennard_jones

    @sc_alpha_max_lennard_jones.setter
    def sc_alpha_max_lennard_jones(self, new_value):
        self._sc_alpha_max_lennard_jones = float(new_value)
        self._check_sc()

    @property
    def runner(self):
        return self._runner

    @runner.setter
    def runner(self, value):
        if not value in ['openmm', 'fake_runner']:
            raise RuntimeError('unknown value for runner {}'.format(value))
        self._runner = value

    @property
    def timesteps(self):
        return self._timesteps

    @timesteps.setter
    def timesteps(self, value):
        value = int(value)
        if value <= 0:
            raise RuntimeError('timesteps must be > 0')
        self._timesteps = value

    @property
    def minimize_steps(self):
        return self._minimize_Steps

    @minimize_steps.setter
    def minimize_steps(self, value):
        value = int(value)
        if value <= 0:
            raise RuntimeError('minimize_steps must be > 0')
        self._minimize_steps = value

    @property
    def implicit_solvent_model(self):
        return self._implicit_solvent_model

    @implicit_solvent_model.setter
    def implicit_solvent_model(self, value):
        if not value in [None, 'obc', 'gbNeck', 'gbNeck2']:
            raise RuntimeError('unknown value for implicit solvent model {}'.format(value))
        self._implicit_solvent_model = value

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value):
        if value is None:
            self._cutoff = None
        else:
            value = float(value)
            if value <= 0:
                raise RuntimeError('cutoff must be > 0')
            self._cutoff = value

    @property
    def use_big_timestep(self):
        return self._use_big_timestep

    @use_big_timestep.setter
    def use_big_timestep(self, value):
        self._use_big_timestep = bool(value)

    @property
    def use_amap(self):
        return self._use_amap

    @use_amap.setter
    def use_amap(self, value):
        self._use_amap = bool(value)

    @property
    def amap_alpha_bias(self):
        return self._amap_alpha_bias

    @amap_alpha_bias.setter
    def amap_alpha_bias(self, value):
        if value < 0:
            raise RuntimeError('amap_alpha_bias < 0')
        self._amap_alpha_bias = value

    @property
    def amap_beta_bias(self):
        return self._amap_beta_bias

    @amap_beta_bias.setter
    def amap_beta_bias(self, value):
        if value < 0:
            raise RuntimeError('amap_beta_bias < 0')
        self._amap_beta_bias = value

    def _check_sc(self):
        if self._softcore:
            assert self._sc_alpha_min >= 0.0
            assert self._sc_alpha_max_coulomb > self._sc_alpha_min
            assert self._sc_alpha_max_lennard_jones > self._sc_alpha_max_coulomb
            assert self._sc_alpha_max_lennard_jones <= 1.0
