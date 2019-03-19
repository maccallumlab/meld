#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import math
import numpy as np  # type: ignore
from collections import namedtuple

from meld.system.restraints import RestraintManager
from meld.pdb_writer import PDBWriter
from simtk.unit import atmosphere  # type: ignore


class TemperatureScaler:
    def __call__(self, alpha: float) -> float:
        pass


class ConstantTemperatureScaler(TemperatureScaler):
    def __init__(self, temperature):
        self._temperature = temperature

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError(f"0 <= alpha <= 1. alpha={alpha}")
        return self._temperature


class LinearTemperatureScaler(TemperatureScaler):
    def __init__(self, alpha_min, alpha_max, temperature_min, temperature_max):
        if alpha_min < 0 or alpha_min > 1:
            raise RuntimeError("0 <= alpha_min <=1")
        if alpha_max < 0 or alpha_max > 1:
            raise RuntimeError("0 <= alpha_max <=1")
        if alpha_min >= alpha_max:
            raise RuntimeError("alpha_min must be < alpha_max")
        if temperature_min <= 0 or temperature_max <= 0:
            raise RuntimeError("temperatures must be positive")

        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._temperature_min = float(temperature_min)
        self._temperature_max = float(temperature_max)
        self._delta_alpha = self._alpha_max - self._alpha_min
        self._delta_temp = self._temperature_max - self._temperature_min

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError("0 <= alpha <=1 1")
        if alpha <= self._alpha_min:
            return self._temperature_min
        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            return self._temperature_min + frac * self._delta_temp
        else:
            return self._temperature_max


class FixedTemperatureScaler(TemperatureScaler):
    def __init__(self, alpha_min, alpha_max, temperatures):
        if alpha_min < 0 or alpha_min > 1:
            raise RuntimeError("0 <= alpha_min <=1")
        if alpha_max < 0 or alpha_max > 1:
            raise RuntimeError("0 <= alpha_max <=1")
        if alpha_min >= alpha_max:
            raise RuntimeError("alpha_min must be < alpha_max")
        if float(temperatures[0]) <= 0 or float(temperatures[-1]) <= 0:
            raise RuntimeError("temperatures must be positive")

        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._temperatures = [float(t) for t in temperatures]
        self._delta_alpha = self._alpha_max - self._alpha_min
        self._diff_alpha = self._delta_alpha / float(len(self._temperatures) - 1)

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError("0 <= alpha <=1 1")
        if alpha <= self._alpha_min:
            return self._temperatures[0]
        elif alpha <= self._alpha_max:
            # without the round there is floating point error where
            # int(1.0) = 0
            index = int(round((alpha - self._alpha_min) / self._diff_alpha))
            return self._temperatures[index]
        else:
            return self._temperatures[-1]


class GeometricTemperatureScaler(TemperatureScaler):
    def __init__(self, alpha_min, alpha_max, temperature_min, temperature_max):
        if alpha_min < 0 or alpha_min > 1:
            raise RuntimeError("0 <= alpha_min <=1")
        if alpha_max < 0 or alpha_max > 1:
            raise RuntimeError("0 <= alpha_max <=1")
        if alpha_min >= alpha_max:
            raise RuntimeError("alpha_min must be < alpha_max")
        if temperature_min <= 0 or temperature_max <= 0:
            raise RuntimeError("temperatures must be positive")

        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._temperature_min = float(temperature_min)
        self._temperature_max = float(temperature_max)
        self._delta_alpha = self._alpha_max - self._alpha_min

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError("0 <= alpha <=1 1")
        if alpha <= self._alpha_min:
            return self._temperature_min
        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            delta = math.log(self._temperature_max) - math.log(self._temperature_min)
            return math.exp(delta * frac + math.log(self._temperature_min))
        else:
            return self._temperature_max


class REST2Scaler:
    def __init__(self, reference_temperature, temperature_scaler):
        """
        Scaler for REST2

        Parameters
        ----------
        reference_temperature: float
            this should be set to the temperature of the simulation, usually 300K
        temperature_scaler: float
            the psuedo temperature to adjust nonbonded and torsion parameters
            of REST2

        When performing REST2 simulations, typically the system temperature is kept
        fixed at 300K. Then the psuedo-temperature of non-solvent nonbonded and
        torsion interactions is adjusted based on the `temperature_scaler` parameter
        according to:
            scale = reference_temperature / temperature_scaler(alpha)

        """
        self.reference_temperature = reference_temperature
        self.scaler = temperature_scaler

    def __call__(self, alpha):
        return self.reference_temperature / self.scaler(alpha)


ExtraBondParam = namedtuple("ExtraBondParam", "i j length force_constant")
ExtraAngleParam = namedtuple("ExtraAngleParam", "i j k angle force_constant")
ExtraTorsParam = namedtuple("ExtraTorsParam", "i j k l phase energy multiplicity")


class System:
    def __init__(self, top_string, mdcrd_string):
        self._top_string = top_string
        self._mdcrd_string = mdcrd_string
        self.restraints = RestraintManager(self)

        self.temperature_scaler = None
        self._coordinates = None
        self._box_vectors = None
        self._n_atoms = None
        self._setup_coords()

        self._atom_names = None
        self._residue_names = None
        self._residue_numbers = None
        self._atom_index = None
        self._setup_indexing()

        self.extra_bonds = []
        self.extra_restricted_angles = []
        self.extra_torsions = []

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
            print(
                f"Could not find atom index for residue_number={residue_number} "
                f"and atom name={atom_name}."
            )
            raise

    def get_pdb_writer(self):
        return PDBWriter(
            range(1, len(self._atom_names) + 1),
            self._atom_names,
            self._residue_numbers,
            self._residue_names,
        )

    def add_extra_bond(self, i, j, length, force_constant):
        self.extra_bonds.append(
            ExtraBondParam(i=i, j=j, length=length, force_constant=force_constant)
        )

    def add_extra_angle(self, i, j, k, angle, force_constant):
        self.extra_restricted_angles.append(
            ExtraAngleParam(i=i, j=j, k=k, angle=angle, force_constant=force_constant)
        )

    def add_extra_torsion(self, i, j, k, l, phase, energy, multiplicity):
        self.extra_torsions.append(
            ExtraTorsParam(
                i=i,
                j=j,
                k=k,
                l=l,
                phase=phase,
                energy=energy,
                multiplicity=multiplicity,
            )
        )

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
        reader = CrdReader(self._mdcrd_string)
        self._coordinates = reader.get_coordinates()
        self._box_vectors = reader.get_box_vectors()
        self._n_atoms = self._coordinates.shape[0]


class CrdReader:
    def __init__(self, crd_string):
        self.crd_string = crd_string
        self._coords = None
        self._box_vectors = None
        self._read()

    def get_coordinates(self):
        return self._coords

    def get_box_vectors(self):
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
        coords = coords.reshape((n_atoms, 3))
        self._coords = coords
        self._box_vectors = box_vectors


class ParmTopReader:
    def __init__(self, top_string):
        self._top_string = top_string

    def get_atom_names(self):
        return self.get_parameter_block("%FLAG ATOM_NAME", chunksize=4)

    def get_residue_names(self):
        res_names = self.get_parameter_block("%FLAG RESIDUE_LABEL", chunksize=4)
        res_numbers = self.get_residue_numbers()
        return [res_names[i - 1] for i in res_numbers]

    def get_residue_numbers(self):
        n_atoms = int(self.get_parameter_block("%FLAG POINTERS", chunksize=8)[0])
        res_pointers = self.get_parameter_block("%FLAG RESIDUE_POINTER", chunksize=8)
        res_pointers = [int(p) for p in res_pointers]
        res_pointers.append(n_atoms + 1)
        residue_numbers = []
        for res_number, (start, end) in enumerate(
            zip(res_pointers[:-1], res_pointers[1:])
        ):
            residue_numbers.extend([res_number + 1] * (end - start))
        return residue_numbers

    def get_parameter_block(self, flag, chunksize):
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

    def get_bonds(self):
        # the amber bonds section contains a triple of integers for each bond:
        # i, j, type_index. We need i, j, but will end up ignoring type_index
        bond_items = self.get_parameter_block(
            "%FLAG BONDS_WITHOUT_HYDROGEN", chunksize=8
        )
        bond_items += self.get_parameter_block("%FLAG BONDS_INC_HYDROGEN", chunksize=8)
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
        return {
            (res_num, atom_name): atom_index
            for res_num, atom_name, atom_index in zip(
                residue_numbers, atom_names, atom_numbers
            )
        }


class RunOptions:
    def __setattr__(self, name, value):
        # open we only allow setting of these attributes
        # all others will raise an error, which catches
        # typos
        allowed_attributes = [
            "remove_com",
            "runner",
            "timesteps",
            "minimize_steps",
            "implicit_solvent_model",
            "cutoff",
            "use_big_timestep",
            "use_bigger_timestep",
            "use_amap",
            "amap_alpha_bias",
            "amap_beta_bias",
            "min_mc",
            "run_mc",
            "ccap",
            "ncap",
            "solvation",
            "enable_pme",
            "enable_pressure_coupling",
            "pressure",
            "pressure_coupling_update_steps",
            "pme_tolerance",
            "use_rest2",
            "rest2_scaler",
            "soluteDielectric",
            "solventDielectric",
            "implicitSolventSaltConc",
            "rdc_patcher",
        ]
        allowed_attributes += ["_{}".format(item) for item in allowed_attributes]
        if name not in allowed_attributes:
            raise ValueError(f"Attempted to set unknown attribute {name}")
        else:
            object.__setattr__(self, name, value)

    def __init__(self, solvation="implicit"):
        self._solvation = solvation
        if solvation == "implicit":
            self.implicit_solvent_model = "gbNeck2"
            self.cutoff = None
            self.enable_pme = False
            self.enable_pressure_coupling = False
        elif solvation == "explicit":
            self.implicit_solvent_model = "vacuum"
            self.cutoff = 0.9
            self.enable_pme = True
            self.enable_pressure_coupling = True
        else:
            raise RuntimeError(f"Unknown value {solvation} for solvation")
        self._runner = "openmm"
        self._timesteps = 5000
        self._minimize_steps = 1000
        self._use_big_timestep = False
        self._use_bigger_timestep = False
        self._use_amap = False
        self._amap_alpha_bias = 1.0
        self._amap_beta_bias = 1.0
        self._min_mc = None
        self._run_mc = None
        self._ccap = False
        self._ncap = False
        self._remove_com = True
        self._pressure = 1.0 * atmosphere
        self._pressure_coupling_update_steps = 25
        self._pme_tolerance = 0.0005
        self._use_rest2 = False
        self._rest2_scaler = None
        self._implicitSolventSaltConc = None
        self._solventDielectric = None
        self._soluteDielectric = None
        self._rdc_patcher = None

    # solvation is a read-only property that must be set
    # when the options are created
    @property
    def solvation(self):
        return self._solvation

    @property
    def enable_pme(self):
        return self._enable_pme

    @enable_pme.setter
    def enable_pme(self, new_value):
        if new_value not in [True, False]:
            raise ValueError("enable_pme must be True or False")
        if new_value:
            if self._solvation == "implicit":
                raise ValueError("Tried to set enable_pme=True with implicit solvation")
        self._enable_pme = new_value

    @property
    def pme_tolerance(self):
        return self._pme_tolerance

    @pme_tolerance.setter
    def pme_tolerance(self, new_value):
        if new_value <= 0:
            raise ValueError("pme_tolerance must be > 0")
        self._pme_tolerance = new_value

    @property
    def enable_pressure_coupling(self):
        return self._enable_pressure_coupling

    @enable_pressure_coupling.setter
    def enable_pressure_coupling(self, new_value):
        if new_value not in [True, False]:
            raise ValueError("enable_pressure_coupling must be True or False")
        if new_value:
            if self._solvation == "implicit":
                raise ValueError(
                    "Tried to set enable_pressure_coupling=True with "
                    "implicit solvation"
                )
        self._enable_pressure_coupling = new_value

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, new_value):
        if new_value <= 0:
            raise ValueError("pressure must be > 0")
        self._pressure = new_value

    @property
    def pressure_coupling_update_steps(self):
        return self._pressure_coupling_update_steps

    @pressure_coupling_update_steps.setter
    def pressure_coupling_update_steps(self, new_value):
        if new_value <= 0:
            raise ValueError("pressure_coupling_update_steps must be > 0")
        self._pressure_coupling_update_steps = new_value

    @property
    def use_rest2(self):
        return self._use_rest2

    @use_rest2.setter
    def use_rest2(self, new_value):
        self._use_rest2 = new_value

    @property
    def rest2_scaler(self):
        return self._rest2_scaler

    @rest2_scaler.setter
    def rest2_scaler(self, new_value):
        self._rest2_scaler = new_value

    @property
    def min_mc(self):
        return self._min_mc

    @min_mc.setter
    def min_mc(self, new_value):
        self._min_mc = new_value

    @property
    def run_mc(self):
        return self._run_mc

    @run_mc.setter
    def run_mc(self, new_value):
        self._run_mc = new_value

    @property
    def remove_com(self):
        return self._remove_com

    @remove_com.setter
    def remove_com(self, new_value):
        self._remove_com = bool(new_value)

    @property
    def runner(self):
        return self._runner

    @runner.setter
    def runner(self, value):
        if value not in ["openmm", "fake_runner"]:
            raise RuntimeError(f"unknown value for runner {value}")
        self._runner = value

    @property
    def implicitSolventSaltConc(self):
        return self._implicitSolventSaltConc

    @implicitSolventSaltConc.setter
    def implicitSolventSaltConc(self, value):
        value = float(value)
        if value <= 0:
            raise RuntimeError("implicitSolventSaltConc must be > 0")
        self._implicitSolventSaltConc = value

    @property
    def solventDielectric(self):
        return self._solventDielectric

    @solventDielectric.setter
    def solventDielectric(self, value):
        value = float(value)
        if value <= 0:
            raise RuntimeError("solventDielectric must be > 0")
        self._solventDielectric = value

    @property
    def soluteDielectric(self):
        return self._soluteDielectric

    @soluteDielectric.setter
    def soluteDielectric(self, value):
        value = float(value)
        if value <= 0:
            raise RuntimeError("soluteDielectric must be > 0")
        self._soluteDielectric = value

    @property
    def timesteps(self):
        return self._timesteps

    @timesteps.setter
    def timesteps(self, value):
        value = int(value)
        if value <= 0:
            raise RuntimeError("timesteps must be > 0")
        self._timesteps = value

    @property
    def minimize_steps(self):
        return self._minimize_steps

    @minimize_steps.setter
    def minimize_steps(self, value):
        value = int(value)
        if value <= 0:
            raise RuntimeError("minimize_steps must be > 0")
        self._minimize_steps = value

    @property
    def implicit_solvent_model(self):
        return self._implicit_solvent_model

    @implicit_solvent_model.setter
    def implicit_solvent_model(self, value):
        if value not in [None, "obc", "gbNeck", "gbNeck2", "vacuum"]:
            raise RuntimeError(f"unknown value for implicit solvent model {value}")
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
                raise RuntimeError("cutoff must be > 0")
            self._cutoff = value

    @property
    def use_big_timestep(self):
        return self._use_big_timestep

    @use_big_timestep.setter
    def use_big_timestep(self, value):
        self._use_big_timestep = bool(value)

    @property
    def use_bigger_timestep(self):
        return self._use_bigger_timestep

    @use_bigger_timestep.setter
    def use_bigger_timestep(self, value):
        self._use_bigger_timestep = bool(value)

    @property
    def use_amap(self):
        return self._use_amap

    @use_amap.setter
    def use_amap(self, value):
        if not value in [True, False]:
            raise ValueError("use_amap must be True or False")
        if value and self._solvation == "explicit":
            raise ValueError("use_amap can not be set with explicit solvent")
        self._use_amap = bool(value)

    @property
    def ccap(self):
        return self._ccap

    @ccap.setter
    def ccap(self, value):
        self._ccap = bool(value)

    @property
    def ncap(self):
        return self._ncap

    @ncap.setter
    def ncap(self, value):
        self._ncap = bool(value)

    @property
    def amap_alpha_bias(self):
        return self._amap_alpha_bias

    @amap_alpha_bias.setter
    def amap_alpha_bias(self, value):
        if value < 0:
            raise RuntimeError("amap_alpha_bias < 0")
        self._amap_alpha_bias = value

    @property
    def amap_beta_bias(self):
        return self._amap_beta_bias

    @amap_beta_bias.setter
    def amap_beta_bias(self, value):
        if value < 0:
            raise RuntimeError("amap_beta_bias < 0")
        self._amap_beta_bias = value

    @property
    def rdc_patcher(self):
        return self._rdc_patcher
    
    @rdc_patcher.setter
    def rdc_patcher(self, value):
        self._rdc_patcher = value

    def sanity_check(self):
        if self._solvation == "implicit":
            if self._enable_pme:
                raise ValueError("enable_pme == True for implicit solvation simulation")
            if self._enable_pressure_coupling:
                raise ValueError(
                    "enable_pressure_coupling == True for implicit"
                    "solvation simulation"
                )

        if self._solvation == "explicit":
            if not self._implicit_solvent_model == "vacuum":
                raise ValueError(
                    'implicit_solvent_model != "vacuum" for explicit '
                    "solvation simulation"
                )
            if self._use_amap == True:
                raise ValueError('use_amap cannot be set with explicit solvent')
