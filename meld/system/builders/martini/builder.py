#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to build a System from Martini OpenMM system
"""

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import app  # type: ignore
from openmm import unit as u  # type: ignore

from meld.system.builders.spec import SystemSpec

logger = logging.getLogger(__name__)


try:
    import martini_openmm as martini  # type: ignore
except ImportError:
    print()
    print("You are trying to use the martini builder functionality.")
    print("This requires the installation of the optional martini_openmm")
    print("dependency, but this could not be imported.")
    raise


# Need to expand options for all use cases
@partial(dataclass, frozen=True)
class MartiniOptions:
    default_temperature: u.Quantity = field(default_factory=lambda: 300.0 * u.kelvin)
    timestep: u.Quantity = field(default_factory=lambda: 20.0 * u.femtoseconds)
    epsilon_r: float = 15.0
    defines_file: Optional[str] = "defines.txt"
    enable_pressure_coupling: bool = False
    pressure: float = 1.01325 * u.bar
    pressure_coupling_update_steps: int = 25
    cutoff: Optional[float] = 1.1 * u.nanometer
    remove_com: bool = True

    def __post_init__(self):
        # Sanity checks for implicit and explicit solvent

        if isinstance(self.default_temperature, u.Quantity):
            object.__setattr__(
                self,
                "default_temperature",
                self.default_temperature.value_in_unit(u.kelvin),
            )
        if self.default_temperature < 0:
            raise ValueError(f"default_temperature must be >= 0")

        if isinstance(self.pressure, u.Quantity):
            object.__setattr__(self, "pressure", self.pressure.value_in_unit(u.bar))
        if self.pressure < 0:
            raise ValueError(f"pressure must be >= 0")


class MartiniSystemBuilder:
    r"""
    Class to handle building a System from SubSystems.
    """

    options: MartiniOptions

    def __init__(self, options):
        """
        Initialize a SystemBuilder

        Args:
            options: Options for martini

        """
        self.options = options

    def build_system(
        self,
        topfile: str,
        grofile: str,
    ) -> SystemSpec:
        """
        Build the system from AmberSubSystems
        """
        conf = app.GromacsGroFile(grofile)
        box_vectors = conf.getPeriodicBoxVectors()

        # get any defines
        defines = {}
        try:
            assert self.options.defines_file is not None
            with open(self.options.defines_file) as def_file:
                for line in def_file:
                    line = line.strip()
                    defines[line] = True
        except FileNotFoundError:
            pass

        top = martini.MartiniTopFile(
            topfile,
            periodicBoxVectors=box_vectors,
            defines=defines,
            epsilon_r=self.options.epsilon_r,
        )

        topology = top.topology

        system, barostat = _create_openmm_system(
            top,
            self.options.cutoff,
            self.options.enable_pressure_coupling,
            self.options.pressure,
            self.options.pressure_coupling_update_steps,
            self.options.remove_com,
            self.options.default_temperature,
        )

        integrator = _create_integrator(
            self.options.default_temperature,
            self.options.timestep,
        )

        coords = conf.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        try:
            vels = conf.getVelocities(
                asNumpy=True
            )  # Gromacs files do not contain velocity information; may be better to set this using Maxwell-Boltzmann
        except AttributeError:
            print("WARNING: No velocities found, setting to zero")
            vels = np.zeros_like(coords)
        try:
            box = conf.getPeriodicBoxVectors()
            # We only support orthorhombic boxes
            box_a = box[0][0].value_in_unit(u.nanometer)
            assert box[0][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            assert box[0][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            box_b = box[1][1].value_in_unit(u.nanometer)
            assert box[1][0] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            assert box[1][2] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            box_c = box[2][2].value_in_unit(u.nanometer)
            assert box[2][0] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            assert box[2][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            box = np.array([box_a, box_b, box_c])
        except AttributeError:
            box = None

        return SystemSpec(
            "explicit",
            system,
            topology,
            integrator,
            barostat,
            coords,
            vels,
            box,
            {
                "builder": "martini3",
            },
        )


def _create_openmm_system(
    parm_object,
    cutoff,
    enable_pressure_coupling,
    pressure,
    pressure_coupling_update_steps,
    remove_com,
    default_temperature,
):
    logger.info("Creating explicit solvent system")
    system, baro = _create_openmm_system_explicit(
        parm_object,
        cutoff,
        enable_pressure_coupling,
        pressure,
        pressure_coupling_update_steps,
        remove_com,
        default_temperature,
    )
    return system, baro


def _create_openmm_system_explicit(
    parm_object,
    cutoff,
    enable_pressure_coupling,
    pressure,
    pressure_couping_update_steps,
    remove_com,
    default_temperature,
):
    if cutoff is None:
        raise ValueError("cutoff must be set for explicit solvent, but got None")

    s = parm_object.create_system(
        nonbonded_cutoff=cutoff,
        remove_com_motion=remove_com,
    )

    baro = None
    if enable_pressure_coupling:
        logger.info("Enabling pressure coupling")
        logger.info(f"Pressure is {pressure}")
        logger.info(
            f"Volume moves attempted every {pressure_couping_update_steps} steps"
        )
        baro = mm.MonteCarloBarostat(
            pressure, default_temperature, pressure_couping_update_steps
        )
        s.addForce(baro)

    return s, baro


def _create_integrator(temperature, timestep):
    logger.info(f"Creating integrator with {timestep} timestep")
    return mm.LangevinIntegrator(temperature * u.kelvin, 1.0 / u.picosecond, timestep)
