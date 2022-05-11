#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to build a System from Martini OpenMM system
"""

from meld import util
from ..spec import SystemSpec
from ... import indexing
from . import subsystem
from . import amap

from typing import List, Optional
from dataclasses import dataclass
from functools import partial
import subprocess
import logging

logger = logging.getLogger(__name__)

from openmm import app  # type: ignore
from openmm.app import forcefield as ff  # type: ignore
import openmm as mm  # type: ignore
from openmm import unit as u  # type: ignore
import numpy as np  # type: ignore

import martini_openmm as martini

# Need to expand options for all use cases
@partial(dataclass, frozen=True)
class MartiniOptions:
    default_temperature: float = 300.0 * u.kelvin
    epsilon_r: float = 15.0 ## What is a reasonable value for this?
    solvation: str = "implicit"
    defines_file: Optional[str] = "defines.txt"
    enable_pressure_coupling: bool = False
    pressure: float = 1.01325 * u.bar
    pressure_coupling_update_steps: int = 25
    cutoff: Optional[float] = 1.1
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
            with open(self.options.defines) as def_file:
                for line in def_file:
                    line = line.strip()
                    defines[line] = True
        except FileNotFoundError:
            pass

        top = martini.MartiniTopfile(
            topfile,
            periodicBoxVectors = box_vectors,
            defines = defines,
            epsilon_r = self.options.epsilon_r
        )

        system, barostat = _create_openmm_system(
            top,
            self.options.solvation_type,
            self.options.cutoff,
            self.options.enable_pressure_coupling,
            self.options.pressure,
            self.options.pressure_coupling_update_steps,
            self.options.remove_com,
            self.options.default_temperature,
        )


        integrator = _create_integrator(
            self.options.default_temperature,
            self.options.use_big_timestep,
            self.options.use_bigger_timestep,
        )

        coords = conf.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        try:
            vels = conf.getPositions(asNumpy=True)
        except AttributeError:
            print("WARNING: No velocities found, setting to zero")
            vels = np.zeros_like(coords)
        try:
            box = conf.getPeriodicBoxVectors()(asNumpy=True)
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
            self.options.solvation,
            system,
            top,
            integrator,
            barostat,
            coords,
            vels,
            box,
            {
                "solvation": self.options.solvation,
                "builder": "martini3",
            },
        )


def _create_openmm_system(
    parm_object,
    solvation_type,
    cutoff,
    implicit_solvent,
    enable_pressure_coupling,
    pressure,
    pressure_coupling_update_steps,
    remove_com,
    default_temperature,
    implicitSolventSaltConc,
):
    if solvation_type == "implicit":
        logger.info("Creating implicit solvent system")
        system = _create_openmm_system_implicit(
            parm_object,
            cutoff,
            implicit_solvent,
            remove_com,
            implicitSolventSaltConc,
            soluteDielectric,
            solventDielectric,
        )
        baro = None
    elif solvation_type == "explicit":
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
    else:
        raise ValueError(f"unknown value for solvation_type: {solvation_type}")

    return system, baro


def _create_openmm_system_implicit(
    parm_object,
    cutoff,
    implicit_solvent,
    remove_com,
    implicitSolventSaltConc,
    soluteDielectric,
    solventDielectric,
):
    if cutoff is None:
        logger.info("Using no cutoff")
        cutoff_type = ff.NoCutoff
        cutoff_dist = 999.0
    else:
        logger.info(f"Using a cutoff of {cutoff}")
        cutoff_type = ff.CutoffNonPeriodic
        cutoff_dist = cutoff

    if implicit_solvent == "obc":
        logger.info('Using "OBC" implicit solvent')
        implicit_type = app.OBC2
    elif implicit_solvent == "gbNeck":
        logger.info('Using "gbNeck" implicit solvent')
        implicit_type = app.GBn
    elif implicit_solvent == "gbNeck2":
        logger.info('Using "gbNeck2" implicit solvent')
        implicit_type = app.GBn2
    elif implicit_solvent == "vacuum" or implicit_solvent is None:
        logger.info("Using vacuum instead of implicit solvent")
        implicit_type = None
    else:
        RuntimeError("Should never get here")

    if implicitSolventSaltConc is None:
        implicitSolventSaltConc = 0.0
    if soluteDielectric is None:
        soluteDielectric = 1.0
    if solventDielectric is None:
        solventDielectric = 78.5

    sys = parm_object.createSystem(
        nonbondedMethod=cutoff_type,
        nonbondedCutoff=cutoff_dist,
        implicitSolvent=implicit_type,
        removeCMMotion=remove_com,
        implicitSolventSaltConc=implicitSolventSaltConc,
        soluteDielectric=soluteDielectric,
        solventDielectric=solventDielectric,
    )
    return sys


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

    s = parm_object.createSystem(
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


def _create_integrator(temperature):
    logger.info("Creating integrator with 2.0 fs timestep")
    timestep = 2.0 * u.femtosecond
    return mm.LangevinIntegrator(temperature * u.kelvin, 1.0 / u.picosecond, timestep)
