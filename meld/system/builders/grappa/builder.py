#
# Copyright 2023 The MELD Contributors
# All rights reserved
#

"""
Module to build a System using the Grappa force field.
"""

import logging
from typing import Optional
import numpy as np
import openmm as mm
from openmm import app
from openmm import unit as u

from meld.system.builders.spec import SystemSpec
from meld.system.builders.grappa.options import GrappaOptions

logger = logging.getLogger(__name__)

try:
    from grappa import OpenmmGrappa # type: ignore
except ImportError:
    logger.error("Could not import grappa. Please ensure it is installed.")
    raise


class GrappaSystemBuilder:
    """
    Class to handle building an OpenMM System using the Grappa force field.
    """

    options: GrappaOptions

    def __init__(self, options: GrappaOptions):
        """
        Initialize a GrappaSystemBuilder.

        Args:
            options: Options for building the system with Grappa.
        """
        self.options = options
        logger.info("GrappaSystemBuilder initialized.")

    def build_system(
        self,
        topology: app.Topology,
        positions: u.Quantity,
        box_vectors: Optional[u.Quantity] = None,
    ) -> SystemSpec:
        """
        Build the OpenMM system using the Grappa force field.

        Args:
            topology: OpenMM Topology object.
            positions: Initial positions for the system.
            box_vectors: Optional periodic box vectors.

        Returns:
            A SystemSpec object.
        """
        logger.info("Building system with Grappa force field.")

        # Load base force field
        logger.info(f"Loading base force field files: {self.options.base_forcefield_files}")
        base_ff = app.ForceField(*self.options.base_forcefield_files)

        hydrogen_mass, constraint_type = _get_hydrogen_mass_and_constraints(
            self.options.use_big_timestep, self.options.use_bigger_timestep
        )

        # Create initial system with base force field
        logger.info("Creating initial system with base force field.")


        if self.options.cutoff is not None:
            nonbonded_method = app.PME
            # Ensure cutoff is a Quantity
            nonbonded_cutoff = float(self.options.cutoff)
            if isinstance(self.options.cutoff, float):
                nonbonded_cutoff = self.options.cutoff * u.nanometer
            else:
                nonbonded_cutoff = self.options.cutoff
            logger.info(f"Using PME with cutoff {nonbonded_cutoff} nm.")
        else:
            nonbonded_method = app.CutoffNonPeriodic
            nonbonded_cutoff = 1.0 * u.nanometer
            logger.info("Using CutoffNonPeriodic for nonbonded interactions.")

        create_system_kwargs = dict(
            topology=topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=nonbonded_cutoff,
            constraints=constraint_type,
            hydrogenMass=hydrogen_mass,
            removeCMMotion=self.options.remove_com,
        )

        system = base_ff.createSystem(**create_system_kwargs)

        #=======================================================
        print("nonbonded_method:", nonbonded_method)
        if self.options.cutoff is not None:
            print("nonbonded_cutoff:", nonbonded_cutoff, type(nonbonded_cutoff))
        #=====================================================

        # Initialize Grappa force field
        logger.info(f"Initializing Grappa with model tag: {self.options.grappa_model_tag}")
        try:
            grappa_ff = OpenmmGrappa.from_tag(self.options.grappa_model_tag)
        except Exception as e:
            logger.error(f"Failed to load Grappa model with tag '{self.options.grappa_model_tag}': {e}")
            raise RuntimeError(f"Grappa model loading failed. Ensure the tag is correct and model is accessible.") from e


        # Parametrize the system using Grappa
        logger.info("Parametrizing system with Grappa.")
        try:
            system = grappa_ff.parametrize_system(system, topology)
            logger.info("System parametrized successfully by Grappa.")
        except Exception as e:
            logger.error(f"Grappa parametrization failed: {e}")
            # It might be useful to log more details about the system and topology if possible
            raise RuntimeError("Grappa parametrization step failed.") from e


        # Create integrator
        integrator = _create_integrator(
            self.options.default_temperature,
            self.options.use_big_timestep,
            self.options.use_bigger_timestep,
        )
        logger.info(f"Integrator created with temperature {self.options.default_temperature}K.")

        # Prepare coordinates and velocities
        coords_nm = np.array([list(pos.value_in_unit(u.nanometer)) for pos in positions])
        vels_nm_ps = np.zeros_like(coords_nm) * (u.nanometer / u.picosecond)

        #============== debugging lines ======================================
        num_particles = system.getNumParticles()
        print("Num particles in system:", num_particles)
        print("Num coordinates:", coords_nm.shape[0])
        print("Any NaNs in coords?", np.isnan(coords_nm).any())
        if num_particles != coords_nm.shape[0]:
            raise ValueError(f"Atom count mismatch; system has {num_particles} particles, but coordinates have {coords_nm.shape[0]}")
        if np.isnan(coords_nm).any():
            nan_rows = np.where(np.isnan(coords_nm).anys(axis=1))[0]
            raise ValueError(f"NaN detected in coordinates at rows: {nan_rows}")
        
        # Optionally, check velocities as well
        if np.isnan(vels_nm_ps).any():
            raise ValueError("NaN detected in velocities!")
        #=====================================================================

        # Handle box vectors
        box_nm = None
        if box_vectors is not None:
            box_nm = box_vectors.value_in_unit(u.nanometer)
            # Assuming orthorhombic box for simplicity, matching AmberSystemBuilder
            if isinstance(box_nm, np.ndarray) and box_nm.shape == (3,3): # it is a matrix
                 # check its diagonal
                 if not (box_nm[0,1] == 0 and box_nm[0,2] == 0 and \
                         box_nm[1,0] == 0 and box_nm[1,2] == 0 and \
                         box_nm[2,0] == 0 and box_nm[2,1] == 0 ):
                     logger.warning("Box vectors are not diagonal (not orthorhombic). MELD might not handle this correctly.")
                 box_nm = np.array([box_nm[0,0], box_nm[1,1], box_nm[2,2]])


        logger.info("SystemSpec creation complete.")
        return SystemSpec(
            solvation=self.options.solvation_type,
            system=system,
            topology=topology,
            integrator=integrator,
            barostat=None,  # Grappa examples do not typically include a barostat by default
            coordinates=coords_nm,
            velocities=vels_nm_ps,
            box_vectors=box_nm,
            builder_info={
                "builder": "grappa",
                "grappa_model_tag": self.options.grappa_model_tag,
                "base_forcefield_files": self.options.base_forcefield_files,
            },
        )


def _get_hydrogen_mass_and_constraints(use_big_timestep: bool, use_bigger_timestep: bool):
    if use_big_timestep:
        logger.info("Enabling hydrogen mass=3 Da, constraining all bonds.")
        constraint_type = app.AllBonds
        hydrogen_mass = 3.0 * u.dalton # OpenMM uses daltons for hydrogenMass
    elif use_bigger_timestep:
        logger.info("Enabling hydrogen mass=4 Da, constraining all bonds.")
        constraint_type = app.AllBonds
        hydrogen_mass = 4.0 * u.dalton
    else:
        logger.info("Using default hydrogen mass, constraining bonds with hydrogen.")
        constraint_type = app.HBonds
        hydrogen_mass = None # Defaults to actual hydrogen mass
    return hydrogen_mass, constraint_type


def _create_integrator(temperature: float, use_big_timestep: bool, use_bigger_timestep: bool):
    if use_big_timestep:
        logger.info("Creating Langevin integrator with 3.0 fs timestep.")
        timestep = 3.0 * u.femtoseconds
    elif use_bigger_timestep:
        logger.info("Creating Langevin integrator with 4.0 fs timestep.") # Adjusted from Amber's 4.5fs
        timestep = 4.0 * u.femtoseconds
    else:
        logger.info("Creating Langevin integrator with 2.0 fs timestep.")
        timestep = 2.0 * u.femtoseconds
    
    # Standard friction coefficient
    friction = 1.0 / u.picosecond
    # Temperature is already in Kelvin from GrappaOptions
    return mm.LangevinIntegrator(temperature * u.kelvin, friction, timestep)
