#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import logging
import math
import random
from tkinter import E
from typing import Dict, List, Optional

import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import app  # type: ignore
from openmm import unit as u  # type: ignore

import os
from genericpath import exists

try:
    from gamd.GamdLogger import GamdLogger   # type: ignore
except:
    pass

from meld import interfaces
from meld.runner import transform
from meld.system import options, restraints
from meld.system.state import SystemState
from meld.util import log_timing

from meld.vault import ENERGY_GROUPS

logger = logging.getLogger(__name__)


GAS_CONSTANT = 8.314e-3


class OpenMMRunner(interfaces.IRunner):
    _always_on_restraints: List[restraints.Restraint]
    _selectable_collections: List[restraints.SelectivelyActiveCollection]
    _options: options.RunOptions
    _simulation: app.Simulation
    _omm_system: mm.System
    _topology: app.Topology
    _integrator: mm.LangevinIntegrator
    _barostat: mm.MonteCarloBarostat
    _timestep: int
    _initialized: bool
    _alpha: float
    _temperature: float
    _transformers: List[transform.TransformerBase]
    _extra_bonds: List[interfaces.ExtraBondParam]
    _extra_restricted_angles: List[interfaces.ExtraAngleParam]
    _extra_torsions: List[interfaces.ExtraTorsParam]

    def __init__(
        self,
        meld_system: interfaces.ISystem,
        options: options.RunOptions,
        communicator: Optional[interfaces.ICommunicator] = None,
        platform: Optional[str] = None,
    ):
        self._omm_system = meld_system.omm_system
        self._topology = meld_system.topology
        self._integrator = meld_system.integrator
        self._barostat = meld_system.barostat
        self._solvation = meld_system.solvation
        self.builder_info = meld_system.builder_info

        # Default to CUDA platform
        platform = platform if platform else "CUDA"
        self.platform = platform

        if communicator:
            # Only need to figure out device id for CUDA
            if platform == "CUDA":
                self._device_id = communicator.negotiate_device_id()
            self._rank: Optional[int] = communicator.rank
        else:
            self._device_id = 0
            self._rank = None

        if meld_system.temperature_scaler is None:
            raise RuntimeError("system does not have temparture_scaler set")
        else:
            self.temperature_scaler = meld_system.temperature_scaler

        self._always_on_restraints = meld_system.restraints.always_active
        self._selectable_collections = (
            meld_system.restraints.selectively_active_collections
        )
        self._options = options
        self._timestep = 0
        self._initialized = False
        self._alpha = 0.0
        self._transformers: List[transform.TransformerBase] = []
        self._extra_bonds = meld_system.extra_bonds
        self._extra_restricted_angles = meld_system.extra_restricted_angles
        self._extra_torsions = meld_system.extra_torsions
        self._parameter_manager = meld_system.param_sampler
        self._mapper = meld_system.mapper
        self._density = meld_system.density

    def prepare_for_timestep(
        self, state: interfaces.IState, alpha: float, timestep: int
    ):
        self._alpha = alpha
        self._timestep = timestep
        assert self.temperature_scaler is not None
        self._temperature = self.temperature_scaler(alpha)
        self._initialize_simulation(state)

    @log_timing(logger)
    def minimize_then_run(self, state: interfaces.IState) -> interfaces.IState:
        return self._run(state, minimize=True)

    @log_timing(logger)
    def run(self, state: interfaces.IState) -> interfaces.IState:
        return self._run(state, minimize=False)

    def get_energy(self, state: interfaces.IState) -> float:
        # update all of the transformers
        self._transformers_update(state)

        # set the coordinates
        coordinates = u.Quantity(state.positions, u.nanometer)
        self._simulation.context.setPositions(coordinates)

        # set the box vectors
        if self._solvation == "explicit":
            box_vector = state.box_vector
            self._simulation.context.setPeriodicBoxVectors(
                [box_vector[0], 0.0, 0.0],
                [0.0, box_vector[1], 0.0],
                [0.0, 0.0, box_vector[2]],
            )

        # set the rdc alignments
        self._set_alignments(state)

        # get the energy
        snapshot = self._simulation.context.getState(getEnergy=True)
        e_potential = snapshot.getPotentialEnergy()
        e_potential = (
            e_potential.value_in_unit(u.kilojoule / u.mole)
            / GAS_CONSTANT
            / self._temperature
        )

        # get the log_prior for parameters being sampled
        log_prior = self._parameter_manager.log_prior(state.parameters, self._alpha)

        return e_potential - log_prior

    def get_group_energies(self, state: interfaces.IState) -> np.ndarray:
        # update all of the transformers
        self._transformers_update(state)

        # set the coordinates
        coordinates = u.Quantity(state.positions, u.nanometer)
        self._simulation.context.setPositions(coordinates)

        # set the box vectors
        if self._solvation == "explicit":
            box_vector = state.box_vector
            self._simulation.context.setPeriodicBoxVectors(
                [box_vector[0], 0.0, 0.0],
                [0.0, box_vector[1], 0.0],
                [0.0, 0.0, box_vector[2]],
            )

        # set the rdc alignments
        self._set_alignments(state)

        group_energies = np.zeros(ENERGY_GROUPS)

        for i in range(ENERGY_GROUPS - 1):
            snapshot = self._simulation.context.getState(getEnergy=True, groups={i})
            e_potential = snapshot.getPotentialEnergy()
            e_potential = (
                e_potential.value_in_unit(u.kilojoule / u.mole)
                / GAS_CONSTANT
                / self._temperature
            )
            group_energies[i] = e_potential

        log_prior = self._parameter_manager.log_prior(state.parameters, self._alpha)
        group_energies[-1] = -log_prior

        return group_energies

    def _get_forces(self, state: interfaces.IState) -> np.ndarray:
        # update all of the transformers
        self._transformers_update(state)

        # set the coordinates
        coordinates = u.Quantity(state.positions, u.nanometer)
        self._simulation.context.setPositions(coordinates)

        # set the box vectors
        if self._solvation == "explicit":
            box_vector = state.box_vector
            self._simulation.context.setPeriodicBoxVectors(
                [box_vector[0], 0.0, 0.0],
                [0.0, box_vector[1], 0.0],
                [0.0, 0.0, box_vector[2]],
            )

        # set the rdc alignments
        self._set_alignments(state)

        # get the forces
        snapshot = self._simulation.context.getState(getForces=True)
        forces = snapshot.getForces(asNumpy=True).value_in_unit(
            u.kilojoule / u.mole / u.nanometer
        )
        return forces

    def _get_max_force_norm(self, state: interfaces.IState) -> float:
        forces = self._get_forces(state)
        return np.max(np.linalg.norm(forces, axis=1))

    def _initialize_simulation(self, state: interfaces.IState) -> None:
        if self._initialized:
            # update temperature and pressure
            if self.builder_info.get("has_alignments", False):
                self._simulation.integrator.setGlobalVariableByName(
                    "kT", self._temperature * GAS_CONSTANT
                )
            elif hasattr(self._integrator, "setTemperature"):           
                self._integrator.setTemperature(self._temperature)
            elif self._options.enable_gamd == True:
                thermal_energy = (
                    self._temperature * u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA
                )
                self._simulation.integrator.setGlobalVariableByName(
                    "thermal_energy", thermal_energy
                )

            if self._barostat:
                self._simulation.context.setParameter(
                    self._barostat.Temperature(), self._temperature
                )

            # update all of the system transformers
            self._transformers_update(state)

        else:
            # we need to set the whole thing from scratch
            self._initialized = True

            _add_extras(
                self._omm_system,
                self._extra_bonds,
                self._extra_restricted_angles,
                self._extra_torsions,
            )

            # setup the transformers
            self._transformers_setup()
            if len(self._always_on_restraints) > 0:
                print("Not all always on restraints were handled.")
                for remaining_always_on in self._always_on_restraints:
                    print("\t", remaining_always_on)
                raise RuntimeError("Not all always on restraints were handled.")

            if len(self._selectable_collections) > 0:
                print("Not all selectable restraints were handled.")
                for remaining_selectable in self._selectable_collections:
                    print("\t", remaining_selectable)
                raise RuntimeError("Not all selectable restraints were handled.")

            self._omm_system = self._transformers_add_interactions(
                state, self._omm_system, self._topology
            )
            self._transformers_finalize(state, self._omm_system, self._topology)

            # setup the platform, CUDA by default and Reference for testing
            properties: Dict[str, str]
            if self.platform == "Reference":
                logger.info("Using Reference platform.")
                platform = mm.Platform.getPlatformByName("Reference")
                properties = {}
            elif self.platform == "CPU":
                logger.info("Using CPU platform.")
                platform = mm.Platform.getPlatformByName("CPU")
                properties = {}
            elif self.platform == "CUDA":
                logger.info("Using CUDA platform.")
                platform = mm.Platform.getPlatformByName("CUDA")
                # The plugin currently requires that we use nvcc, as
                # nvrtc is not able to compile code that uses the cub
                # library, which we use in the plugin.
                # We can force the use of nvcc by setting CudaCompiler.
                # We set it to the default value, which will reflect the
                # OPENMM_CUDA_COMPILER environmnet variable if set.
                compiler = platform.getPropertyDefaultValue("CudaCompiler")
                logger.debug(f"Using CUDA compiler {compiler}.")
                properties = {
                    "CudaDeviceIndex": str(self._device_id),
                    "CudaPrecision": "mixed",
                    "CudaCompiler": compiler,
                }
            else:
                raise RuntimeError(f"Unknown platform {self.platform}.")
            # forcegroups = self._forcegroupify(sys)
            # create the simulation object
            self._simulation = _create_openmm_simulation(
                self._topology, self._omm_system, self._integrator, platform, properties
            )
            # forcegroups=self._forcegroupify(sys)
            # self._simulation = _create_openmm_simulation(
            #     prmtop.topology, sys, self._integrator, platform, properties
            # )
            self._transformers_update(state)

    def _forcegroupify(self, system):
        forcegroups = {}
        for i in range(system.getNumForces()):
            # logger.info(f"{i}th force \n")
            force = system.getForce(i)
            force.setForceGroup(i)
            forcegroups[force] = i
        return forcegroups

    def _transformers_setup(self) -> None:
        trans_types = [
            transform.ConfinementRestraintTransformer,
            transform.CartesianRestraintTransformer,
            transform.YZCartesianTransformer,
            transform.COMRestraintTransformer,
            transform.AbsoluteCOMRestraintTransformer,
            transform.MeldRestraintTransformer,
            transform.REST2Transformer,
        ]

        for tt in trans_types:
            trans = tt(
                self._parameter_manager,
                self._mapper,
                self._density,
                self.builder_info,
                self._options,
                self._always_on_restraints,
                self._selectable_collections,
            )
            self._transformers.append(trans)

    def _transformers_add_interactions(
        self, state: interfaces.IState, sys, topol
    ) -> mm.System:
        for t in self._transformers:
            sys = t.add_interactions(state, sys, topol)
        return sys

    def _transformers_finalize(self, state: interfaces.IState, sys, topol) -> None:
        for t in self._transformers:
            t.finalize(state, sys, topol)

    def _transformers_update(self, state: interfaces.IState) -> None:
        for t in self._transformers:
            t.update(state, self._simulation, self._alpha, self._timestep)

    def _run_min_mc(self, state: interfaces.IState) -> interfaces.IState:
        if self._options.min_mc is not None:
            logger.info("Running MCMC before minimization.")
            logger.info(f"Starting energy {self.get_energy(state):.3f}")
            logger.info(
                f"Starting maximum force norm {self._get_max_force_norm(state):.3f}"
            )
            state.energy = self.get_energy(state)
            state = self._options.min_mc.update(state, self)
            logger.info(f"Ending energy {self.get_energy(state):.3f}")
            logger.info(
                f"Ending maximum force norm {self._get_max_force_norm(state):.3f}"
            )
        return state

    def _run_mc(self, state: interfaces.IState) -> interfaces.IState:
        if self._options.run_mc is not None:
            logger.info("Running MCMC.")
            logger.debug(f"Starting energy {self.get_energy(state):.3f}")
            state.energy = self.get_energy(state)
            state = self._options.run_mc.update(state, self)
            logger.debug(f"Ending energy {self.get_energy(state):.3f}")
        return state

    def _run(self, state: interfaces.IState, minimize: bool) -> interfaces.IState:
        # update the transformers to account for sampled parameters
        # stored in the state
        self._transformers_update(state)

        assert abs(state.alpha - self._alpha) < 1e-6

        # Run Monte Carlo position updates
        if minimize:
            state = self._run_min_mc(state)
        else:
            state = self._run_mc(state)

        # Run Monte Carlo parameter updates
        state = self._run_param_mc(state)

        # Run Monte Carlo mapper updates
        state = self._run_mapper_mc(state)

        coordinates = u.Quantity(state.positions, u.nanometer)
        velocities = u.Quantity(state.velocities, u.nanometer / u.picosecond)
        box_vectors = u.Quantity(state.box_vector, u.nanometer)

        # set the positions
        self._simulation.context.setPositions(coordinates)

        # if explicit solvent, then set the box vectors
        if self._solvation == "explicit":
            self._simulation.context.setPeriodicBoxVectors(
                [box_vectors[0].value_in_unit(u.nanometer), 0.0, 0.0],
                [0.0, box_vectors[1].value_in_unit(u.nanometer), 0.0],
                [0.0, 0.0, box_vectors[2].value_in_unit(u.nanometer)],
            )

        # set the rdc alignments
        self._set_alignments(state)

        # run energy minimization
        if minimize:
            logger.info("Running minimization.")
            pre_state = self._simulation.context.getState(
                getForces=True, getEnergy=True
            )
            pre_energy = (
                pre_state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
                / GAS_CONSTANT
                / self._temperature
            )
            pre_forces = pre_state.getForces().value_in_unit(
                u.kilojoule_per_mole / u.nanometer
            )
            pre_norm = np.max(np.linalg.norm(pre_forces, axis=1))
            logger.info(f"Starting energy {pre_energy:.3f}.")
            logger.info(f"Starting maximum force norm {pre_norm:.3f}.")

            self._simulation.minimizeEnergy(maxIterations=self._options.minimize_steps)

            post_state = self._simulation.context.getState(
                getForces=True, getEnergy=True
            )
            post_energy = (
                post_state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
                / GAS_CONSTANT
                / self._temperature
            )
            post_forces = post_state.getForces().value_in_unit(
                u.kilojoule_per_mole / u.nanometer
            )
            post_norm = np.linalg.norm(post_forces, axis=1)
            post_max = np.max(post_norm)
            post_index = np.argmax(post_norm)
            logger.info(f"Ending energy {post_energy:.3f}.")
            logger.info(
                f"Ending maximum force norm {post_max:.3f} on particle {post_index}."
            )

        # set the velocities
        # check to see if velocities initialized to zero
        if np.all(velocities._value == 0.0):
            logger.info(
                "All velocities are zero, this is likely because input files do not contain velocity info. Generating velocities from Maxwell-Boltzmann distribution"
            )
            self._simulation.context.setVelocitiesToTemperature(self._temperature)
        else:
            self._simulation.context.setVelocities(velocities)

        # run timesteps
        self._simulation.step(self._options.timesteps)

        # extract coords, vels, energy and strip units
        if self._solvation == "implicit":
            snapshot = self._simulation.context.getState(
                getPositions=True, getVelocities=True, getEnergy=True
            )
        elif self._solvation == "explicit":
            snapshot = self._simulation.context.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=True,
            )
        coordinates = snapshot.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        velocities = snapshot.getVelocities(asNumpy=True).value_in_unit(
            u.nanometer / u.picosecond
        )
        _check_for_nan(coordinates, velocities, self._rank)

        # if explicit solvent, the recover the box vectors
        if self._solvation == "explicit":
            box_vector = snapshot.getPeriodicBoxVectors().value_in_unit(u.nanometer)
            box_vector = np.array(
                (box_vector[0][0], box_vector[1][1], box_vector[2][2])
            )
        # just store zeros for implicit solvent
        else:
            box_vector = np.zeros(3)
        # get the energy
        e_potential = (
            snapshot.getPotentialEnergy().value_in_unit(u.kilojoule / u.mole)
            / GAS_CONSTANT
            / self._temperature
        )

        if self._options.enable_gamd == True:
            gamd_logger = self.register_gamd_logger(self._integrator, self._simulation)
            gamd_logger.mark_energies()
            gamd_logger.write_to_gamd_log(self._timestep)
            gamd_logger.close()

        # store in state
        state.positions = coordinates
        state.velocities = velocities
        state.energy = e_potential
        state.box_vector = box_vector
        state.rdc_alignments = self._gather_alignments()

        return state

    def _gather_alignments(self):
        values = []
        if self.builder_info.get("has_alignments", False):
            for i in range(self.builder_info["num_alignments"]):
                for j in range(5):
                    a = self._simulation.context.getParameter(f"rdc_{i}_s{j + 1}")
                    values.append(a)
        values = np.array(values, dtype=np.float64)
        return values

    def _set_alignments(self, state):
        if self.builder_info.get("has_alignments", False):
            alignments = state.rdc_alignments.reshape(-1, 5)
            for i in range(alignments.shape[0]):
                for j in range(5):
                    self._simulation.context.setParameter(
                        f"rdc_{i}_s{j + 1}", alignments[i, j]
                    )

    def _run_param_mc(self, state):
        if not self._parameter_manager.has_parameters():
            return state

        if self._options.param_mcmc_steps is None:
            raise RuntimeError(
                "There are sampled parameters, but param_mcmc_steps is not set."
            )

        energy = self.get_energy(state)

        for _ in range(self._options.param_mcmc_steps):
            trial_params = self._parameter_manager.sample(state.parameters)
            if not self._parameter_manager.is_valid(trial_params):
                accept = False
            else:
                trial_state = SystemState(
                    state.positions,
                    state.velocities,
                    state.alpha,
                    state.energy,
                    state.group_energies,
                    state.box_vector,
                    trial_params,
                    state.mappings,
                )
                trial_energy = self.get_energy(trial_state)

                delta = trial_energy - energy

                if delta < 0:
                    accept = True
                else:
                    if random.random() < math.exp(-delta):
                        accept = True
                    else:
                        accept = False

            if accept:
                state = trial_state
                energy = trial_energy

        # Update transfomers in case we rejected the
        # last MCMC move
        if not accept:
            self._transformers_update(state)

        return state

    def _run_mapper_mc(self, state):
        if not self._mapper.has_mappers():
            return state

        if self._options.mapper_mcmc_steps is None:
            raise RuntimeError(
                "There are mapped atom groups, but mapper_mcmc_steps is not set."
            )

        energy = self.get_energy(state)

        accept = False

        for _ in range(self._options.mapper_mcmc_steps):
            trial_mappings = self._mapper.sample(state.mappings)

            trial_state = SystemState(
                state.positions,
                state.velocities,
                state.alpha,
                state.energy,
                state.group_energies,
                state.box_vector,
                state.parameters,
                trial_mappings,
            )
            trial_energy = self.get_energy(trial_state)

            delta = trial_energy - energy

            if delta < 0:
                accept = True
            else:
                if random.random() < math.exp(-delta):
                    accept = True
                else:
                    accept = False

            if accept:
                state = trial_state
                energy = trial_energy

        # Update transfomers in case we rejected the
        # last MCMC move
        if not accept:
            self._transformers_update(state)
        return state

    def register_gamd_logger(self, integrator, simulation):
        gamd_log_filename = os.path.join("Logs", f"gamd_{self._rank:03d}.log")
        if exists(gamd_log_filename):
            write_mode = "a"
        else:
            write_mode = "w"
        gamd_logger = GamdLogger(
            gamd_log_filename,
            write_mode,
            integrator,
            simulation,
            integrator.first_boost_type,
            integrator.first_boost_group,
            integrator.second_boost_type,
            integrator.second_boost_group,
        )
        if write_mode == "w":
            gamd_logger.write_header()
        return gamd_logger


def _check_for_nan(
    coordinates: np.ndarray, velocities: np.ndarray, rank: Optional[int]
) -> None:
    output_rank = 0 if rank is None else rank
    if np.isnan(coordinates).any():
        raise RuntimeError("Coordinates for rank {} contain NaN", output_rank)
    if np.isnan(velocities).any():
        raise RuntimeError("Velocities for rank {} contain NaN", output_rank)


def _create_openmm_simulation(topology, system, integrator, platform, properties):
    return app.Simulation(topology, system, integrator, platform, properties)


def _add_extras(system, bonds, restricted_angles, torsions):
    # add the extra bonds
    if bonds:
        f = [f for f in system.getForces() if isinstance(f, mm.HarmonicBondForce)][0]
        for bond in bonds:
            f.addBond(bond.i, bond.j, bond.length, bond.force_constant)

    # add the extra restricted_angles
    if restricted_angles:
        # create the new force for restricted angles
        f = mm.CustomAngleForce(
            "0.5 * k_ra * (theta - theta0_ra)^2 / sin(theta * 3.1459 / 180)"
        )
        f.addPerAngleParameter("k_ra")
        f.addPerAngleParameter("theta0_ra")
        for angle in restricted_angles:
            f.addAngle(
                angle.i,
                angle.j,
                angle.k,
                (angle.force_constant, angle.angle),
            )
        system.addForce(f)

    # add the extra torsions
    if torsions:
        f = [f for f in system.getForces() if isinstance(f, mm.PeriodicTorsionForce)][0]
        for tors in torsions:
            f.addTorsion(
                tors.i,
                tors.j,
                tors.k,
                tors.l,
                tors.multiplicity,
                tors.phase,
                tors.energy,
            )
