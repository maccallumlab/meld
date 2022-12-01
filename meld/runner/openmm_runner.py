#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import interfaces
from meld.system import options
from meld.system import restraints
from meld.system.state import SystemState
from meld.runner import transform
from meld.util import log_timing
from meldplugin import MeldForce  # type: ignore

from openmm import app  # type: ignore
from openmm.app import forcefield as ff  # type: ignore
from simtk import openmm as mm  # type: ignore
from simtk import unit as u  # type: ignore
import parmed
import logging
import numpy as np  # type: ignore
import tempfile
from collections import namedtuple
from typing import Optional, List, Dict, NamedTuple
import random
import math

logger = logging.getLogger(__name__)


GAS_CONSTANT = 8.314e-3


# namedtuples to store simulation information
class PressureCouplingParams(NamedTuple):
    enable: bool
    pressure: float
    temperature: float
    steps: int


class PMEParams(NamedTuple):
    enable: bool
    tolerance: float


class OpenMMRunner(interfaces.IRunner):
    _parm_string: str
    _always_on_restraints: List[restraints.Restraint]
    _selectable_collections: List[restraints.SelectivelyActiveCollection]
    _options: options.RunOptions
    _simulation: app.Simulation
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
        system: interfaces.ISystem,
        options: options.RunOptions,
        communicator: Optional[interfaces.ICommunicator] = None,
        platform: str = None,
    ):
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

        if system.temperature_scaler is None:
            raise RuntimeError("system does not have temparture_scaler set")
        else:
            self.temperature_scaler = system.temperature_scaler
        self._parm_string = system.top_string
        self._always_on_restraints = system.restraints.always_active
        self._selectable_collections = system.restraints.selectively_active_collections
        self._options = options
        self._timestep = 0
        self._initialized = False
        self._alpha = 0.0
        self._transformers: List[transform.TransformerBase] = []
        self._extra_bonds = system.extra_bonds
        self._extra_restricted_angles = system.extra_restricted_angles
        self._extra_torsions = system.extra_torsions
        self._parameter_manager = system.param_sampler
        self._mapper = system.mapper
        self._density = system.density

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
        if self._options.solvation == "explicit":
            box_vector = state.box_vector
            self._simulation.context.setPeriodicBoxVectors(
                [box_vector[0], 0.0, 0.0],
                [0.0, box_vector[1], 0.0],
                [0.0, 0.0, box_vector[2]],
            )

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

    def _initialize_simulation(self, state: interfaces.IState) -> None:
        if self._initialized:
            # update temperature and pressure
            self._integrator.setTemperature(self._temperature)
            if self._options.enable_pressure_coupling:
                self._simulation.context.setParameter(
                    self._barostat.Temperature(), self._temperature
                )

            # update all of the system transformers
            self._transformers_update(state)

        else:
            # we need to set the whole thing from scratch
            self._initialized = True

            prmtop = _parm_top_from_string(self._parm_string)

            # create parameter objects
            pme_params = PMEParams(
                enable=self._options.enable_pme, tolerance=self._options.pme_tolerance
            )
            pcouple_params = PressureCouplingParams(
                enable=self._options.enable_pressure_coupling,
                temperature=self._temperature,
                pressure=self._options.pressure,
                steps=self._options.pressure_coupling_update_steps,
            )

            # build the system
            sys, barostat = _create_openmm_system(
                prmtop,
                self._options.solvation,
                self._options.cutoff,
                self._options.use_big_timestep,
                self._options.use_bigger_timestep,
                self._options.implicit_solvent_model,
                pme_params,
                pcouple_params,
                self._options.remove_com,
                self._temperature,
                self._extra_bonds,
                self._extra_restricted_angles,
                self._extra_torsions,
                self._options.implicitSolventSaltConc,
                self._options.soluteDielectric,
                self._options.solventDielectric,
            )
            self._barostat = barostat

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

            sys = self._transformers_add_interactions(state, sys, prmtop.topology)
            self._transformers_finalize(state, sys, prmtop.topology)

            # create the integrator
            self._integrator = _create_integrator(
                self._temperature,
                self._options.use_big_timestep,
                self._options.use_bigger_timestep,
            )

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
                prmtop.topology, sys, self._integrator, platform, properties
            )
            # forcegroups=self._forcegroupify(sys)
            # self._simulation = _create_openmm_simulation(
            #     prmtop.topology, sys, self._integrator, platform, properties
            # )
            self._transformers_update(state)
    def _forcegroupify(self,system):
        forcegroups = {}
        for i in range(system.getNumForces()):
            # logger.info(f"{i}th force \n")
            force = system.getForce(i)
            force.setForceGroup(i)
            forcegroups[force] = i
        return forcegroups

    def _transformers_setup(self) -> None:
        # CMAPTransformer is different, so we add it separately
        self._transformers.append(
            transform.CMAPTransformer(self._options, self._parm_string)
        )

        trans_types = [
            transform.ConfinementRestraintTransformer,
            transform.RDCRestraintTransformer,
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
            state.energy = self.get_energy(state)
            state = self._options.min_mc.update(state, self)
            logger.info(f"Ending energy {self.get_energy(state):.3f}")
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
        if self._options.solvation == "explicit":
            self._simulation.context.setPeriodicBoxVectors(
                [box_vectors[0].value_in_unit(u.nanometer), 0.0, 0.0],
                [0.0, box_vectors[1].value_in_unit(u.nanometer), 0.0],
                [0.0, 0.0, box_vectors[2].value_in_unit(u.nanometer)],
            )

        # run energy minimization
        if minimize:
            self._simulation.minimizeEnergy(maxIterations=self._options.minimize_steps)

        # set the velocities
        self._simulation.context.setVelocities(velocities)
    
        # run timesteps
        self._simulation.step(self._options.timesteps)

        # extract coords, vels, energy and strip units
        if self._options.solvation == "implicit":
            snapshot = self._simulation.context.getState(
                getPositions=True, getVelocities=True, getEnergy=True
            )
        elif self._options.solvation == "explicit":
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
        if self._options.solvation == "explicit":
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

        # store in state
        state.positions = coordinates
        state.velocities = velocities
        state.energy = e_potential
        state.box_vector = box_vector

        return state

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

        for _ in range(self._options.mapper_mcmc_steps):
            trial_mappings = self._mapper.sample(state.mappings)

            trial_state = SystemState(
                state.positions,
                state.velocities,
                state.alpha,
                state.energy,
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


def _parm_top_from_string(parm_string):
    with tempfile.NamedTemporaryFile(mode="w") as parm_file:
        parm_file.write(parm_string)
        parm_file.flush()
        prm_top = app.AmberPrmtopFile(parm_file.name)
        return prm_top


def _create_openmm_system(
    parm_object,
    solvation_type,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
    implicit_solvent,
    pme_params,
    pcouple_params,
    remove_com,
    temperature,
    extra_bonds,
    extra_restricted_angles,
    extra_torsions,
    implicitSolventSaltConc,
    soluteDielectric,
    solventDielectric,
):
    if solvation_type == "implicit":
        logger.info("Creating implicit solvent system")
        system, baro = (
            _create_openmm_system_implicit(
                parm_object,
                cutoff,
                use_big_timestep,
                use_bigger_timestep,
                implicit_solvent,
                remove_com,
                implicitSolventSaltConc,
                soluteDielectric,
                solventDielectric,
            ),
            None,
        )
    elif solvation_type == "explicit":
        logger.info("Creating explicit solvent system")
        system, baro = _create_openmm_system_explicit(
            parm_object,
            cutoff,
            use_big_timestep,
            use_bigger_timestep,
            pme_params,
            pcouple_params,
            remove_com,
            temperature,
        )
    else:
        raise ValueError(f"unknown value for solvation_type: {solvation_type}")

    _add_extras(system, extra_bonds, extra_restricted_angles, extra_torsions)

    return system, baro


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


def _get_hydrogen_mass_and_constraints(use_big_timestep, use_bigger_timestep):
    if use_big_timestep:
        logger.info("Enabling hydrogen mass=3, constraining all bonds")
        constraint_type = ff.AllBonds
        hydrogen_mass = 3.0 * u.gram / u.mole
    elif use_bigger_timestep:
        logger.info("Enabling hydrogen mass=4, constraining all bonds")
        constraint_type = ff.AllBonds
        hydrogen_mass = 4.0 * u.gram / u.mole
    else:
        logger.info("Enabling hydrogen mass=1, constraining bonds with hydrogen")
        constraint_type = ff.HBonds
        hydrogen_mass = None
    return hydrogen_mass, constraint_type


def _create_openmm_system_implicit(
    parm_object,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
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

    hydrogen_mass, constraint_type = _get_hydrogen_mass_and_constraints(
        use_big_timestep, use_bigger_timestep
    )

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
        constraints=constraint_type,
        implicitSolvent=implicit_type,
        removeCMMotion=remove_com,
        hydrogenMass=hydrogen_mass,
        implicitSolventSaltConc=implicitSolventSaltConc,
        soluteDielectric=soluteDielectric,
        solventDielectric=solventDielectric,
    )
    return sys


def _create_openmm_system_explicit(
    parm_object,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
    pme_params,
    pcouple_params,
    remove_com,
    temperature,
):
    if cutoff is None:
        raise ValueError("cutoff must be set for explicit solvent, but got None")
    else:
        if pme_params.enable:
            logger.info(f"Using PME with tolerance {pme_params.tolerance}")
            cutoff_type = ff.PME
        else:
            logger.info("Using reaction field")
            cutoff_type = ff.CutoffPeriodic

        logger.info(f"Using a cutoff of {cutoff}")
        cutoff_dist = cutoff

    hydrogen_mass, constraint_type = _get_hydrogen_mass_and_constraints(
        use_big_timestep, use_bigger_timestep
    )

    s = parm_object.createSystem(
        nonbondedMethod=cutoff_type,
        nonbondedCutoff=cutoff_dist,
        constraints=constraint_type,
        implicitSolvent=None,
        removeCMMotion=remove_com,
        hydrogenMass=hydrogen_mass,
        rigidWater=True,
        ewaldErrorTolerance=pme_params.tolerance,
    )

    baro = None
    if pcouple_params.enable:
        logger.info("Enabling pressure coupling")
        logger.info(f"Pressure is {pcouple_params.pressure}")
        logger.info(f"Volume moves attempted every {pcouple_params.steps} steps")
        baro = mm.MonteCarloBarostat(
            pcouple_params.pressure, pcouple_params.temperature, pcouple_params.steps
        )
        s.addForce(baro)

    return s, baro


def _create_integrator(temperature, use_big_timestep, use_bigger_timestep):
    if use_big_timestep:
        logger.info("Creating integrator with 3.5 fs timestep")
        timestep = 3.5 * u.femtosecond
    elif use_bigger_timestep:
        logger.info("Creating integrator with 4.5 fs timestep")
        timestep = 4.5 * u.femtosecond
    else:
        logger.info("Creating integrator with 2.0 fs timestep")
        timestep = 2.0 * u.femtosecond
    return mm.LangevinIntegrator(temperature * u.kelvin, 1.0 / u.picosecond, timestep)
