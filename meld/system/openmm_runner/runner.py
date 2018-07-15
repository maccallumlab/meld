#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from simtk.openmm.app import AmberPrmtopFile, OBC2, GBn, GBn2, Simulation  #type: ignore
from simtk.openmm.app import forcefield as ff  #type: ignore
from simtk.openmm import (  #type: ignore
    LangevinIntegrator,
    Platform,
    CustomExternalForce,
    CustomCentroidBondForce,
    MonteCarloBarostat,
    HarmonicBondForce,
    HarmonicAngleForce,
    PeriodicTorsionForce,
    CustomAngleForce,
)
from simtk.unit import (  #type: ignore
    Quantity,
    kelvin,
    picosecond,
    femtosecond,
    angstrom,
    kilojoule,
    mole,
    gram,
    nanometer,
    atmosphere,
)
from meld.system.restraints import (
    SelectableRestraint,
    NonSelectableRestraint,
    DistanceRestraint,
    TorsionRestraint,
    ConfinementRestraint,
    DistProfileRestraint,
    TorsProfileRestraint,
    CartesianRestraint,
    YZCartesianRestraint,
    COMRestraint,
    AbsoluteCOMRestraint,
    RdcRestraint,
    HyperbolicDistanceRestraint,
)
from meld.system.openmm_runner import cmap
from meld.system.openmm_runner import transform
import logging
from meld.util import log_timing
import numpy as np  #type: ignore
import tempfile
from collections import Callable, namedtuple

logger = logging.getLogger(__name__)

try:
    from meldplugin import MeldForce, RdcForce  #type: ignore
except ImportError:
    logger.warning(
        "Could not import meldplugin. "
        "Are you sure it is installed correctly?\n"
        "Attempts to use meld restraints will fail."
    )


GAS_CONSTANT = 8.314e-3


# namedtuples to store simulation information
PressureCouplingParams = namedtuple(
    "PressureCouplingParams", ["enable", "pressure", "temperature", "steps"]
)
PMEParams = namedtuple("PMEParams", ["enable", "tolerance"])


class OpenMMRunner:
    def __init__(self, system, options, communicator=None, test=False):
        if communicator:
            self._device_id = communicator.negotiate_device_id()
            self._rank = communicator.rank
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
        self._test = test
        self._simulation = None
        self._integrator = None
        self._barostat = None
        self._timestep = None
        self._initialized = False
        self._alpha = 0.
        self._temperature = None
        self._force_dict = {}
        self._transformers = []
        self._extra_bonds = system.extra_bonds
        self._extra_restricted_angles = system.extra_restricted_angles
        self._extra_torsions = system.extra_torsions

    def prepare_for_timestep(self, alpha, timestep):
        self._alpha = alpha
        self._timestep = timestep
        self._temperature = self.temperature_scaler(alpha)
        self._initialize_simulation()

    @log_timing(logger)
    def minimize_then_run(self, state):
        return self._run(state, minimize=True)

    @log_timing(logger)
    def run(self, state):
        return self._run(state, minimize=False)

    def get_energy(self, state):
        # set the coordinates
        coordinates = Quantity(state.positions, angstrom)

        # set the box vectors
        self._simulation.context.setPositions(coordinates)
        if self._options.solvation == "explicit":
            box_vector = state.box_vector / 10.  # Angstrom to nm
            self._simulation.context.setPeriodicBoxVectors(
                [box_vector[0], 0., 0.],
                [0., box_vector[1], 0.],
                [0., 0., box_vector[2]],
            )

        # get the energy
        snapshot = self._simulation.context.getState(
            getPositions=True, getVelocities=True, getEnergy=True
        )
        snapshot = self._simulation.context.getState(getEnergy=True)
        e_potential = snapshot.getPotentialEnergy()
        e_potential = (
            e_potential.value_in_unit(kilojoule / mole)
            / GAS_CONSTANT
            / self._temperature
        )
        return e_potential

    def _initialize_simulation(self):
        if self._initialized:
            # update temperature and pressure
            self._integrator.setTemperature(self._temperature)
            if self._options.enable_pressure_coupling:
                self._simulation.context.setParameter(
                    self._barostat.Temperature(), self._temperature
                )

            # update all of the system transformers
            self._transformers_update()

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
            )

            self._barostat = barostat

            if self._options.use_amap:
                adder = cmap.CMAPAdder(
                    self._parm_string,
                    self._options.amap_alpha_bias,
                    self._options.amap_beta_bias,
                    self._options.ccap,
                    self._options.ncap,
                )
                adder.add_to_openmm(sys)

            # setup the transformers
            self._transformers_setup()
            if len(self._always_on_restraints) > 0:
                print("Not all always on restraints were handled.")
                for r in self._always_on_restraints:
                    print("\t", r)
                raise RuntimeError("Not all always on restraints were handled.")

            if len(self._selectable_collections) > 0:
                print("Not all selectable restraints were handled.")
                for r in self._selectable_collections:
                    print("\t", r)
                raise RuntimeError("Not all selectable restraints were handled.")

            sys = self._transformers_add_interactions(sys, prmtop.topology)
            self._transformers_finalize(sys, prmtop.topology)

            # create the integrator
            self._integrator = _create_integrator(
                self._temperature,
                self._options.use_big_timestep,
                self._options.use_bigger_timestep,
            )

            # setup the platform, CUDA by default and Reference for testing
            if self._test:
                platform = Platform.getPlatformByName("Reference")
                properties = {}
            else:
                platform = Platform.getPlatformByName("CUDA")
                properties = {
                    "CudaDeviceIndex": str(self._device_id),
                    "CudaPrecision": "mixed",
                }

            # create the simulation object
            self._simulation = _create_openmm_simulation(
                prmtop.topology, sys, self._integrator, platform, properties
            )

            self._transformers_update()

    def _transformers_setup(self):
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
                self._options, self._always_on_restraints, self._selectable_collections
            )
            self._transformers.append(trans)

    def _transformers_add_interactions(self, sys, topol):
        for t in self._transformers:
            sys = t.add_interactions(sys, topol)
        return sys

    def _transformers_finalize(self, sys, topol):
        for t in self._transformers:
            t.finalize(sys, topol)

    def _transformers_update(self):
        for t in self._transformers:
            t.update(self._simulation, self._alpha, self._timestep)

    def _run_min_mc(self, state):
        if self._options.min_mc is not None:
            logger.info("Running MCMC before minimization.")
            logger.info(f"Starting energy {self.get_energy(state):.3f}")
            state.energy = self.get_energy(state)
            state = self._options.min_mc.update(state, self)
            logger.info(f"Ending energy {self.get_energy(state):.3f}")
        return state

    def _run_mc(self, state):
        if self._options.run_mc is not None:
            logger.info("Running MCMC.")
            logger.debug(f"Starting energy {self.get_energy(state):.3f}")
            state.energy = self.get_energy(state)
            state = self._options.run_mc.update(state, self)
            logger.debug(f"Ending energy {self.get_energy(state):.3f}")
        return state

    def _run(self, state, minimize):
        assert abs(state.alpha - self._alpha) < 1e-6  # run Monte Carlo
        if minimize:
            state = self._run_min_mc(state)
        else:
            state = self._run_mc(state)

        # add units to coordinates and velocities (we store in Angstrom, openmm
        # uses nm
        coordinates = Quantity(state.positions, angstrom)
        velocities = Quantity(state.velocities, angstrom / picosecond)
        box_vectors = Quantity(state.box_vector, angstrom)

        # set the positions
        self._simulation.context.setPositions(coordinates)

        # if explicit solvent, then set the box vectors
        if self._options.solvation == "explicit":
            self._simulation.context.setPeriodicBoxVectors(
                [box_vectors[0].value_in_unit(nanometer), 0., 0.],
                [0., box_vectors[1].value_in_unit(nanometer), 0.],
                [0., 0., box_vectors[2].value_in_unit(nanometer)],
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
        coordinates = snapshot.getPositions(asNumpy=True).value_in_unit(angstrom)
        velocities = snapshot.getVelocities(asNumpy=True).value_in_unit(
            angstrom / picosecond
        )
        _check_for_nan(coordinates, velocities, self._rank)

        # if explicit solvent, the recover the box vectors
        if self._options.solvation == "explicit":
            box_vector = snapshot.getPeriodicBoxVectors().value_in_unit(angstrom)
            box_vector = np.array(
                (box_vector[0][0], box_vector[1][1], box_vector[2][2])
            )
        # just store zeros for implicit solvent
        else:
            box_vector = np.zeros(3)

        # get the energy
        e_potential = (
            snapshot.getPotentialEnergy().value_in_unit(kilojoule / mole)
            / GAS_CONSTANT
            / self._temperature
        )

        # store in state
        state.positions = coordinates
        state.velocities = velocities
        state.energy = e_potential
        state.box_vector = box_vector

        return state


def _check_for_nan(coordinates, velocities, rank):
    if np.isnan(coordinates).any():
        raise RuntimeError("Coordinates for rank {} contain NaN", rank)
    if np.isnan(velocities).any():
        raise RuntimeError("Velocities for rank {} contain NaN", rank)


def _create_openmm_simulation(topology, system, integrator, platform, properties):
    return Simulation(topology, system, integrator, platform, properties)


def _parm_top_from_string(parm_string):
    with tempfile.NamedTemporaryFile(mode="w") as parm_file:
        parm_file.write(parm_string)
        parm_file.flush()
        prm_top = AmberPrmtopFile(parm_file.name)
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
        f = [f for f in system.getForces() if isinstance(f, HarmonicBondForce)][0]
        for bond in bonds:
            f.addBond(bond.i - 1, bond.j - 1, bond.length, bond.force_constant)

    # add the extra restricted_angles
    if restricted_angles:
        # create the new force for restricted angles
        f = CustomAngleForce(
            "0.5 * k_ra * (theta - theta0_ra)^2 / sin(theta * 3.1459 / 180)"
        )
        f.addPerAngleParameter("k_ra")
        f.addPerAngleParameter("theta0_ra")
        for angle in restricted_angles:
            f.addAngle(
                angle.i - 1,
                angle.j - 1,
                angle.k - 1,
                (angle.force_constant, angle.angle),
            )
        system.addForce(f)

    # add the extra torsions
    if torsions:
        f = [f for f in system.getForces() if isinstance(f, PeriodicTorsionForce)][0]
        for tors in torsions:
            f.addTorsion(
                tors.i - 1,
                tors.j - 1,
                tors.k - 1,
                tors.l - 1,
                tors.multiplicity,
                tors.phase,
                tors.energy,
            )


def _get_hydrogen_mass_and_constraints(use_big_timestep, use_bigger_timestep):
    if use_big_timestep:
        logger.info("Enabling hydrogen mass=3, constraining all bonds")
        constraint_type = ff.AllBonds
        hydrogen_mass = 3.0 * gram / mole
    elif use_bigger_timestep:
        logger.info("Enabling hydrogen mass=4, constraining all bonds")
        constraint_type = ff.AllBonds
        hydrogen_mass = 4.0 * gram / mole
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
):
    if cutoff is None:
        logger.info("Using no cutoff")
        cutoff_type = ff.NoCutoff
        cutoff_dist = 999.
    else:
        logger.info(f"Using a cutoff of {cutoff}")
        cutoff_type = ff.CutoffNonPeriodic
        cutoff_dist = cutoff

    hydrogen_mass, constraint_type = _get_hydrogen_mass_and_constraints(
        use_big_timestep, use_bigger_timestep
    )

    if implicit_solvent == "obc":
        logger.info('Using "OBC" implicit solvent')
        implicit_type = OBC2
    elif implicit_solvent == "gbNeck":
        logger.info('Using "gbNeck" implicit solvent')
        implicit_type = GBn
    elif implicit_solvent == "gbNeck2":
        logger.info('Using "gbNeck2" implicit solvent')
        implicit_type = GBn2
    elif implicit_solvent == "vacuum" or implicit_solvent is None:
        logger.info("Using vacuum instead of implicit solvent")
        implicit_type = None
    else:
        RuntimeError("Should never get here")
    return parm_object.createSystem(
        nonbondedMethod=cutoff_type,
        nonbondedCutoff=cutoff_dist,
        constraints=constraint_type,
        implicitSolvent=implicit_type,
        removeCMMotion=remove_com,
        hydrogenMass=hydrogen_mass,
    )


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
        baro = MonteCarloBarostat(
            pcouple_params.pressure, pcouple_params.temperature, pcouple_params.steps
        )
        s.addForce(baro)

    return s, baro


def _create_integrator(temperature, use_big_timestep, use_bigger_timestep):
    if use_big_timestep:
        logger.info("Creating integrator with 3.5 fs timestep")
        timestep = 3.5 * femtosecond
    elif use_bigger_timestep:
        logger.info("Creating integrator with 4.5 fs timestep")
        timestep = 4.5 * femtosecond
    else:
        logger.info("Creating integrator with 2.0 fs timestep")
        timestep = 2.0 * femtosecond
    return LangevinIntegrator(temperature * kelvin, 1.0 / picosecond, timestep)
