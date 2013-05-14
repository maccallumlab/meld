from traits.api import HasTraits, Trait, BaseInt, BaseFloat, Enum, Bool
from simtk.openmm.app import AmberPrmtopFile, OBC2, GBn, GBn2, Simulation
from simtk.openmm.app import forcefield as ff
from simtk.openmm import LangevinIntegrator
from simtk.unit import kelvin, picosecond, femtosecond, angstrom
from simtk.unit import Quantity, kilojoule, mole


gas_constant = 8.314e-3


class ReplicaRunner(object):
    def initialize(self):
        pass

    def minimize_then_run(self, state):
        pass

    def run(self, state):
        pass

    def get_energy(self, state):
        pass

    def set_alpha(self, state):
        pass


class FakeSystemRunner(object):
    '''
    Fake runner for test purposes.
    '''
    def set_alpha(self, alpha):
        pass

    def minimize_then_run(self, state):
        return state

    def run(self, state):
        return state

    def get_energy(self, state):
        return 0.


class OpenMMRunner(object):
    def __init__(self, system, options):
        if system.temperature_scaler is None:
            raise RuntimeError('system does not have temparture_scaler set')
        else:
            self.temperature_scaler = system.temperature_scaler
        self._parm_string = system.top_string

        self._options = options
        self._simulation = None
        self._alpha = 0.
        self._temperature = None

    def set_alpha(self, alpha):
        self._alpha = alpha
        self._temperature = self.temperature_scaler(alpha)
        self._initialize_simulation()

    def minimize_then_run(self, state):
        return self._run(state, minimize=True)

    def run(self, state):
        return self._run(state, minimize=False)

    def get_energy(self, state):
        # set the coordinates
        coordinates = Quantity(state.positions, angstrom)
        self._simulation.context.setPositions(coordinates)

        # get the energy
        snapshot = self._simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        e_potential = snapshot.getPotentialEnergy()
        e_potential = e_potential.value_in_unit(kilojoule / mole) / gas_constant / self._temperature

        return e_potential

    def _initialize_simulation(self):
        prmtop = _parm_top_from_string(self._parm_string)
        sys = _create_openmm_system(prmtop, self._options.cutoff, self._options.use_big_timestep,
                                    self._options.implicit_solvent_model)
        integrator = _create_integrator(self._temperature, self._options.use_big_timestep)
        self._simulation = _create_openmm_simulation(prmtop.topology, sys, integrator)

    def _run(self, state, minimize):
        assert state.alpha == self._alpha

        # add units to coordinates and velocities (we store in Angstrom, openmm
        # uses nm
        coordinates = Quantity(state.positions, angstrom)
        velocities = Quantity(state.velocities, angstrom / picosecond)

        # set the positions
        self._simulation.context.setPositions(coordinates)

        # run energy minimization
        if minimize:
            self._simulation.minimizeEnergy(self._options.minimize_steps)

        # set the velocities
        self._simulation.context.setVelocities(velocities)

        # run timesteps

        self._simulation.step(self._options.timesteps)

        # extract coords, vels, energy and strip units
        snapshot = self._simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        coordinates = snapshot.getPositions(asNumpy=True).value_in_unit(angstrom)
        velocities = snapshot.getVelocities(asNumpy=True).value_in_unit(angstrom / picosecond)
        e_potential = snapshot.getPotentialEnergy().value_in_unit(kilojoule / mole) / gas_constant / self._temperature

        # store in state
        state.positions = coordinates
        state.velocities = velocities
        state.energy = e_potential

        return state


def _create_openmm_simulation(topology, system, integrator):
    return Simulation(topology, system, integrator)


def _parm_top_from_string(parm_string):
    return AmberPrmtopFile(parm_string=parm_string)


def _create_openmm_system(parm_object, cutoff, use_big_timestep, implicit_solvent):
    if cutoff is None:
        cutoff_type = ff.NoCutoff
        cutoff_dist = 999.
    else:
        cutoff_type = ff.CutoffNonPeriodic
        cutoff_dist = cutoff

    if use_big_timestep:
        constraint_type = ff.HAngles
    else:
        constraint_type = ff.HBonds

    if implicit_solvent == 'obc':
        implicit_type = OBC2
    elif implicit_solvent == 'gbNeck':
        implicit_type = GBn
    elif implicit_solvent == 'gbNeck2':
        implicit_type = GBn2
    return parm_object.createSystem(nonbondedMethod=cutoff_type, nonbondedCutoff=cutoff_dist,
                                    constraints=constraint_type, implicitSolvent=implicit_type)


def _create_integrator(temperature, use_big_timestep):
    if use_big_timestep:
        timestep = 3.5 * femtosecond
    else:
        timestep = 2.0 * femtosecond
    return LangevinIntegrator(temperature * kelvin, 1.0 / picosecond, timestep)


class PositiveInt(BaseInt):
    default_value = 1

    # Describe the trait type
    info_text = 'an integer > 0'

    def validate(self, object, name, value):
        value = super(PositiveInt, self).validate(object, name, value)
        if value > 0:
            return value
        self.error(object, name, value)


class PositiveFloat(BaseFloat):
    default_value = 1

    # Describe the trait type
    info_text = 'a float > 0'

    def validate(self, object, name, value):
        value = super(PositiveFloat, self).validate(object, name, value)
        if value > 0:
            return value
        self.error(object, name, value)


class RunOptions(HasTraits):
    runner = Enum('openmm', 'fake_runner')
    timesteps = PositiveInt(5000)
    minimize_steps = PositiveInt(1000)
    implicit_solvent_model = Enum('gbNeck2', 'gbNeck', 'obc')
    cutoff = Trait(None, None, PositiveFloat)  # allow None or PositiveFloat; default to None
    use_big_timestep = Bool(False)
    use_amap = Bool(False)


def get_runner(system, options):
    if options.runner == 'openmm':
        return OpenMMRunner(system, options)
    elif options.runner == 'fake_runner':
        return FakeSystemRunner()
    else:
        raise RuntimeError('Unknown type of runner: {}'.format(options.runner))
