from simtk.openmm.app import AmberPrmtopFile, OBC2, GBn, GBn2, Simulation
from simtk.openmm.app import forcefield as ff
from simtk.openmm import LangevinIntegrator, Platform, CustomExternalForce
from simtk.unit import kelvin, picosecond, femtosecond, angstrom
from simtk.unit import Quantity, kilojoule, mole, gram
from meld.system.restraints import SelectableRestraint, NonSelectableRestraint, DistanceRestraint, TorsionRestraint
from meld.system.restraints import ConfinementRestraint, DistProfileRestraint, TorsProfileRestraint
from meld.system.restraints import CartesianRestraint, YZCartesianRestraint, XAxisCOMRestraint
from meld.system.restraints import RdcRestraint, HyperbolicDistanceRestraint
from . import softcore
import cmap
import logging
from meld.util import log_timing
import numpy as np
import tempfile

logger = logging.getLogger(__name__)

try:
    from meldplugin import MeldForce, RdcForce
except ImportError as e:
    logger.warning('Could not import meldplugin. Are you sure it is installed correctly?\n'
                   'Attempts to use meld restraints will fail.')

try:
    from onedimcomplugin import OneDimComForce
except:
    logger.warning('Could not import onedimcomplugin. Are you sure it is installed correctly?\n'
                   'Attempts to use center of mass restraints will fail.')


GAS_CONSTANT = 8.314e-3


class OpenMMRunner(object):
    def __init__(self, system, options, communicator=None):
        if communicator:
            self._device_id = communicator.negotiate_device_id()
            self._rank = communicator.rank
        else:
            self._device_id = 0
            self._rank = None

        if system.temperature_scaler is None:
            raise RuntimeError('system does not have temparture_scaler set')
        else:
            self.temperature_scaler = system.temperature_scaler
        self._parm_string = system.top_string
        self._always_on_restraints = system.restraints.always_active
        self._selectable_collections = system.restraints.selectively_active_collections
        self._options = options
        self._simulation = None
        self._integrator = None
        self._initialized = False
        self._alpha = 0.
        self._temperature = None
        self._sc_lambda_coulomb = 1.0
        self._sc_lambda_lj = 1.0
        self._force_dict = {}

    def set_alpha_and_timestep(self, alpha, timestep):
        self._alpha = alpha
        self._timestep = timestep
        self._temperature = self.temperature_scaler(alpha)
        self._update_softcore()
        self._initialize_simulation()

    def _update_softcore(self):
        alpha = self._alpha
        a1 = self._options.sc_alpha_min
        a2 = self._options.sc_alpha_max_coulomb
        a3 = self._options.sc_alpha_max_lennard_jones

        if self._options.softcore:
            self._sc_lambda_coulomb = alpha
            self._sc_lambda_lj = alpha
            #logger.info('updating softcore')
            #if alpha <= a1:
            #    self._sc_lambda_coulomb = 1.0
            #    self._sc_lambda_lj = 1.0
            #elif alpha >= a3:
            #    self._sc_lambda_coulomb = 0.0
            #    self._sc_lambda_lj = 0.0
            #elif alpha < a2:
            #    self._sc_lambda_lj = 1.0
            #    self._sc_lambda_coulomb = 1.0 - (alpha - a1) / (a2 - a1)
            #else:
            #    self._sc_lambda_coulomb = 0.0
            #    self._sc_lambda_lj = 1.0 - (alpha - a2) / (a3 - a2)

    @log_timing(logger)
    def minimize_then_run(self, state):
        return self._run(state, minimize=True)

    @log_timing(logger)
    def run(self, state):
        return self._run(state, minimize=False)

    def get_energy(self, state):
        # set the coordinates
        coordinates = Quantity(state.positions, angstrom)
        self._simulation.context.setPositions(coordinates)

        # get the energy
        snapshot = self._simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        e_potential = snapshot.getPotentialEnergy()
        e_potential = e_potential.value_in_unit(kilojoule / mole) / GAS_CONSTANT / self._temperature
        return e_potential

    def _initialize_simulation(self):
        if self._initialized:
            self._integrator.setTemperature(self._temperature)
            if self._options.softcore:
                print self._sc_lambda_lj
                self._simulation.context.setParameter('qq_lambda', self._sc_lambda_coulomb)
                self._simulation.context.setParameter('lj_lambda', self._sc_lambda_lj)
                self._simulation.context.setParameter('sc_lambda', self._sc_lambda_lj)
                logger.debug('set sc %d %f %f %f', self._rank, self._sc_lambda_coulomb, self._sc_lambda_lj, self._sc_lambda_lj)

            meld_rests = _update_always_active_restraints(self._always_on_restraints, self._alpha,
                                                          self._timestep, self._force_dict)
            _update_selectively_active_restraints(self._selectable_collections, meld_rests, self._alpha,
                                                  self._timestep, self._force_dict)
            for force in self._force_dict.values():
                if force:
                    force.updateParametersInContext(self._simulation.context)

        else:
            self._initialized = True

            # we need to set the whole thing from scratch
            prmtop = _parm_top_from_string(self._parm_string)
            sys = _create_openmm_system(prmtop, self._options.cutoff, self._options.use_big_timestep,
                                        self._options.implicit_solvent_model, self._options.remove_com)

            if self._options.softcore:
                sys = softcore.add_soft_core(sys)

            if self._options.use_amap:
                adder = cmap.CMAPAdder(self._parm_string, self._options.amap_alpha_bias, self._options.amap_beta_bias, self._options.ccap, self._options.ncap)
                adder.add_to_openmm(sys)

            meld_rests = _add_always_active_restraints(sys, self._always_on_restraints, self._alpha,
                                                       self._timestep, self._force_dict)
            _add_selectively_active_restraints(sys, self._selectable_collections, meld_rests, self._alpha,
                                               self._timestep, self._force_dict)

            self._integrator = _create_integrator(self._temperature, self._options.use_big_timestep)

            platform = Platform.getPlatformByName('CUDA')
            properties = {'CudaDeviceIndex': str(self._device_id)}

            self._simulation = _create_openmm_simulation(prmtop.topology, sys, self._integrator,
                                                         platform, properties)

            if self._options.softcore:
                self._simulation.context.setParameter('qq_lambda', self._sc_lambda_coulomb)
                self._simulation.context.setParameter('lj_lambda', self._sc_lambda_lj)
                self._simulation.context.setParameter('sc_lambda', self._sc_lambda_lj)
                logger.debug('set sc %d %f %f %f', self._rank, self._sc_lambda_coulomb, self._sc_lambda_lj, self._sc_lambda_lj)

    def _run_min_mc(self, state):
        if self._options.min_mc is not None:
            logger.info('Running MCMC before minimization.')
            logger.info('Starting energy {:.3f}'.format(self.get_energy(state)))
            state.energy = self.get_energy(state)
            state = self._options.min_mc.update(state, self)
            logger.info('Ending energy {:.3f}'.format(self.get_energy(state)))
            if self._options.remove_com:
                state.positions = state.positions - np.mean(state.positions, axis=0)
        return state

    def _run_mc(self, state):
        if self._options.run_mc is not None:
            logger.info('Running MCMC.')
            logger.debug('Starting energy {:.3f}'.format(self.get_energy(state)))
            state.energy = self.get_energy(state)
            state = self._options.run_mc.update(state, self)
            logger.debug('Ending energy {:.3f}'.format(self.get_energy(state)))
            if self._options.remove_com:
                state.positions = state.positions - np.mean(state.positions, axis=0)
        return state

    def _run(self, state, minimize):
        assert abs(state.alpha - self._alpha) < 1e-6

        # run Monte Carlo
        if minimize:
            state = self._run_min_mc(state)
        else:
            state = self._run_mc(state)

        # add units to coordinates and velocities (we store in Angstrom, openmm
        # uses nm
        coordinates = Quantity(state.positions, angstrom)
        velocities = Quantity(state.velocities, angstrom / picosecond)

        # set the positions
        self._simulation.context.setPositions(coordinates)

        # run energy minimization
        if minimize:
            self._simulation.minimizeEnergy(maxIterations=self._options.minimize_steps)

        # set the velocities
        self._simulation.context.setVelocities(velocities)

        # run timesteps

        self._simulation.step(self._options.timesteps)

        # extract coords, vels, energy and strip units
        snapshot = self._simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
        coordinates = snapshot.getPositions(asNumpy=True).value_in_unit(angstrom)
        velocities = snapshot.getVelocities(asNumpy=True).value_in_unit(angstrom / picosecond)
        _check_for_nan(coordinates, velocities, self._rank)
        e_potential = snapshot.getPotentialEnergy().value_in_unit(kilojoule / mole) / GAS_CONSTANT / self._temperature

        # store in state
        state.positions = coordinates
        state.velocities = velocities
        state.energy = e_potential

        return state


def _check_for_nan(coordinates, velocities, rank):
    if np.isnan(coordinates).any():
        raise RuntimeError('Coordinates for rank {} contain NaN', rank)
    if np.isnan(velocities).any():
        raise RuntimeError('Velocities for rank {} contain NaN', rank)


def _create_openmm_simulation(topology, system, integrator, platform, properties):
    return Simulation(topology, system, integrator, platform, properties)


def _parm_top_from_string(parm_string):
    with tempfile.NamedTemporaryFile() as parm_file:
        parm_file.write(parm_string)
        parm_file.flush()
        prm_top = AmberPrmtopFile(parm_file.name)
        return prm_top


def _create_openmm_system(parm_object, cutoff, use_big_timestep, implicit_solvent, remove_com):
    if cutoff is None:
        cutoff_type = ff.NoCutoff
        cutoff_dist = 999.
    else:
        cutoff_type = ff.CutoffNonPeriodic
        cutoff_dist = cutoff

    if use_big_timestep:
        constraint_type = ff.AllBonds
        hydrogen_mass = 3.0 * gram / mole
    else:
        constraint_type = ff.HBonds
        hydrogen_mass = None

    if implicit_solvent == 'obc':
        implicit_type = OBC2
    elif implicit_solvent == 'gbNeck':
        implicit_type = GBn
    elif implicit_solvent == 'gbNeck2':
        implicit_type = GBn2
    elif implicit_solvent == 'vacuum':
        implicit_type = None
    elif implicit_solvent is None:
        implicit_type = None
    else:
        RuntimeError('Should never get here')
    return parm_object.createSystem(nonbondedMethod=cutoff_type, nonbondedCutoff=cutoff_dist,
                                    constraints=constraint_type, implicitSolvent=implicit_type,
                                    removeCMMotion=remove_com, hydrogenMass=hydrogen_mass)


def _create_integrator(temperature, use_big_timestep):
    if use_big_timestep:
        timestep = 3.5 * femtosecond
    else:
        timestep = 2.0 * femtosecond
    return LangevinIntegrator(temperature * kelvin, 1.0 / picosecond, timestep)


def _split_always_active_restraints(restraint_list):
    selectable_restraints = []
    nonselectable_restraints = []
    for rest in restraint_list:
        if isinstance(rest, SelectableRestraint):
            selectable_restraints.append(rest)
        elif isinstance(rest, NonSelectableRestraint):
            nonselectable_restraints.append(rest)
        else:
            raise RuntimeError('Unknown type of restraint {}'.format(rest))
    return selectable_restraints, nonselectable_restraints


def _add_always_active_restraints(system, restraint_list, alpha, timestep, force_dict):
    selectable_restraints, nonselectable_restraints = _split_always_active_restraints(restraint_list)
    nonselectable_restraints = _add_rdc_restraints(system, nonselectable_restraints, alpha, timestep, force_dict)
    nonselectable_restraints = _add_confinement_restraints(system, nonselectable_restraints, alpha, timestep, force_dict)
    nonselectable_restraints = _add_cartesian_restraints(system, nonselectable_restraints, alpha, timestep, force_dict)
    nonselectable_restraints = _add_yzcartesian_restraints(system, nonselectable_restraints, alpha, timestep, force_dict)
    nonselectable_restraints = _add_xcom_restraints(system, nonselectable_restraints, alpha, timestep, force_dict)
    if nonselectable_restraints:
        raise NotImplementedError('Non-meld restraints are not implemented yet')
    return selectable_restraints


def _update_always_active_restraints(restraint_list, alpha, timestep, force_dict):
    selectable_restraints, nonselectable_restraints = _split_always_active_restraints(restraint_list)
    nonselectable_restraints = _update_rdc_restraints(nonselectable_restraints, alpha, timestep, force_dict)
    nonselectable_restraints = _update_confinement_restraints(nonselectable_restraints, alpha, timestep, force_dict)
    nonselectable_restraints = _update_cartesian_restraints(nonselectable_restraints, alpha, timestep, force_dict)
    nonselectable_restraints = _update_yzcartesian_restraints(nonselectable_restraints, alpha, timestep, force_dict)
    nonselectable_restraints = _update_xcom_restraints(nonselectable_restraints, alpha, timestep, force_dict)
    if nonselectable_restraints:
        raise NotImplementedError('Non-meld restraints are not implemented yet')
    return selectable_restraints


def _add_selectively_active_restraints(system, collections, always_on, alpha, timestep, force_dict):
    if not (collections or always_on):
        # we don't need to do anything
        force_dict['meld'] = None
        return

    # otherwise we need a MeldForce
    meld_force = MeldForce()

    if always_on:
        group_list = []
        for rest in always_on:
            rest_index = _add_meld_restraint(rest, meld_force, alpha, timestep)
            group_index = meld_force.addGroup([rest_index], 1)
            group_list.append(group_index)
        meld_force.addCollection(group_list, len(group_list))
    for coll in collections:
        group_indices = []
        for group in coll.groups:
            restraint_indices = []
            for rest in group.restraints:
                rest_index = _add_meld_restraint(rest, meld_force, alpha, timestep)
                restraint_indices.append(rest_index)
            group_index = meld_force.addGroup(restraint_indices, group.num_active)
            group_indices.append(group_index)
        meld_force.addCollection(group_indices, coll.num_active)
    system.addForce(meld_force)
    force_dict['meld'] = meld_force


def _update_selectively_active_restraints(collections, always_on, alpha, timestep, force_dict):
    meld_force = force_dict['meld']
    dist_index = 0
    hyper_index = 0
    tors_index = 0
    dist_prof_index = 0
    tors_prof_index = 0
    if always_on:
        for rest in always_on:
            dist_index, hyper_index, tors_index, dist_prof_index, tors_prof_index = _update_meld_restraint(rest, meld_force, alpha, timestep,
                                                                            dist_index, hyper_index, tors_index, dist_prof_index, tors_prof_index)
    for coll in collections:
        for group in coll.groups:
            for rest in group.restraints:
                dist_index, hyper_index, tors_index, dist_prof_index, tors_prof_index = _update_meld_restraint(rest, meld_force, alpha, timestep,
                                                                                dist_index, hyper_index, tors_index, dist_prof_index, tors_prof_index)


def _add_confinement_restraints(system, restraint_list, alpha, timestep, force_dict):
    # split restraints into confinement and others
    confinement_restraints = [r for r in restraint_list if isinstance(r, ConfinementRestraint)]
    nonconfinement_restraints = [r for r in restraint_list if not isinstance(r, ConfinementRestraint)]

    if confinement_restraints:
        # create the confinement force
        confinement_force = CustomExternalForce(
            'step(r - radius) * force_const * (radius - r)^2; r=sqrt(x*x + y*y + z*z)')
        confinement_force.addPerParticleParameter('radius')
        confinement_force.addPerParticleParameter('force_const')

        # add the atoms
        for r in confinement_restraints:
            weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
            confinement_force.addParticle(r.atom_index - 1, [r.radius, weight])
        system.addForce(confinement_force)
        force_dict['confine'] = confinement_force
    else:
        force_dict['confine'] = None

    return nonconfinement_restraints


def _update_confinement_restraints(restraint_list, alpha, timestep, force_dict):
    # split restraints into confinement and others
    confinement_restraints = [r for r in restraint_list if isinstance(r, ConfinementRestraint)]
    other_restraints = [r for r in restraint_list if not isinstance(r, ConfinementRestraint)]

    if confinement_restraints:
        confinement_force = force_dict['confine']
        for index, r in enumerate(confinement_restraints):
            weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
            confinement_force.setParticleParameters(index, r.atom_index - 1, [r.radius, weight])
    return other_restraints


def _add_cartesian_restraints(system, restraint_list, alpha, timestep, force_dict):
    # split restraints into confinement and others
    cartesian_restraints = [r for r in restraint_list if isinstance(r, CartesianRestraint)]
    noncartesian_restraints = [r for r in restraint_list if not isinstance(r, CartesianRestraint)]

    if cartesian_restraints:
        # create the confinement force
        cartesian_force = CustomExternalForce(
            '0.5 * cart_force_const * r_eff^2; r_eff = max(0.0, r - cart_delta);'
            'r = sqrt(dx*dx + dy*dy + dz*dz);'
            'dx = x - cart_x;'
            'dy = y - cart_y;'
            'dz = z - cart_z;')
        cartesian_force.addPerParticleParameter('cart_x')
        cartesian_force.addPerParticleParameter('cart_y')
        cartesian_force.addPerParticleParameter('cart_z')
        cartesian_force.addPerParticleParameter('cart_delta')
        cartesian_force.addPerParticleParameter('cart_force_const')

        # add the atoms
        for r in cartesian_restraints:
            weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
            cartesian_force.addParticle(r.atom_index - 1, [r.x, r.y, r.z, r.delta, weight])
        system.addForce(cartesian_force)
        force_dict['cartesian'] = cartesian_force
    else:
        force_dict['cartesian'] = None

    return noncartesian_restraints


def _update_cartesian_restraints(restraint_list, alpha, timestep, force_dict):
    # split restraints into confinement and others
    cartesian_restraints = [r for r in restraint_list if isinstance(r, CartesianRestraint)]
    other_restraints = [r for r in restraint_list if not isinstance(r, CartesianRestraint)]

    if cartesian_restraints:
        cartesian_force = force_dict['cartesian']
        for index, r in enumerate(cartesian_restraints):
            weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
            cartesian_force.setParticleParameters(index, r.atom_index - 1, [r.x, r.y, r.z, r.delta, weight])
    return other_restraints


def _add_yzcartesian_restraints(system, restraint_list, alpha, timestep, force_dict):
    # split restraints into confinement and others
    cartesian_restraints = [r for r in restraint_list if isinstance(r, YZCartesianRestraint)]
    noncartesian_restraints = [r for r in restraint_list if not isinstance(r, YZCartesianRestraint)]

    if cartesian_restraints:
        # create the confinement force
        cartesian_force = CustomExternalForce(
            '0.5 * cart_force_const * r_eff^2; r_eff = max(0.0, r - cart_delta);'
            'r = sqrt(dy*dy + dz*dz);'
            'dy = y - cart_y;'
            'dz = z - cart_z;')
        cartesian_force.addPerParticleParameter('cart_y')
        cartesian_force.addPerParticleParameter('cart_z')
        cartesian_force.addPerParticleParameter('cart_delta')
        cartesian_force.addPerParticleParameter('cart_force_const')

        # add the atoms
        for r in cartesian_restraints:
            weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
            cartesian_force.addParticle(r.atom_index - 1, [r.y, r.z, r.delta, weight])
        system.addForce(cartesian_force)
        force_dict['yzcartesian'] = cartesian_force
    else:
        force_dict['yzcartesian'] = None

    return noncartesian_restraints


def _update_yzcartesian_restraints(restraint_list, alpha, timestep, force_dict):
    # split restraints into confinement and others
    cartesian_restraints = [r for r in restraint_list if isinstance(r, YZCartesianRestraint)]
    other_restraints = [r for r in restraint_list if not isinstance(r, YZCartesianRestraint)]

    if cartesian_restraints:
        cartesian_force = force_dict['yzcartesian']
        for index, r in enumerate(cartesian_restraints):
            weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
            cartesian_force.setParticleParameters(index, r.atom_index - 1, [r.y, r.z, r.delta, weight])
    return other_restraints


def _add_xcom_restraints(system, restraint_list, alpha, timestep, force_dict):
    # split restraints into confinement and others
    my_restraints = [r for r in restraint_list if isinstance(r, XAxisCOMRestraint)]
    other_restraints = [r for r in restraint_list if not isinstance(r, XAxisCOMRestraint)]

    if len(my_restraints) == 1:
        rest = my_restraints[0]
        weight = rest.force_const * rest.scaler(alpha) * rest.ramp(timestep)
        force = OneDimComForce(rest.indices1,
                               rest.indices2,
                               rest.weights1,
                               rest.weights2,
                               rest.force_const,
                               rest.positioner(alpha))

        system.addForce(force)
        force_dict['xcom'] = force
    elif len(my_restraints) == 0:
        force_dict['xcom'] = None
    else:
        raise RuntimeError('Cannot have more than one XAxisCOMRestraint')

    return other_restraints


def _update_xcom_restraints(restraint_list, alpha, timestep, force_dict):
    # split restraints into confinement and others
    my_restraints = [r for r in restraint_list if isinstance(r, XAxisCOMRestraint)]
    other_restraints = [r for r in restraint_list if not isinstance(r, XAxisCOMRestraint)]

    if my_restraints:
        force = force_dict['xcom']
        rest = my_restraints[0]
        weight = rest.force_const * rest.scaler(alpha) * rest.ramp(timestep)
        position = rest.positioner(alpha)
        force.setForceConst(weight)
        force.setR0(position)
    return other_restraints


def _add_rdc_restraints(system, restraint_list, alpha, timestep, force_dict):
    # split restraints into rdc and non-rdc
    rdc_restraint_list = [r for r in restraint_list if isinstance(r, RdcRestraint)]
    nonrdc_restraint_list = [r for r in restraint_list if not isinstance(r, RdcRestraint)]

    # if we have any rdc restraints
    if rdc_restraint_list:
        rdc_force = RdcForce()
        # make a dictionary based on the experiment index
        expt_dict = DefaultOrderedDict(list)
        for r in rdc_restraint_list:
            #expt_dict.get(r.expt_index, []).append(r)
            expt_dict[r.expt_index].append(r)

        # loop over the experiments and add the restraints to openmm
        for experiment in expt_dict:
            rests = expt_dict[experiment]
            rest_ids = []
            for r in rests:
                scale = r.scaler(alpha) * r.ramp(timestep)
                r_id = rdc_force.addRdcRestraint(
                    r.atom_index_1 - 1,
                    r.atom_index_2 - 1,
                    r.kappa, r.d_obs, r.tolerance,
                    r.force_const * scale, r.weight)
                rest_ids.append(r_id)
            rdc_force.addExperiment(rest_ids)

        system.addForce(rdc_force)
        force_dict['rdc'] = rdc_force
    else:
        force_dict['rdc'] = None

    return nonrdc_restraint_list


def _update_rdc_restraints(restraint_list, alpha, timestep, force_dict):
    # split restraints into rdc and non-rdc
    rdc_restraint_list = [r for r in restraint_list if isinstance(r, RdcRestraint)]
    nonrdc_restraint_list = [r for r in restraint_list if not isinstance(r, RdcRestraint)]

    # if we have any rdc restraints
    if rdc_restraint_list:
        rdc_force = force_dict['rdc']
        # make a dictionary based on the experiment index
        expt_dict = OrderedDict()
        for r in rdc_restraint_list:
            expt_dict.get(r.expt_index, []).append(r)

        # loop over the experiments and update the restraints
        index = 0
        for experiment in expt_dict:
            rests = expt_dict[experiment]
            for r in rests:
                scale = r.scaler(alpha) * r.ramp(timestep)
                rdc_force.updateRdcRestraint(
                    index,
                    r.atom_index_1 - 1,
                    r.atom_index_2 - 1,
                    r.kappa, r.d_obs, r.tolerance,
                    r.force_const * scale, r.weight)
                index = index + 1

    return nonrdc_restraint_list


def _add_meld_restraint(rest, meld_force, alpha, timestep):
    scale = rest.scaler(alpha) * rest.ramp(timestep)
    if isinstance(rest, DistanceRestraint):
        rest_index = meld_force.addDistanceRestraint(rest.atom_index_1 - 1, rest.atom_index_2 - 1,
                                                    rest.r1, rest.r2, rest.r3, rest.r4,
                                                    rest.k * scale)

    elif isinstance(rest, HyperbolicDistanceRestraint):
        rest_index = meld_force.addHyperbolicDistanceRestraint(rest.atom_index_1 - 1, rest.atom_index_2 - 1,
                                                               rest.r1, rest.r2, rest.r3, rest.r4,
                                                               rest.k * scale, rest.asymptote * scale)

    elif isinstance(rest, TorsionRestraint):
        rest_index = meld_force.addTorsionRestraint(rest.atom_index_1 - 1, rest.atom_index_2 - 1,
                                                    rest.atom_index_3 - 1, rest.atom_index_4 - 1,
                                                    rest.phi, rest.delta_phi, rest.k * scale)
    elif isinstance(rest, DistProfileRestraint):
        rest_index = meld_force.addDistProfileRestraint(rest.atom_index_1 - 1, rest.atom_index_2 - 1,
                                                        rest.r_min, rest.r_max, rest.n_bins,
                                                        rest.spline_params[:, 0], rest.spline_params[:, 1],
                                                        rest.spline_params[:, 2], rest.spline_params[:, 3],
                                                        rest.scale_factor * scale)
    elif isinstance(rest, TorsProfileRestraint):
        rest_index = meld_force.addTorsProfileRestraint(rest.atom_index_1 - 1, rest.atom_index_2 - 1,
                                                        rest.atom_index_3 - 1, rest.atom_index_4 - 1,
                                                        rest.atom_index_5 - 1, rest.atom_index_6 - 1,
                                                        rest.atom_index_7 - 1, rest.atom_index_8 - 1,
                                                        rest.n_bins,
                                                        rest.spline_params[:, 0], rest.spline_params[:, 1],
                                                        rest.spline_params[:, 2], rest.spline_params[:, 3],
                                                        rest.spline_params[:, 4], rest.spline_params[:, 5],
                                                        rest.spline_params[:, 6], rest.spline_params[:, 7],
                                                        rest.spline_params[:, 8], rest.spline_params[:, 9],
                                                        rest.spline_params[:, 10], rest.spline_params[:, 11],
                                                        rest.spline_params[:, 12], rest.spline_params[:, 13],
                                                        rest.spline_params[:, 14], rest.spline_params[:, 15],
                                                        rest.scale_factor * scale)
    else:
        raise RuntimeError('Do not know how to handle restraint {}'.format(rest))
    return rest_index


def _update_meld_restraint(rest, meld_force, alpha, timestep, dist_index, hyper_index, tors_index, dist_prof_index, tors_prof_index):
    scale = rest.scaler(alpha) * rest.ramp(timestep)
    if isinstance(rest, DistanceRestraint):
        meld_force.modifyDistanceRestraint(dist_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1, rest.r1,
                                           rest.r2, rest.r3, rest.r4, rest.k * scale)
        dist_index += 1
    elif isinstance(rest, HyperbolicDistanceRestraint):
        meld_force.modifyHyperbolicDistanceRestraint(hyper_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1,
                                                     rest.r1, rest.r2, rest.r3, rest.r4, rest.k * scale,
                                                     rest.asymptote * scale)
        hyper_index +=1
    elif isinstance(rest, TorsionRestraint):
        meld_force.modifyTorsionRestraint(tors_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1, rest.atom_index_3 - 1,
                                          rest.atom_index_4 - 1, rest.phi, rest.delta_phi, rest.k * scale)
        tors_index += 1
    elif isinstance(rest, DistProfileRestraint):
        meld_force.modifyDistProfileRestraint(dist_prof_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1,
                                              rest.r_min, rest.r_max, rest.n_bins,
                                              rest.spline_params[:, 0], rest.spline_params[:, 1],
                                              rest.spline_params[:, 2], rest.spline_params[:, 3],
                                              rest.scale_factor * scale)
        dist_prof_index += 1
    elif isinstance(rest, TorsProfileRestraint):
        meld_force.modifyTorsProfileRestraint(tors_prof_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1,
                                              rest.atom_index_3 - 1, rest.atom_index_4 - 1,
                                              rest.atom_index_5 - 1, rest.atom_index_6 - 1,
                                              rest.atom_index_7 - 1, rest.atom_index_8 - 1,
                                              rest.n_bins,
                                              rest.spline_params[:, 0], rest.spline_params[:, 1],
                                              rest.spline_params[:, 2], rest.spline_params[:, 3],
                                              rest.spline_params[:, 4], rest.spline_params[:, 5],
                                              rest.spline_params[:, 6], rest.spline_params[:, 7],
                                              rest.spline_params[:, 8], rest.spline_params[:, 9],
                                              rest.spline_params[:, 10], rest.spline_params[:, 11],
                                              rest.spline_params[:, 12], rest.spline_params[:, 13],
                                              rest.spline_params[:, 14], rest.spline_params[:, 15],
                                              rest.scale_factor * scale)
        tors_prof_index += 1
    else:
        raise RuntimeError('Do not know how to handle restraint {}'.format(rest))
    return dist_index, hyper_index, tors_index, dist_prof_index, tors_prof_index


#
# ordered, default dictionary to hold list of restraints

from collections import OrderedDict, Callable


class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory,
                          copy.deepcopy(self.items()))

    def __repr__(self):
        return 'OrderedDefaultDict(%s, %s)' % (self.default_factory,
                                        OrderedDict.__repr__(self))
