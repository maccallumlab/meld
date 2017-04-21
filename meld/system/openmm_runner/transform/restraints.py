#
# All rights reserved
#

'''
This module implements transformers that add restraint forces
to the openmm system before simulation
'''

from simtk.openmm import CustomExternalForce
from meld.system import restraints
from collections import OrderedDict
from meld.system.openmm_runner.transform import TransformerBase
from meldplugin import MeldForce
from simtk import openmm as mm


class ConfinementRestraintTransformer(TransformerBase):
    '''
    Transformer to handle confinement restraints

    '''
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        self.restraints = [r for r in always_active_restraints
                           if isinstance(r, restraints.ConfinementRestraint)]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, system, topology):
        if self.active:
            # create the confinement force
            confinement_force = CustomExternalForce(
                'step(r - radius) * force_const * (radius - r)^2;'
                'r=sqrt(x*x + y*y + z*z)')
            confinement_force.addPerParticleParameter('radius')
            confinement_force.addPerParticleParameter('force_const')

            # add the atoms
            for r in self.restraints:
                weight = r.force_const
                confinement_force.addParticle(r.atom_index - 1,
                                              [r.radius, weight])
            system.addForce(confinement_force)
            self.force = confinement_force

        return system

    def update(self, simulation, alpha, timestep):
        if self.active:
            for index, r in enumerate(self.restraints):
                weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
                self.force.setParticleParameters(
                    index, r.atom_index - 1, [r.radius, weight])
            self.force.updateParametersInContext(simulation.context)


class RDCRestraintTransformer(TransformerBase):
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        self.restraints = [r for r in always_active_restraints
                           if isinstance(r, restraints.RdcRestraint)]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, system, topology):
        if self.active:
            rdc_force = restraints.RdcForce()
            expt_dict = DefaultOrderedDict(list)
            # make a dictionary based on the experiment index
            for r in self.restraints:
                expt_dict[r.expt_index].append(r)

            # loop over the experiments and add the restraints to openmm
            for experiment in expt_dict:
                rests = expt_dict[experiment]
                rest_ids = []
                for r in rests:
                    r_id = rdc_force.addRdcRestraint(
                        r.atom_index_1 - 1,
                        r.atom_index_2 - 1,
                        r.kappa, r.d_obs, r.tolerance,
                        r.force_const, r.weight)
                    rest_ids.append(r_id)
                rdc_force.addExperiment(rest_ids)

            system.addForce(rdc_force)
            self.force = rdc_force
        return system

    def update(self, simulation, alpha, timestep):
        if self.active:
            # make a dictionary based on the experiment index
            expt_dict = OrderedDict()
            for r in self.restraints:
                expt_dict.get(r.expt_index, []).append(r)

            # loop over the experiments and update the restraints
            index = 0
            for experiment in expt_dict:
                rests = expt_dict[experiment]
                for r in rests:
                    scale = r.scaler(alpha) * r.ramp(timestep)
                    self.force.updateRdcRestraint(
                        index,
                        r.atom_index_1 - 1,
                        r.atom_index_2 - 1,
                        r.kappa, r.d_obs, r.tolerance,
                        r.force_const * scale, r.weight)
                    index = index + 1
            self.force.updateParametersInContext(simulation.context)


class CartesianRestraintTransformer(TransformerBase):
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        self.restraints = [r for r in always_active_restraints
                           if isinstance(r, restraints.CartesianRestraint)]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, system, topology):
        if self.active:
            # create the confinement force
            cartesian_force = CustomExternalForce(
                '0.5 * cart_force_const * r_eff^2;'
                'r_eff = max(0.0, r - cart_delta);'
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
            for r in self.restraints:
                weight = r.force_const
                cartesian_force.addParticle(r.atom_index - 1,
                                            [r.x, r.y, r.z, r.delta, weight])
            system.addForce(cartesian_force)
            self.force = cartesian_force
        return system

    def update(self, simulation, alpha, timestep):
        if self.active:
            for index, r in enumerate(self.restraints):
                weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
                self.force.setParticleParameters(index, r.atom_index - 1,
                                                 [r.x, r.y, r.z, r.delta, weight])
            self.force.updateParametersInContext(simulation.context)


class YZCartesianTransformer(TransformerBase):
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        self.restraints = [r for r in always_active_restraints
                           if isinstance(r, restraints.YZCartesianRestraint)]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, system, topology):
        if self.active:
            # create the confinement force
            cartesian_force = CustomExternalForce(
                '0.5 * cart_force_const * r_eff2;'
                'r_eff2 = max(0.0, r2 - cart_delta^2);'
                'r2 = dy*dy + dz*dz;'
                'dy = y - cart_y;'
                'dz = z - cart_z;')
            cartesian_force.addPerParticleParameter('cart_y')
            cartesian_force.addPerParticleParameter('cart_z')
            cartesian_force.addPerParticleParameter('cart_delta')
            cartesian_force.addPerParticleParameter('cart_force_const')

            # add the atoms
            for r in self.restraints:
                weight = r.force_const
                cartesian_force.addParticle(r.atom_index - 1,
                                            [r.y, r.z, r.delta, weight])
            system.addForce(cartesian_force)
            self.force = cartesian_force
        return system

    def update(self, simulation, alpha, timestep):
        if self.active:
            for index, r in enumerate(self.restraints):
                weight = r.force_const * r.scaler(alpha) * r.ramp(timestep)
                self.force.setParticleParameters(index, r.atom_index - 1,
                                                 [r.y, r.z, r.delta, weight])
            self.force.updateParametersInContext(simulation.context)


class COMRestraintTransformer(TransformerBase):
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        self.restraints = [r for r in always_active_restraints
                           if isinstance(r, restraints.COMRestraint)]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if len(self.restraints) > 1:
            raise RuntimeError('Cannot have more than one COMRestraint')

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, system, topology):
        if self.active:
            rest = self.restraints[0]
            # convert indices from 1-based to 0-based
            rest_indices1 = [r - 1 for r in rest.indices1]
            rest_indices2 = [r - 1 for r in rest.indices2]

            # create the expression for the energy
            components = []
            if 'x' in rest.dims:
                components.append('(x1-x2)*(x1-x2)')
            if 'y' in rest.dims:
                components.append('(y1-y2)*(y1-y2)')
            if 'z' in rest.dims:
                components.append('(z1-z2)*(z1-z2)')
            dist_expr = 'dist = sqrt({});'.format(' + '.join(components))
            energy_expr = '0.5 * com_k * (dist - com_ref_dist)*(dist-com_ref_dist);'
            expr = '\n'.join([energy_expr, dist_expr])

            # create the force
            force = mm.CustomCentroidBondForce(2, expr)
            force.addPerBondParameter('com_k')
            force.addPerBondParameter('com_ref_dist')

            # create the restraint with parameters
            if rest.weights1:
                g1 = force.addGroup(rest_indices1, rest.weights1)
            else:
                g1 = force.addGroup(rest_indices1)
            if rest.weights2:
                g2 = force.addGroup(rest_indices2, rest.weights2)
            else:
                g2 = force.addGroup(rest_indices2)
            force_const = rest.force_const
            pos = rest.positioner(0)
            force.addBond([g1, g2], [force_const, pos])

            system.addForce(force)
            self.force = force
        return system

    def update(self, simulation, alpha, timestep):
        if self.active:
            rest = self.restraints[0]
            weight = rest.force_const * rest.scaler(alpha) * rest.ramp(timestep)
            position = rest.positioner(alpha)
            groups, _ = self.force.getBondParameters(0)
            self.force.setBondParameters(0, groups, [weight, position])
            self.force.updateParametersInContext(simulation.context)


class AbsoluteCOMRestraintTransformer(TransformerBase):
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        self.restraints = [r for r in always_active_restraints
                           if isinstance(r, restraints.AbsoluteCOMRestraint)]
        _delete_from_always_active(self.restraints, always_active_restraints)

        if len(self.restraints) > 1:
            raise RuntimeError('Cannot have more than one AbsoluteCOMRestraint')

        if self.restraints:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, system, topology):
        if self.active:
            rest = self.restraints[0]
            # convert indices from 1-based to 0-based
            indices = [r - 1 for r in rest.indices]

            # create the expression for the energy
            components = []
            if 'x' in rest.dims:
                components.append('(x1-abscom_x)*(x1-abscom_x)')
            if 'y' in rest.dims:
                components.append('(y1-abscom_y)*(y1-abscom_y)')
            if 'z' in rest.dims:
                components.append('(z1-abscom_z)*(z1-abscom_z)')
            dist_expr = 'dist2={};'.format(' + '.join(components))
            energy_expr = '0.5 * com_k * dist2;'
            expr = '\n'.join([energy_expr, dist_expr])

            # create the force
            force = mm.CustomCentroidBondForce(1, expr)
            force.addPerBondParameter('com_k')
            force.addPerBondParameter('abscom_x')
            force.addPerBondParameter('abscom_y')
            force.addPerBondParameter('abscom_z')

            # create the restraint with parameters
            if rest.weights:
                g1 = force.addGroup(indices, rest.weights)
            else:
                g1 = force.addGroup(indices)
            force_const = rest.force_const
            pos_x = rest.position[0]
            pos_y = rest.position[1]
            pos_z = rest.position[2]
            force.addBond([g1], [force_const, pos_x, pos_y, pos_z])

            system.addForce(force)
            self.force = force
        return system

    def update(self, simulation, alpha, timestep):
        if self.active:
            rest = self.restraints[0]
            weight = rest.force_const * rest.scaler(alpha) * rest.ramp(timestep)
            pos_x = rest.position[0]
            pos_y = rest.position[1]
            pos_z = rest.position[2]
            groups, _ = self.force.getBondParameters(0)
            self.force.setBondParameters(0, groups, [weight, pos_x, pos_y, pos_z])
            self.force.updateParametersInContext(simulation.context)


class MeldRestraintTransformer(TransformerBase):
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        self.always_on = [r for r in always_active_restraints
                          if isinstance(r, restraints.SelectableRestraint)]
        _delete_from_always_active(self.always_on, always_active_restraints)

        self.selective_on = [r for r in selectively_active_restraints]
        for r in self.selective_on:
            selectively_active_restraints.remove(r)

        if self.always_on or self.selective_on:
            self.active = True
        else:
            self.active = False

        self.force = None

    def add_interactions(self, system, topology):
        if self.active:
            meld_force = MeldForce()
            if self.always_on:
                group_list = []
                for rest in self.always_on:
                    rest_index = _add_meld_restraint(rest, meld_force, 0, 0)
                    group_index = meld_force.addGroup([rest_index], 1)
                    group_list.append(group_index)
                meld_force.addCollection(group_list, len(group_list))
            for coll in self.selective_on:
                group_indices = []
                for group in coll.groups:
                    restraint_indices = []
                    for rest in group.restraints:
                        rest_index = _add_meld_restraint(rest, meld_force, 0, 0)
                        restraint_indices.append(rest_index)
                    group_index = meld_force.addGroup(restraint_indices, group.num_active)
                    group_indices.append(group_index)
                meld_force.addCollection(group_indices, coll.num_active)
            system.addForce(meld_force)
            self.force = meld_force
        return system

    def update(self, simulation, alpha, timestep):
        if self.active:
            dist_index = 0
            hyper_index = 0
            tors_index = 0
            dist_prof_index = 0
            tors_prof_index = 0
            gmm_index = 0
            if self.always_on:
                for rest in self.always_on:
                    (dist_index, hyper_index, tors_index,
                     dist_prof_index, tors_prof_index, gmm_index) = (
                        _update_meld_restraint(rest, self.force, alpha, timestep,
                                                dist_index, hyper_index, tors_index,
                                                dist_prof_index, tors_prof_index, gmm_index))
            for coll in self.selective_on:
                for group in coll.groups:
                    for rest in group.restraints:
                        (dist_index, hyper_index, tors_index,
                         dist_prof_index, tors_prof_index, gmm_index) = (
                            _update_meld_restraint(rest, self.force, alpha, timestep,
                                                    dist_index, hyper_index,
                                                    tors_index, dist_prof_index,
                                                    tors_prof_index, gmm_index))
            self.force.updateParametersInContext(simulation.context)


def _add_meld_restraint(rest, meld_force, alpha, timestep):
    scale = rest.scaler(alpha) * rest.ramp(timestep)

    if isinstance(rest, restraints.DistanceRestraint):
        rest_index = meld_force.addDistanceRestraint(
            rest.atom_index_1 - 1, rest.atom_index_2 - 1, rest.r1, rest.r2,
            rest.r3, rest.r4, rest.k * scale)

    elif isinstance(rest, restraints.HyperbolicDistanceRestraint):
        rest_index = meld_force.addHyperbolicDistanceRestraint(
            rest.atom_index_1 - 1, rest.atom_index_2 - 1, rest.r1, rest.r2,
            rest.r3, rest.r4, rest.k * scale, rest.asymptote * scale)

    elif isinstance(rest, restraints.TorsionRestraint):
        rest_index = meld_force.addTorsionRestraint(
            rest.atom_index_1 - 1, rest.atom_index_2 - 1,
            rest.atom_index_3 - 1, rest.atom_index_4 - 1,
            rest.phi, rest.delta_phi, rest.k * scale)

    elif isinstance(rest, restraints.DistProfileRestraint):
        rest_index = meld_force.addDistProfileRestraint(
            rest.atom_index_1 - 1, rest.atom_index_2 - 1, rest.r_min,
            rest.r_max, rest.n_bins, rest.spline_params[:, 0],
            rest.spline_params[:, 1], rest.spline_params[:, 2],
            rest.spline_params[:, 3], rest.scale_factor * scale)

    elif isinstance(rest, restraints.TorsProfileRestraint):
        rest_index = meld_force.addTorsProfileRestraint(
            rest.atom_index_1 - 1, rest.atom_index_2 - 1,
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

    elif isinstance(rest, restraints.GMMDistanceRestraint):
        nd = rest.n_distances
        nc = rest.n_components
        a = [a - 1 for a in rest.atoms]
        w = rest.weights
        m = list(rest.means.flatten())

        d, o = _setup_precisions(rest.precisions, nd, nc,)
        rest_index = meld_force.addGMMRestraint(nd, nc, scale, a, w, m, d, o)

    else:
        raise RuntimeError(
            'Do not know how to handle restraint {}'.format(rest))

    return rest_index


def _update_meld_restraint(rest, meld_force, alpha, timestep, dist_index,
                           hyper_index, tors_index, dist_prof_index,
                           tors_prof_index, gmm_index):
    scale = rest.scaler(alpha) * rest.ramp(timestep)

    if isinstance(rest, restraints.DistanceRestraint):
        meld_force.modifyDistanceRestraint(
            dist_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1, rest.r1,
            rest.r2, rest.r3, rest.r4, rest.k * scale)
        dist_index += 1

    elif isinstance(rest, restraints.HyperbolicDistanceRestraint):
        meld_force.modifyHyperbolicDistanceRestraint(
            hyper_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1, rest.r1,
            rest.r2, rest.r3, rest.r4, rest.k * scale, rest.asymptote * scale)
        hyper_index += 1

    elif isinstance(rest, restraints.TorsionRestraint):
        meld_force.modifyTorsionRestraint(
            tors_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1,
            rest.atom_index_3 - 1, rest.atom_index_4 - 1, rest.phi,
            rest.delta_phi, rest.k * scale)
        tors_index += 1

    elif isinstance(rest, restraints.DistProfileRestraint):
        meld_force.modifyDistProfileRestraint(
            dist_prof_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1,
            rest.r_min, rest.r_max, rest.n_bins,
            rest.spline_params[:, 0], rest.spline_params[:, 1],
            rest.spline_params[:, 2], rest.spline_params[:, 3],
            rest.scale_factor * scale)
        dist_prof_index += 1

    elif isinstance(rest, restraints.TorsProfileRestraint):
        meld_force.modifyTorsProfileRestraint(
            tors_prof_index, rest.atom_index_1 - 1, rest.atom_index_2 - 1,
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

    elif isinstance(rest, restraints.GMMDistanceRestraint):
        nd = rest.n_distances
        nc = rest.n_components
        a = [a - 1 for a in rest.atoms]
        w = rest.weights
        m = list(rest.means.flatten())
        d, o = _setup_precisions(rest.precisions, nd, nc)
        rest_index = meld_force.modifyGMMRestraint(gmm_index, nd, nc, scale, a, w, m, d, o)
        gmm_index += 1

    else:
        raise RuntimeError(
            'Do not know how to handle restraint {}'.format(rest))

    return (dist_index, hyper_index, tors_index,
            dist_prof_index, tors_prof_index, gmm_index)


def _setup_precisions(precisions, n_distances, n_conditions):
    # The normalization of our GMMs will blow up
    # due to division by zero if the precisions
    # are zero, so we clamp this to a very
    # small value.
    diags = []
    for i in range(n_conditions):
        for j in range(n_distances):
            diags.append(precisions[i, j, j,])

    off_diags = []
    for i in range(n_conditions):
        for j in range(n_distances):
            for k in range(j+1, n_distances):
                off_diags.append(precisions[i, j, k])

    return diags, off_diags


def _delete_from_always_active(restraints, always_active):
    for restraint in restraints:
        always_active.remove(restraint)


class DefaultOrderedDict(OrderedDict):
    def __init__(self, default_factory=None, *a, **kw):
        isnone = default_factory is None
        callable = isinstance(default_factory, Callable)
        if not isnone and not callable:
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
        return 'OrderedDefaultDict(%s, %s)'.format(
            self.default_factory, OrderedDict.__repr__(self))
