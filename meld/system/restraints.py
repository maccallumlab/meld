import math


class RestraintRegistry(type):
    """
    Metaclass that maintains a registry of restraint types.

    All classes that decend from Restraint inherit RestraintRegistry as their
    metaclass. RestraintRegistry will automatically maintain a map between
    the class attribute '_restraint_key_' and all restraint types.

    The function get_constructor_for_key is used to get the class for the
    corresponding key.
    """
    _restraint_registry = {}

    def __init__(cls, name, bases, attrs):
        if name in ['Restraint', 'SelectableRestraint', 'NonSelectableRestraint']:
            pass    # we don't register the base classes
        else:
            try:
                key = attrs['_restraint_key_']
            except KeyError:
                raise RuntimeError(
                    'Restraint type {} subclasses Restraint, but does not set _restraint_key_'.format(name))
            if key in RestraintRegistry._restraint_registry:
                raise RuntimeError(
                    'Trying to register two different classes with _restraint_key_ = {}.'.format(key))
            RestraintRegistry._restraint_registry[key] = cls

    @classmethod
    def get_constructor_for_key(self, key):
        """Get the constructor for the restraint type matching key."""
        try:
            return RestraintRegistry._restraint_registry[key]
        except KeyError:
            raise RuntimeError(
                'Unknown restraint type "{}".'.format(key))


class Restraint(object):
    """Abstract class for all restraints."""
    __metaclass__ = RestraintRegistry


class SelectableRestraint(Restraint):
    """Abstract class for selectable restraints."""
    pass


class NonSelectableRestraint(Restraint):
    """Abstract class for non-selectable restraints."""
    pass


class DistanceRestraint(SelectableRestraint):
    """
    Distance restraint

    :param system: a System object
    :param scaler: a force scaler
    :param atom_1_res_index: integer, starting from 1
    :param atom_1_name: atom name
    :param atom_2_res_index: integer, starting from 1
    :param atom_2_name: atom name
    :param r1: in nanometers
    :param r2: in nanometers
    :param r3: in nanometers
    :param r4: in nanometers
    :param k: in :math:`kJ/mol/nm^2`
    """

    _restraint_key_ = 'distance'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name, atom_2_res_index, atom_2_name,
                 r1, r2, r3, r4, k):
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.k = k
        self.scaler = scaler
        self.ramp = ramp
        self._check(system)

    def _check(self, system):
        if self.r1 < 0 or self.r2 < 0 or self.r3 < 0 or self.r4 < 0:
            raise RuntimeError('r1 to r4 must be > 0. r1={} r2={} r3={} r4={}.'.format(
                self.r1, self.r2, self.r3, self.r4))
        if self.r2 < self.r1:
            raise RuntimeError('r2 must be >= r1. r1={} r2={}.'.format(self.r1, self.r2))
        if self.r3 < self.r2:
            raise RuntimeError('r3 must be >= r2. r2={} r3={}.'.format(self.r2, self.r3))
        if self.r4 < self.r3:
            raise RuntimeError('r4 must be >= r3. r3={} r4={}.'.format(self.r3, self.r4))
        if self.k < 0:
            raise RuntimeError('k must be >= 0. k={}.'.format(self.k))


class TorsionRestraint(SelectableRestraint):
    """
    A torsion restraint

    :param system: System
    :param scaler:  force scaler
    :param atom_1_res_index: integer, starting from 1
    :param atom_1_name: atom name
    :param atom_2_res_index: integer, starting from 1
    :param atom_2_name: atom name
    :param atom_3_res_index: integer, starting from 1
    :param atom_3_name: atom name
    :param atom_4_res_index: integer, starting from 1
    :param atom_4_name: atom name
    :param phi: equilibrium value, degrees
    :param delta_phi: flat within delta_phi, degrees
    :param k: :math:`kJ/mol/degree^2`
    """

    _restraint_key_ = 'torsion'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name, atom_2_res_index, atom_2_name,
                 atom_3_res_index, atom_3_name, atom_4_res_index, atom_4_name,
                 phi, delta_phi, k):

        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.atom_index_3 = system.index_of_atom(atom_3_res_index, atom_3_name)
        self.atom_index_4 = system.index_of_atom(atom_4_res_index, atom_4_name)
        self.phi = phi
        self.delta_phi = delta_phi
        self.k = k
        self.scaler = scaler
        self.ramp = ramp
        self._check()

    def _check(self):
        if len(set([self.atom_index_1, self.atom_index_2, self.atom_index_3, self.atom_index_4])) != 4:
            raise RuntimeError('All four indices of a torsion restraint must be unique.')
        if self.phi < -180 or self.phi > 180:
            raise RuntimeError('-180 <= phi <= 180. phi was {}.'.format(self.phi))
        if self.delta_phi < 0 or self.delta_phi > 180:
            raise RuntimeError('0 <= delta_phi < 180. delta_phi was {}.'.format(self.delta_phi))
        if self.k < 0:
            raise RuntimeError('k >= 0. k was {}.'.format(self.k))


class DistProfileRestraint(SelectableRestraint):
    _restraint_key_ = 'dist_prof'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name, atom_2_res_index, atom_2_name,
                 r_min, r_max, n_bins, spline_params, scale_factor):
        self.scaler = scaler
        self.ramp = ramp
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.r_min = r_min
        self.r_max = r_max
        self.n_bins = n_bins
        self.spline_params = spline_params
        self.scale_factor = scale_factor
        self._check()

    def _check(self):
        assert self.r_min >= 0.
        assert self.r_max > self.r_min
        assert self.n_bins > 0
        assert self.spline_params.shape[0] == self.n_bins
        assert self.spline_params.shape[1] == 4


class TorsProfileRestraint(SelectableRestraint):
    _restraint_key_ = 'tors_prof'

    def __init__(self, system, scaler, ramp,
                 atom_1_res_index, atom_1_name, atom_2_res_index, atom_2_name,
                 atom_3_res_index, atom_3_name, atom_4_res_index, atom_4_name,
                 atom_5_res_index, atom_5_name, atom_6_res_index, atom_6_name,
                 atom_7_res_index, atom_7_name, atom_8_res_index, atom_8_name,
                 n_bins, spline_params, scale_factor):
        self.scaler = scaler
        self.ramp = ramp
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.atom_index_3 = system.index_of_atom(atom_3_res_index, atom_3_name)
        self.atom_index_4 = system.index_of_atom(atom_4_res_index, atom_4_name)
        self.atom_index_5 = system.index_of_atom(atom_5_res_index, atom_5_name)
        self.atom_index_6 = system.index_of_atom(atom_6_res_index, atom_6_name)
        self.atom_index_7 = system.index_of_atom(atom_7_res_index, atom_7_name)
        self.atom_index_8 = system.index_of_atom(atom_8_res_index, atom_8_name)
        self.n_bins = n_bins
        self.spline_params = spline_params
        self.scale_factor = scale_factor
        self._check()

    def _check(self):
        assert self.n_bins > 0
        n_params = self.n_bins * self.n_bins
        assert self.spline_params.shape[0] == n_params
        assert self.spline_params.shape[1] == 16


class RdcRestraint(NonSelectableRestraint):
    """
    Residual Dipolar Coupling Restraint

    :param system: a System object
    :param scaler: a force scaler
    :param atom_1_res_index: integer, starting from 1
    :param atom_1_name: atom name
    :param atom_2_res_index:  integer, starting from 1
    :param atom_2_name: atom name
    :param kappa: prefactor for RDC calculation in :math:`Hz / Angstrom^3`
    :param d_obs: observed dipolar coupling in Hz
    :param tolerance: calculed couplings within tolerance (in Hz) of d_obs will have zero energy and force
    :param force_const: force sonstant in :math:`kJ/mol/Hz^2`
    :param weight: dimensionless weight to place on this restraint
    :param expt_index: integer experiment id

    Typical values for kappa are:

    - 1H - 1H: :math:`-360300 \ Hz / Angstrom^3`
    - 13C - 1H: :math:`-90600 \ Hz / Angstrom^3`
    - 15N - 1H: :math:`36500 \ Hz / Angstrom^3`

    """

    _restraint_key_ = 'rdc'

    def __init__(self, system, scaler, ramp, atom_1_res_index, atom_1_name, atom_2_res_index, atom_2_name,
                 kappa, d_obs, tolerance, force_const, weight, expt_index):
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.kappa = float(kappa)
        self.d_obs = float(d_obs)
        self.tolerance = float(tolerance)
        self.force_const = float(force_const)
        self.weight = float(weight)
        self.expt_index = int(expt_index)
        self.scaler = scaler
        self.ramp = ramp
        self._check(system)

    def _check(self, system):
        if self.atom_index_1 == self.atom_index_2:
            raise ValueError('atom1 and atom2 must be different')
        if self.tolerance < 0:
            raise ValueError('tolerance must be > 0')
        if self.force_const < 0:
            raise ValueError('force_constant must be > 0')
        if self.weight < 0:
            raise ValueError('weight must be > 0')


class ConfinementRestraint(NonSelectableRestraint):
    """
    Confinement restraint

    :param system: a System object
    :param scaler: a force scaler
    :param res_index: integer, starting from 1
    :param atom_name: atom name
    :param raidus: calculed couplings within tolerance (in Hz) of d_obs will have zero energy and force
    :param force_const: force sonstant in :math:`kJ/mol/Hz^2`

    Confines an atom to be within radius of the origin. These restraints are typically set to somewhat
    larger than the expected radius of gyration of the protein and help to keep the structures comapct
    even when the protein is unfolded. Typically used with a ConstantScaler.

    """

    _restraint_key_ = 'confine'

    def __init__(self, system, scaler, ramp, res_index, atom_name, radius, force_const):
        self.atom_index = system.index_of_atom(res_index, atom_name)
        self.radius = float(radius)
        self.force_const = float(force_const)
        self.scaler = scaler
        self.ramp = ramp
        self._check(system)

    def _check(self, system):
        if self.radius < 0:
            raise ValueError('radius must be > 0')
        if self.force_const < 0:
            raise ValueError('force_constant must be > 0')


class CartesianRestraint(NonSelectableRestraint):
    _restraint_key_ = 'cartesian'

    def __init__(self, system, scaler, ramp, res_index, atom_name, x, y, z, delta, force_const):
        self.atom_index = system.index_of_atom(res_index, atom_name)
        self.x = x
        self.y = y
        self.z = z
        self.delta = delta
        self.force_const = force_const
        self.scaler = scaler
        self.ramp = ramp
        self._check()

    def _check(self):
        if self.delta < 0:
            raise ValueError('delta must be non-negative')
        if self.force_const < 0:
            raise ValueError('force_const must be non-negative')


class AlwaysActiveCollection(object):
    '''
    '''
    def __init__(self):
        self._restraints = []

    @property
    def restraints(self):
        return self._restraints

    def add_restraint(self, restraint):
        if not isinstance(restraint, Restraint):
            raise RuntimeError('Tried to add unknown restraint of type {}.'.format(str(type(restraint))))
        self._restraints.append(restraint)


class SelectivelyActiveCollection(object):
    '''
    '''
    def __init__(self, restraint_list, num_active):
        self._groups = []
        if not restraint_list:
            raise RuntimeError('SelectivelyActiveCollection cannot have empty restraint list.')
        for rest in restraint_list:
            self._add_restraint(rest)

        if num_active < 0:
            raise RuntimeError('num_active must be >= 0.')
        n_rest = len(self._groups)
        if num_active > n_rest:
            raise RuntimeError('num active must be <= num_groups ({}).'.format(n_rest))
        self._num_active = num_active

    @property
    def groups(self):
        return self._groups

    @property
    def num_active(self):
        return self._num_active

    def _add_restraint(self, restraint):
        if isinstance(restraint, RestraintGroup):
            self._groups.append(restraint)
        elif not isinstance(restraint, SelectableRestraint):
            raise RuntimeError('Cannot add restraint of type {} to SelectivelyActiveCollection'.format(
                str(type(restraint))))
        else:
            group = RestraintGroup([restraint], 1)
            self._groups.append(group)


class RestraintGroup(object):
    def __init__(self, rest_list, num_active):
        self._restraints = []
        if not rest_list:
            raise RuntimeError('rest_list cannot be empty.')
        for rest in rest_list:
            self._add_restraint(rest)

        if num_active < 0:
            raise RuntimeError('num_active must be >= 0.')
        n_rest = len(self._restraints)
        if num_active > n_rest:
            raise RuntimeError('num_active must be <= n_rest ({}).'.format(n_rest))
        self._num_active = num_active

    @property
    def restraints(self):
        return self._restraints

    @property
    def num_active(self):
        return self._num_active

    def _add_restraint(self, rest):
        if not isinstance(rest, SelectableRestraint):
            raise RuntimeError('Can only add SelectableRestraints to a RestraintGroup.')
        self._restraints.append(rest)


class RestraintManager(object):
    '''
    '''
    def __init__(self, system):
        self._system = system
        self._always_active = AlwaysActiveCollection()
        self._selective_collections = []

    @property
    def always_active(self):
        return self._always_active.restraints

    @property
    def selectively_active_collections(self):
        return self._selective_collections

    def add_as_always_active(self, restraint):
        self._always_active.add_restraint(restraint)

    def add_as_always_active_list(self, restraint_list):
        for r in restraint_list:
            self.add_as_always_active(r)

    def add_selectively_active_collection(self, rest_list, num_active):
        self._selective_collections.append(SelectivelyActiveCollection(rest_list, num_active))

    def create_restraint(self, rest_type, scaler=None, ramp=None, **kwargs):
        if scaler is None:
            scaler = ConstantScaler()
        else:
            if not isinstance(scaler, RestraintScaler):
                raise ValueError('scaler must be a subclass of RestraintScaler, you tried to add a {}.'.format(type(scaler)))

        if ramp is None:
            ramp = ConstantRamp()
        else:
            if not isinstance(ramp, TimeRamp):
                raise ValueError('ramp must be a subclass of TimeRamp, you tried to add a {}.'.format(type(ramp)))

        return RestraintRegistry.get_constructor_for_key(rest_type)(self._system, scaler, ramp, **kwargs)

    def create_restraint_group(self, rest_list, num_active):
        return RestraintGroup(rest_list, num_active)

    def create_scaler(self, scaler_type, **kwargs):
        return ScalerRegistry.get_constructor_for_key(scaler_type)(**kwargs)


class ScalerRegistry(type):
    '''
    Metaclass that maintains a registry of scaler types.

    All classes that decend from Scaler inherit ScalerRegistry as their
    metaclass. ScalerRegistry will automatically maintain a map between
    the class attribute '_scaler_key_' and all scaler types.

    The function get_constructor_for_key is used to get the class for the
    corresponding key.
    '''
    _scaler_registry = {}

    def __init__(cls, name, bases, attrs):
        if name in ['AlphaMapper', 'RestraintScaler', 'TimeRamp', 'Positioner']:
            pass    # we don't register the base classes
        else:
            try:
                key = attrs['_scaler_key_']
            except KeyError:
                raise RuntimeError(
                    'Scaler type {} subclasses Scaler, but does not set _scaler_key_'.format(name))
            if key in ScalerRegistry._scaler_registry:
                raise RuntimeError(
                    'Trying to register two different classes with _scaler_key_ = {}.'.format(key))
            ScalerRegistry._scaler_registry[key] = cls

    @classmethod
    def get_constructor_for_key(self, key):
        '''Get the constructor for the scaler type matching key.'''
        try:
            return ScalerRegistry._scaler_registry[key]
        except KeyError:
            raise RuntimeError(
                'Unknown scaler type "{}".'.format(key))


class AlphaMapper(object):
    '''Base class for all scalers.'''
    __metaclass__ = ScalerRegistry

    def _check_alpha_range(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError('0 >= alpha >= 1. alpha is {}.'.format(alpha))

    def _handle_boundaries(self, alpha):
        if alpha <= self._alpha_min:
            return 1.
        elif alpha >= self._alpha_max:
            return 0.
        else:
            return None

    def _check_alpha_min_max(self):
        if self._alpha_min < 0 or self._alpha_min > 1 or self._alpha_max < 0 or self._alpha_max > 1:
            raise RuntimeError('alpha_min and alpha_max must be in range [0, 1]. alpha_min={} alpha_max={}.'.format(
                self._alpha_min, self._alpha_max))
        if self._alpha_min >= self._alpha_max:
            raise RuntimeError('alpha_max must be less than alpha_min. alpha_min={} alpha_max={}.'.format(
                self._alpha_min, self._alpha_max))


class RestraintScaler(AlphaMapper):
    '''Base class for all resraint scaler classes.'''


class ConstantScaler(RestraintScaler):
    '''This scaler is "always on" and always returns a value of 1.0".'''

    _scaler_key_ = 'constant'

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        return 1.0


class LinearScaler(RestraintScaler):
    '''This scaler linearly interpolates between 0 and 1 from alpha_min to alpha_max.'''

    _scaler_key_ = 'linear'

    def __init__(self, alpha_min, alpha_max, strength_at_alpha_min=1.0, strength_at_alpha_max=0.0):
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._delta = alpha_max - alpha_min
        self._check_alpha_min_max()

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        scale = self._handle_boundaries(alpha)
        if scale is None:
            scale = 1.0 - (alpha - self._alpha_min) / self._delta
        scale = (1.0 - scale) * (self._strength_at_alpha_max - self._strength_at_alpha_min) + self._strength_at_alpha_min
        return scale


class NonLinearScaler(RestraintScaler):
    '''
    '''

    _scaler_key_ = 'nonlinear'

    def __init__(self, alpha_min, alpha_max, factor, strength_at_alpha_min=1.0, strength_at_alpha_max=0.0):
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()
        if factor < 1:
            raise RuntimeError('factor must be >= 1. factor={}.'.format(factor))
        self._factor = factor

    def __call__(self, alpha):
        self._check_alpha_range(alpha)
        scale = self._handle_boundaries(alpha)
        if scale is None:
            delta = (alpha - self._alpha_min) / (self._alpha_max - self._alpha_min)
            norm = 1.0 / (math.exp(self._factor) - 1.0)
            scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
        scale = (1.0 - scale) * (self._strength_at_alpha_max - self._strength_at_alpha_min) + self._strength_at_alpha_min
        return scale


class GeometricScaler(RestraintScaler):
    _scaler_key_ = 'geometric'

    def __init__(self, alpha_min, alpha_max, strength_at_alpha_min, strength_at_alpha_max):
        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = float(strength_at_alpha_min)
        self._strength_at_alpha_max = float(strength_at_alpha_max)
        self._delta_alpha = self._alpha_max - self._alpha_min
        self._check_alpha_min_max()

    def __call__(self, alpha):
        self._check_alpha_range(alpha)

        if alpha < 0 or alpha > 1:
            raise RuntimeError('0 <= alpha <=1 1')

        elif alpha <= self._alpha_min:
            return self._strength_at_alpha_min

        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            delta = math.log(self._strength_at_alpha_max) - math.log(self._strength_at_alpha_min)
            return math.exp(delta * frac + math.log(self._strength_at_alpha_min))

        else:
            return self._strength_at_alpha_max


class TimeRamp(AlphaMapper):
    '''Base class for all time ramp classes.'''


class ConstantRamp(TimeRamp):
    '''TimeRamp that always returns 1.0'''
    _scaler_key_ = 'constant_ramp'

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError('Timestep is < 0.')
        return 1.0


class LinearRamp(TimeRamp):
    '''TimeRamp that interpolates linearly'''

    _scaler_key_ = 'linear_ramp'

    def __init__(self, start_time, end_time, start_weight, end_weight):
        self.t_start = float(start_time)
        self.t_end = float(end_time)
        self.w_start = float(start_weight)
        self.w_end = float(end_weight)

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError('Timestep is < 0.')
        if timestep < self.t_start:
            return self.w_start
        elif timestep < self.t_end:
            return (self.w_start + (self.w_end - self.w_start) *
                    (float(timestep) - self.t_start) / (self.t_end - self.t_start))
        else:
            return self.w_end


class NonLinearRamp(TimeRamp):
    '''TimeRamp that interpolates non-linearly'''

    _scaler_key_ = 'nonlinear_ramp'

    def __init__(self, start_time, end_time, start_weight, end_weight, factor):
        if end_time <= start_time:
            raise ValueError('end_time must be > start_time')
        if factor < 1.0:
            raise ValueError('factor myst be > 1.0')

        self.t_start = float(start_time)
        self.t_end = float(end_time)
        self.w_start = float(start_weight)
        self.w_end = float(end_weight)
        self.factor = float(factor)

    def __call__(self, timestep):
        if timestep < 0:
            raise ValueError('timestep is < 0.')

        if timestep < self.t_start:
            return self.w_start
        elif timestep < self.t_end:
            # we scale differently depending on if we are ramping up or down
            # we change more slowly at lower values and more rapidly at
            # higher values
            #
            # this is for scaling up
            if self.w_end > self.w_start:
                delta = 1.0 - (float(timestep) - self.t_start) / (self.t_end - self.t_start)
                norm = 1.0 / (math.exp(self.factor) - 1.0)
                scale = norm * (math.exp(self.factor * (1.0 - delta)) - 1.0)
                return scale * (self.w_end - self.w_start) + self.w_start
            # this is for scaling down
            else:
                delta = (float(timestep) - self.t_start) / (self.t_end - self.t_start)
                norm = 1.0 / (math.exp(self.factor) - 1.0)
                scale = norm * (math.exp(self.factor * (1.0 - delta)) - 1.0)
                return (1.0 - scale) * (self.w_end - self.w_start) + self.w_start
        else:
            return self.w_end


class TimeRampSwitcher(TimeRamp):
    '''
    Switches between two TimeRamp objects.

    Class first_ramp before switching time. At the switching
    time it switches to second_ramp, which it uses thereafter.
    '''

    _scaler_key_ = 'ramp_switcher'

    def __init__(self, first_ramp, second_ramp, switching_time):
        self.first_ramp = first_ramp
        self.second_ramp = second_ramp
        self.switching_time = switching_time

    def __call__(self, timestep):
        if timestep < self.switching_time:
            return self.first_ramp(timestep)
        else:
            return self.second_ramp(timestep)


class Positioner(AlphaMapper):
    '''Base class for all positioner classes.'''


class ConstantPositioner(Positioner):
    '''Always returns the supplied value.'''

    _scaler_key_ = 'constant_positioner'

    def __init__(self, value):
        self._value = value

    def __call__(self, alpha):
        if alpha < 0:
            raise ValueError('alpha must be >= 0')
        if alpha > 1:
            raise ValueError('alpha must be <= 1')

        return self._value

class LinearPositioner(Positioner):
    '''Position restraints linearly within a range'''

    _scaler_key_ = 'linear_positioner'

    def __init__(self, alpha_min, alpha_max, pos_min, pos_max):
        if alpha_max <= alpha_min:
            raise ValueError('alpha_max must be > alpha_min')

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.pos_min = float(pos_min)
        self.pos_max = float(pos_max)

    def __call__(self, alpha):
        if alpha < 0:
            raise ValueError('alpha was < 0')
        if alpha > 1:
            raise ValueError('alpha was > 1')
        if alpha < self.alpha_min:
            return self.pos_min
        elif alpha < self.alpha_max:
            delta = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
            return delta * (self.pos_max - self.pos_min) + self.pos_min
        else:
            return self.pos_max
