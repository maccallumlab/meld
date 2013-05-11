class RestraintRegistry(type):
    '''
    Metaclass that maintains a registry of restraint types.

    All classes that decend from Restraint inherit RestraintRegistry as their
    metaclass. RestraintRegistry will automatically maintain a map between
    the class attribute '_restraint_key_' and all restraint types.

    The function get_constructor_for_key is used to get the class for the
    corresponding key.
    '''
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
        '''Get the constructor for the restraint type matching key.'''
        try:
            return RestraintRegistry._restraint_registry[key]
        except KeyError:
            raise RuntimeError(
                'Unknown restraint type "{}".'.format(key))


class Restraint(object):
    '''Abstract class for all restraints.'''
    __metaclass__ = RestraintRegistry


class SelectableRestraint(Restraint):
    '''Abstract class for selectable restraints.'''
    pass


class NonSelectableRestraint(Restraint):
    '''Abstract class for non-selectable restraints.'''
    pass


class DistanceRestraint(SelectableRestraint):
    '''
    '''

    _restraint_key_ = 'distance'

    def __init__(self, system, atom_1_res_index, atom_1_name, atom_2_res_index, atom_2_name,
                 r1, r2, r3, r4, k):
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.r1 = r1
        self.r2 = r2
        self.r3 = r3
        self.r4 = r4
        self.k = k
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
    '''
    '''

    _restraint_key_ = 'torsion'

    def __init__(self, system, atom_1_res_index, atom_1_name, atom_2_res_index, atom_2_name,
                 atom_3_res_index, atom_3_name, atom_4_res_index, atom_4_name,
                 phi, delta_phi, k):
        self.atom_index_1 = system.index_of_atom(atom_1_res_index, atom_1_name)
        self.atom_index_2 = system.index_of_atom(atom_2_res_index, atom_2_name)
        self.atom_index_3 = system.index_of_atom(atom_3_res_index, atom_3_name)
        self.atom_index_4 = system.index_of_atom(atom_4_res_index, atom_4_name)
        self.phi = phi
        self.delta_phi = delta_phi
        self.k = k
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
        self._restraints = []
        if not restraint_list:
            raise RuntimeError('SelectivelyActiveCollection cannot have empty restraint list.')
        for rest in restraint_list:
            self._add_restraint(rest)

        if num_active < 0:
            raise RuntimeError('num_active must be >= 0.')
        n_rest = len(self._restraints)
        if num_active > n_rest:
            raise RuntimeError('num active must be <= num_restraints ({}).'.format(n_rest))
        self._num_active = num_active

    @property
    def restraints(self):
        return self._restraints

    @property
    def num_active(self):
        return self._num_active

    def _add_restraint(self, restraint):
        if isinstance(restraint, RestraintGroup):
            self._restraints.append(restraint)
        elif not isinstance(restraint, SelectableRestraint):
            raise RuntimeError('Cannot add restraint of type {} to SelectivelyActiveCollection'.format(
                str(type(restraint))))
        else:
            group = RestraintGroup([restraint], 1)
            self._restraints.append(group)


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

    def create_restraint(self, rest_type, **kwargs):
        return RestraintRegistry.get_constructor_for_key(rest_type)(self._system, **kwargs)

    def create_restraint_group(self, rest_list, num_active):
        return RestraintGroup(rest_list, num_active)
