from traits.api import HasTraits, Trait, BaseInt, BaseFloat, Enum, Bool


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
