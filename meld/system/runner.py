from .openmm_runner import OpenMMRunner


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


def get_runner(system, options):
    if options.runner == 'openmm':
        return OpenMMRunner(system, options)
    elif options.runner == 'fake_runner':
        return FakeSystemRunner()
    else:
        raise RuntimeError('Unknown type of runner: {}'.format(options.runner))
