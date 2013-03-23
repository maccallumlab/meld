class FakeSystem(object):
    def get_runner(self):
        return FakeSystemRunner()


class FakeSystemRunner(object):
    def initialize(self):
        pass

    def minimize_then_run(self, state):
        return state

    def run(self, state):
        return state

    def get_energy(self, state):
        return 0.

    def set_lambda(self, lambda_):
        pass
