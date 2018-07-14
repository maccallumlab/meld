#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld.system.openmm_runner import OpenMMRunner


class ReplicaRunner:
    def initialize(self):
        pass

    def minimize_then_run(self, state):
        pass

    def run(self, state):
        pass

    def get_energy(self, state):
        pass

    def prepare_for_timestep(self, alpha, timestep):
        pass


class FakeSystemRunner:
    """
    Fake runner for test purposes.
    """

    def __init__(self, system, options, communicator=None):
        self.temperature_scaler = system.temperature_scaler

    def set_alpha_and_timestep(self, alpha, timestep):
        pass

    def minimize_then_run(self, state):
        return state

    def run(self, state):
        return state

    def get_energy(self, state):
        return 0.

    def prepare_for_timestep(self, alpha, timestep):
        pass


def get_runner(system, options, comm):
    if options.runner == "openmm":
        return OpenMMRunner(system, options, comm)
    elif options.runner == "fake_runner":
        return FakeSystemRunner(system, options, comm)
    else:
        raise RuntimeError(f"Unknown type of runner: {options.runner}")
