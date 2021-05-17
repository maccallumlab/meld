#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
A module that defines a base class for ReplicaRunners
"""

from .state import SystemState
from .temperature import TemperatureScaler
from typing import Optional


class ReplicaRunner:
    """
    Base ReplicaRunner
    """

    temperature_scaler: Optional[TemperatureScaler]

    def initialize(self) -> None:
        pass

    def minimize_then_run(self, state: SystemState) -> SystemState:
        pass

    def run(self, state: SystemState) -> SystemState:
        pass

    def get_energy(self, state: SystemState) -> float:
        pass

    def prepare_for_timestep(self, alpha, timestep):
        pass


class FakeSystemRunner(ReplicaRunner):
    """
    Fake runner for test purposes.
    """

    def __init__(
        self,
        system,
        options,
        communicator,
    ) -> None:
        self.temperature_scaler = system.temperature_scaler

    def set_alpha_and_timestep(self, alpha: float, timestep: int) -> None:
        pass

    def minimize_then_run(self, state: SystemState) -> SystemState:
        return state

    def run(self, state: SystemState) -> SystemState:
        return state

    def get_energy(self, state: SystemState) -> float:
        return 0.0

    def prepare_for_timestep(self, alpha: float, timestep: int) -> None:
        pass
