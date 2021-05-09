#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld.system.state import SystemState
from meld.system.system import TemperatureScaler, System
from meld.system import RunOptions
from meld.comm import MPICommunicator
from typing import Optional


class ReplicaRunner:
    temperature_scaler: TemperatureScaler

    def initialize(self) -> None:
        pass

    def minimize_then_run(self, state: SystemState) -> SystemState:
        pass

    def run(self, state: SystemState) -> SystemState:
        pass

    def get_energy(self, state: SystemState) -> float:
        pass

    def prepare_for_timestep(self, state, alpha, timestep):
        pass


class FakeSystemRunner(ReplicaRunner):
    """
    Fake runner for test purposes.
    """

    def __init__(
        self,
        system: System,
        options: RunOptions,
        communicator: Optional[MPICommunicator] = None,
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

    def prepare_for_timestep(self, state, alpha: float, timestep: int) -> None:
        pass
