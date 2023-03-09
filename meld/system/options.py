"""
Module to handle options for a MELD run
"""

from meld.util import strip_unit
from meld.system import temperature
from meld.system import montecarlo
from openmm import unit as u  # type: ignore

from dataclasses import dataclass
from functools import partial
from typing import Optional


@partial(dataclass, frozen=True)
class RunOptions:
    runner: str = "openmm"
    timesteps: int = 5000
    minimize_steps: int = 1000
    min_mc: Optional[montecarlo.MonteCarloScheduler] = None
    run_mc: Optional[montecarlo.MonteCarloScheduler] = None
    use_rest2: bool = False
    rest2_scaler: Optional[temperature.REST2Scaler] = None
    param_mcmc_steps: int = 0
    mapper_mcmc_steps: int = 0
    pressure: float = 1.0 * u.bar

    def __post_init__(self):
        if self.runner not in ["openmm", "fake_runner"]:
            raise ValueError(f"Unknown runner: {self.runner}")

        if isinstance(self.pressure, u.Quantity):
            object.__setattr__(self, "pressure", self.pressure.value_in_unit(u.bar))
        if self.pressure < 0:
            raise ValueError("Pressure must be positive")

        if self.timesteps <= 0:
            raise ValueError("Timesteps cannot be negative")

        if self.minimize_steps < 0:
            raise ValueError("Minimize steps must be positive")

        if self.param_mcmc_steps < 0:
            raise ValueError("param_mcmc_steps cannot be negative")

        if self.mapper_mcmc_steps < 0:
            raise ValueError("mapper_mcmc_steps cannot be negative")
