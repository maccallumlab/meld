"""
Module to handle options for a MELD run
"""

from meld.util import strip_unit
from meld.system import temperature
from meld.system import montecarlo
from meld.system import patchers
from openmm import unit as u  # type: ignore

from typing import Optional


class RunOptions:
    """
    Class to store options for MELD run
    """

    def __setattr__(self, name, value):
        # open we only allow setting of these attributes
        # all others will raise an error, which catches
        # typos
        _allowed_attributes = [
            "runner",
            "timesteps",
            "minimize_steps",
            "min_mc",
            "run_mc",
            "use_rest2",
            "rest2_scaler",
            "param_mcmc_steps",
            "mapper_mcmc_steps",
            "pressure",
            "solvation",
        ]
        _allowed_attributes += ["_{}".format(item) for item in _allowed_attributes]
        if name not in _allowed_attributes:
            raise ValueError(f"Attempted to set unknown attribute {name}")
        else:
            object.__setattr__(self, name, value)

    def __init__(self):
        """
        Initialize RunOptions
        """
        self._runner = "openmm"
        self._timesteps = 5000
        self._minimize_steps = 1000
        self._min_mc = None
        self._run_mc = None
        self._use_rest2 = False
        self._rest2_scaler = None
        self._param_mcmc_steps = None
        self._mapper_mcmc_steps = None
        self._solvation = "implicit"

    @property
    def solvation(self) -> str:
        return self._solvation

    @solvation.setter
    def solvation(self, value: str):
        if value not in ["explicit", "implicit"]:
            raise ValueError(f"Invalid solvation type {value}")
        self._solvation = value

    @property
    def pressure(self) -> float:
        """
        Target pressure
        """
        return self._pressure

    @pressure.setter
    def pressure(self, new_value: u.Quantity) -> None:
        pressure = strip_unit(new_value, u.bar)
        if pressure <= 0:
            raise ValueError("pressure must be > 0")
        self._pressure = pressure

    @property
    def use_rest2(self) -> bool:
        """
        User REST2 for temperature scaling
        """
        return self._use_rest2

    @use_rest2.setter
    def use_rest2(self, new_value: bool) -> None:
        self._use_rest2 = new_value

    @property
    def rest2_scaler(self) -> temperature.REST2Scaler:
        """
        Scaler for REST2 temperature scaling
        """
        return self._rest2_scaler

    @rest2_scaler.setter
    def rest2_scaler(self, new_value: temperature.REST2Scaler) -> None:
        self._rest2_scaler = new_value

    @property
    def min_mc(self) -> montecarlo.MonteCarloScheduler:
        """
        MonteCarlo Scheduler used during minimization
        """
        return self._min_mc

    @min_mc.setter
    def min_mc(self, new_value: montecarlo.MonteCarloScheduler) -> None:
        self._min_mc = new_value

    @property
    def run_mc(self) -> montecarlo.MonteCarloScheduler:
        """
        MonteCarloScheduler used during run
        """
        return self._run_mc

    @run_mc.setter
    def run_mc(self, new_value: montecarlo.MonteCarloScheduler) -> None:
        self._run_mc = new_value

    @property
    def runner(self) -> str:
        """
        ReplicaRunner to use during run. One of [openmm, fake_runner]
        """
        return self._runner

    @runner.setter
    def runner(self, value: str) -> None:
        if value not in ["openmm", "fake_runner"]:
            raise RuntimeError(f"unknown value for runner {value}")
        self._runner = value

    @property
    def timesteps(self) -> int:
        """
        Number of timesteps per stage
        """
        return self._timesteps

    @timesteps.setter
    def timesteps(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            raise RuntimeError("timesteps must be > 0")
        self._timesteps = value

    @property
    def minimize_steps(self) -> int:
        """
        Number of minimization steps at start of calculation
        """
        return self._minimize_steps

    @minimize_steps.setter
    def minimize_steps(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            raise RuntimeError("minimize_steps must be > 0")
        self._minimize_steps = value

    @property
    def param_mcmc_steps(self):
        return self._param_mcmc_steps

    @param_mcmc_steps.setter
    def param_mcmc_steps(self, value):
        if value < 0:
            raise RuntimeError("param_mcmc_steps must be positive")
        self._param_mcmc_steps = value

    @property
    def mapper_mcmc_steps(self):
        return self._mapper_mcmc_steps

    @mapper_mcmc_steps.setter
    def mapper_mcmc_steps(self, value):
        if value < 0:
            raise RuntimeError("param_mcmc_steps must be positive")
        self._mapper_mcmc_steps = value