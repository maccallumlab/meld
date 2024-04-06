#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import random
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Generic, List, NamedTuple, Optional, TypeVar

import numpy as np  # type: ignore
from openmm import unit as u  # type: ignore

from meld.system import scalers, temperature

Number = TypeVar("Number", int, float)


#
# Priors
#
class Prior(Generic[Number], metaclass=ABCMeta):
    @abstractmethod
    def log_prior(self, value: Number, alpha: float) -> float:
        pass


class ContinuousPrior(Prior[float], metaclass=ABCMeta):
    @abstractmethod
    def log_prior(self, value: float, alpha: float) -> float:
        pass


class DiscretePrior(Prior[int], metaclass=ABCMeta):
    @abstractmethod
    def log_prior(self, value: int, alpha: float) -> float:
        pass


class UniformDiscretePrior(DiscretePrior):
    def log_prior(self, value: int, alpha: float) -> float:
        return 0.0


class UniformContinuousPrior(ContinuousPrior):
    def log_prior(self, value: float, alpha: float) -> float:
        return 0.0


class ExponentialDiscretePrior(DiscretePrior):
    k: float

    def __init__(self, k: float):
        self.k = k

    def log_prior(self, value: int, alpha: float) -> float:
        return self.k * value


class ScaledExponentialDiscretePrior(DiscretePrior):
    """
    Exponential prior on a discrete variable, scaled by temperature and force constant.

    Args:
        u0: log_prior in units of kT at T(alpha=0)
        temperature_scaler: determines temperature as a function of alpha
        scaler: scales prior based on alpha

    The log_prior is calculated as:

        log_prior = u0 * scaler(alpha) * temperature_scaler(0.0) / temperature_scaler(alpha)
    """

    temperature_scaler: temperature.TemperatureScaler
    scaler: scalers.RestraintScaler
    u0: float

    def __init__(
        self,
        u0: float,
        temperature_scaler: Optional[temperature.TemperatureScaler],
        scaler: Optional[scalers.RestraintScaler],
    ):
        self.u0 = u0
        if temperature_scaler is None:
            self.temperature_scaler = temperature.ConstantTemperatureScaler(
                298 * u.kelvin
            )
        else:
            self.temperature_scaler = temperature_scaler

        if scaler is None:
            self.scaler = scalers.ConstantScaler()
        else:
            self.scaler = scaler

    def log_prior(self, value: int, alpha: float) -> float:
        T0 = self.temperature_scaler(0.0)
        T = self.temperature_scaler(alpha)
        return self.u0 * self.scaler(alpha) * T0 / T * value


class ExponentialContinuousPrior(ContinuousPrior):
    k: float

    def __init__(self, k: float):
        self.k = k

    def log_prior(self, value: float, alpha: float) -> float:
        return self.k * value


#
# Samplers
#
class Sampler(Generic[Number], metaclass=ABCMeta):
    @abstractmethod
    def is_valid(self, value: Number) -> bool:
        pass

    @abstractmethod
    def sample(self, value: Number) -> Number:
        pass


class DiscreteSampler(Sampler[int]):
    def __init__(self, min: int, max: int, step_size: int):
        self.min = min
        self.max = max
        assert step_size > 0
        self.step_size = step_size

    def is_valid(self, value: int) -> bool:
        return value >= self.min and value <= self.max

    def sample(self, value: int) -> int:
        return random.randint(value - self.step_size, value + self.step_size)


class ContinuousSampler(Sampler[float]):
    min: float
    max: float
    std: float

    def __init__(self, min: float, max: float, std: float):
        self.min = min
        self.max = max
        assert std > 0
        self.std = std

    def is_valid(self, value: float) -> bool:
        return value >= self.min and value <= self.max

    def sample(self, value: float) -> float:
        return random.gauss(value, self.std)


#
# Parameters
#
class Parameter(Generic[Number], metaclass=ABCMeta):
    name: str
    index: int
    sampler: Sampler[Number]
    prior: Prior[Number]

    @abstractmethod
    def is_valid(self, value: Number) -> bool:
        pass

    @abstractmethod
    def sample(self, value: Number) -> Number:
        pass

    @abstractmethod
    def log_prior(self, value: Number, alpha: float) -> float:
        pass


class DiscreteParameter(Parameter[int]):
    def __init__(
        self, name: str, index: int, sampler: DiscreteSampler, prior: DiscretePrior
    ):
        self.name = name
        self.index = index
        self._sampler = sampler
        self._prior = prior

    @property
    def min(self):
        return self._sampler.min

    @property
    def max(self):
        return self._sampler.max

    def is_valid(self, value: int) -> bool:
        return self._sampler.is_valid(value)

    def sample(self, value: int) -> int:
        return self._sampler.sample(value)

    def log_prior(self, value: int, alpha: float) -> float:
        return self._prior.log_prior(value, alpha)


class ContinuousParameter(Parameter[float]):
    def __init__(
        self, name: str, index: int, sampler: ContinuousSampler, prior: ContinuousPrior
    ):
        self.name = name
        self.index = index
        self._sampler = sampler
        self._prior = prior

    @property
    def min(self):
        return self._sampler.min

    @property
    def max(self):
        return self._sampler.max

    def is_valid(self, value: float) -> bool:
        return self._sampler.is_valid(value)

    def sample(self, value: float) -> float:
        return self._sampler.sample(value)

    def log_prior(self, value: float, alpha: float) -> float:
        return self._prior.log_prior(value, alpha)


#
# State
#
class ParameterState(NamedTuple):
    discrete: np.ndarray
    continuous: np.ndarray


#
# Parameter Manager
#
class ParameterManager:
    def __init__(self):
        self._init_values_discrete: List[int] = []
        self._init_values_continuous: List[float] = []
        self.parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._discrete_by_index: List[DiscreteParameter] = []
        self._continuous_by_index: List[ContinuousParameter] = []

    def has_parameters(self):
        return len(self.parameters)

    def add_discrete_parameter(
        self, name: str, init_value: int, prior: DiscretePrior, sampler: DiscreteSampler
    ) -> "Parameter":
        assert sampler.is_valid(init_value)
        assert name not in self.parameters

        index = len(self._init_values_discrete)
        param = DiscreteParameter(name, index, sampler, prior)

        self._init_values_discrete.append(init_value)
        self._discrete_by_index.append(param)
        self.parameters[name] = param
        return param

    def add_continuous_parameter(
        self,
        name: str,
        init_value: float,
        prior: ContinuousPrior,
        sampler: ContinuousSampler,
    ) -> "Parameter":
        assert sampler.is_valid(init_value)
        assert name not in self.parameters

        index = len(self._init_values_continuous)
        param = ContinuousParameter(name, index, sampler, prior)

        self._init_values_continuous.append(init_value)
        self._continuous_by_index.append(param)
        self.parameters[name] = param
        return param

    def get_initial_state(self) -> ParameterState:
        return ParameterState(
            np.array(self._init_values_discrete, dtype=np.int32),
            np.array(self._init_values_continuous, dtype=np.float64),
        )

    def extract_value(self, parameter: Parameter, param_state: ParameterState):
        param = self.parameters[parameter.name]

        if isinstance(param, DiscreteParameter):
            return param_state.discrete[param.index]
        else:
            return param_state.continuous[param.index]

    def is_valid(self, param_state: ParameterState) -> bool:
        valid = True
        assert len(self._discrete_by_index) == len(param_state.discrete)
        assert len(self._continuous_by_index) == len(param_state.continuous)

        for i, p_disc in enumerate(self._discrete_by_index):
            v = param_state.discrete[i]
            valid = valid and p_disc.is_valid(v)

        for i, p_cont in enumerate(self._continuous_by_index):
            v = param_state.continuous[i]
            valid = valid and p_cont.is_valid(v)
        return valid

    def log_prior(self, param_state: ParameterState, alpha: float) -> float:
        total = 0.0
        assert len(self._discrete_by_index) == len(param_state.discrete)
        assert len(self._continuous_by_index) == len(param_state.continuous)

        for i, p_disc in enumerate(self._discrete_by_index):
            v = param_state.discrete[i]
            total += p_disc.log_prior(v, alpha)

        for i, p_cont in enumerate(self._continuous_by_index):
            v = param_state.continuous[i]
            total += p_cont.log_prior(v, alpha)
        return total

    def sample(self, param_state: ParameterState) -> ParameterState:
        n_discrete = len(param_state.discrete)
        n_continuous = len(param_state.continuous)
        assert len(self._discrete_by_index) == n_discrete
        assert len(self._continuous_by_index) == n_continuous

        if random.random() < n_discrete / (n_discrete + n_continuous):
            sample_discrete = True
        else:
            sample_discrete = False

        if sample_discrete:
            sample_index = random.randrange(n_discrete)
        else:
            sample_index = random.randrange(n_continuous)

        new_discrete = np.zeros_like(param_state.discrete)
        new_continuous = np.zeros_like(param_state.continuous)

        for i, p_disc in enumerate(self._discrete_by_index):
            v = self.extract_value(self._discrete_by_index[i], param_state)
            if not sample_discrete or i != sample_index:
                new_discrete[i] = v
            else:
                new_discrete[i] = p_disc.sample(v)

        for i, p_cont in enumerate(self._continuous_by_index):
            v = self.extract_value(self._continuous_by_index[i], param_state)
            if sample_discrete or i != sample_index:
                new_continuous[i] = v
            else:
                new_continuous[i] = p_cont.sample(v)

        return ParameterState(new_discrete, new_continuous)
