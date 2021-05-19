import random
from collections import OrderedDict
from typing import NamedTuple, Union, TypeVar, Generic, List
from abc import ABCMeta, abstractmethod
import numpy as np  # type: ignore


Number = TypeVar("Number", int, float)

#
# Priors
#
class Prior(Generic[Number], metaclass=ABCMeta):
    @abstractmethod
    def log_prior(self, value: Number) -> float:
        pass


class ContinuousPrior(Prior[float], metaclass=ABCMeta):
    @abstractmethod
    def log_prior(self, value: float) -> float:
        pass


class DiscretePrior(Prior[int], metaclass=ABCMeta):
    @abstractmethod
    def log_prior(self, value: int) -> float:
        pass


class UniformDiscretePrior(DiscretePrior):
    def log_prior(self, value: int) -> float:
        return 0.0


class UniformContinuousPrior(ContinuousPrior):
    def log_prior(self, value: float) -> float:
        return 0.0


class ExponentialDiscretePrior(DiscretePrior):
    k: float

    def __init__(self, k: float):
        self.k = k

    def log_prior(self, value: int) -> float:
        return self.k * value


class ExponentialContinuousPrior(ContinuousPrior):
    k: float

    def __init__(self, k: float):
        self.k = k

    def log_prior(self, value: float) -> float:
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
    def log_prior(self, value: Number) -> float:
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

    def log_prior(self, value: int) -> float:
        return self._prior.log_prior(value)


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

    def log_prior(self, value: float) -> float:
        return self._prior.log_prior(value)


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

    def extract_value(
        self, parameter: Parameter, param_state: ParameterState
    ) -> Number:
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

    def log_prior(self, param_state: ParameterState) -> float:
        total = 0.0
        assert len(self._discrete_by_index) == len(param_state.discrete)
        assert len(self._continuous_by_index) == len(param_state.continuous)

        for i, p_disc in enumerate(self._discrete_by_index):
            v = param_state.discrete[i]
            total += p_disc.log_prior(v)

        for i, p_cont in enumerate(self._continuous_by_index):
            v = param_state.continuous[i]
            total += p_cont.log_prior(v)
        return total

    def sample(self, param_state: ParameterState) -> ParameterState:
        assert len(self._discrete_by_index) == len(param_state.discrete)
        assert len(self._continuous_by_index) == len(param_state.continuous)

        new_discrete = np.zeros_like(param_state.discrete)
        new_continuous = np.zeros_like(param_state.continuous)

        for i, p_disc in enumerate(self._discrete_by_index):
            v = self.extract_value(self._discrete_by_index[i], param_state)
            new_discrete[i] = p_disc.sample(v)

        for i, p_cont in enumerate(self._continuous_by_index):
            v = self.extract_value(self._continuous_by_index[i], param_state)
            new_continuous[i] = p_cont.sample(v)

        return ParameterState(new_discrete, new_continuous)
