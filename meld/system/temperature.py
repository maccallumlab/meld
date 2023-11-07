"""
Module to handle temperature scaling
"""

import math
from abc import ABC, abstractmethod

from openmm import unit as u  # type: ignore

from meld.util import strip_unit


class TemperatureScaler(ABC):
    """
    Base class for temperature scalers
    """

    @abstractmethod
    def __call__(self, alpha: float) -> float:
        pass


class ConstantTemperatureScaler(TemperatureScaler):
    """
    Constant temperature scaler
    """

    def __init__(self, temperature: u.Quantity):
        """
        Initialize a ConstantTemperatureScaler

        Args:
            temperature: constant temperature to return
        """
        self._temperature = strip_unit(temperature, u.kelvin)

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError(f"0 <= alpha <= 1. alpha={alpha}")
        return self._temperature


class LinearTemperatureScaler(TemperatureScaler):
    """
    Scale temperature linearly between alpha_min and alpha_max
    """

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        temperature_min: u.Quantity,
        temperature_max: u.Quantity,
    ):
        """
        Initialize LinearTemperatureScaler

        Args:
            alpha_min: temperature is ``temperature_min`` when ``alpha <= alpha_min``
            alpha_max: temperature is ``temperature_max`` when ``alpha >= alpha_max``
            temperature_min: minimum temperature in Kelvin
            termperature_max: maximum temperature in Kelvin
        """
        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._temperature_min = strip_unit(temperature_min, u.kelvin)
        self._temperature_max = strip_unit(temperature_max, u.kelvin)

        if self._alpha_min < 0 or self._alpha_min > 1:
            raise RuntimeError("0 <= alpha_min <=1")
        if self._alpha_max < 0 or self._alpha_max > 1:
            raise RuntimeError("0 <= alpha_max <=1")
        if self._alpha_min >= self._alpha_max:
            raise RuntimeError("alpha_min must be < alpha_max")
        if self._temperature_min <= 0 or self._temperature_max <= 0:
            raise RuntimeError("temperatures must be positive")

        self._delta_alpha = self._alpha_max - self._alpha_min
        self._delta_temp = self._temperature_max - self._temperature_min

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError("0 <= alpha <=1 1")
        if alpha <= self._alpha_min:
            return self._temperature_min
        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            return self._temperature_min + frac * self._delta_temp
        else:
            return self._temperature_max


class GeometricTemperatureScaler(TemperatureScaler):
    """
    Scale temperature geometrically
    """

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        temperature_min: u.Quantity,
        temperature_max: u.Quantity,
    ):
        """
        Initialize a GeometricTemperatureScaler

        Args:
            alpha_min: temperature is ``temperature_min`` when ``alpha <= alpha_min``
            alpha_max: temperature is ``temperature_max`` when ``alpha >= alpha_max``
            temperature_min: minimum temperature, in Kelvin
            temperature_max: maximum temperature, in Kelvin
        """
        self._temperature_min = strip_unit(temperature_min, u.kelvin)
        self._temperature_max = strip_unit(temperature_max, u.kelvin)

        if alpha_min < 0 or alpha_min > 1:
            raise RuntimeError("0 <= alpha_min <=1")
        if alpha_max < 0 or alpha_max > 1:
            raise RuntimeError("0 <= alpha_max <=1")
        if alpha_min >= alpha_max:
            raise RuntimeError("alpha_min must be < alpha_max")
        if self._temperature_min <= 0 or self._temperature_max <= 0:
            raise RuntimeError("temperatures must be positive")

        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._delta_alpha = self._alpha_max - self._alpha_min

    def __call__(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError("0 <= alpha <=1 1")
        if alpha <= self._alpha_min:
            return self._temperature_min
        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            delta = math.log(self._temperature_max) - math.log(self._temperature_min)
            return math.exp(delta * frac + math.log(self._temperature_min))
        else:
            return self._temperature_max


class REST2Scaler:
    """
    Scaler for REST2

    Scales protein-protein and protein-water interactions without
    scaling water-water interactions, rather than scaling the
    temperature.

    When performing REST2 simulations, typically the system temperature is kept
    fixed at 300K. Then the psuedo-temperature of non-solvent nonbonded and
    torsion interactions is adjusted based on the ``temperature_scaler`` parameter
    according to:
    :code:`scale = reference_temperature / temperature_scaler(alpha)`
    """

    def __init__(
        self,
        reference_temperature: u.Quantity,
        temperature_scaler: TemperatureScaler,
    ):
        """
        Initialize a REST2Scaler

        Args:
            reference_temperature: this should be set to the temperature of
                the simulation, usually 300K
            temperature_scaler: the psuedo-temperature to adjust nonbonded and
                torsion parameters of REST2
        """
        self.reference_temperature = strip_unit(reference_temperature, u.kelvin)
        self.scaler = temperature_scaler

    def __call__(self, alpha):
        return self.reference_temperature / self.scaler(alpha)
