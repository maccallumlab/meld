import math
from meld.util import strip_unit
from typing import Dict, Any, Optional, Union, List, NamedTuple
from openmm import unit as u  # type: ignore

STRENGTH_AT_ALPHA_MAX = 1e-3  # default strength of restraints at alpha=1.0


class ScalerRegistry(type):
    """
    Metaclass that maintains a registry of scaler types.

    All classes that decend from Scaler inherit ScalerRegistry as their
    metaclass. ScalerRegistry will automatically maintain a map between
    the class attribute '_scaler_key_' and all scaler types.

    The function get_constructor_for_key is used to get the class for the
    corresponding key.
    """

    _scaler_registry: Dict[str, type] = {}

    def __init__(cls, name, bases, attrs):
        if name in [
            "AlphaMapper",
            "RestraintScaler",
            "BlurScaler",
            "TimeRamp",
            "Positioner",
        ]:
            pass  # we don't register the base classes
        else:
            try:
                key = attrs["_scaler_key_"]
            except KeyError:
                raise RuntimeError(
                    f"Scaler type {name} subclasses Scaler, but"
                    "does not set _scaler_key_"
                )
            if key in ScalerRegistry._scaler_registry:
                raise RuntimeError(
                    "Trying to register two different classes"
                    f"with _scaler_key_ = {key}."
                )
            ScalerRegistry._scaler_registry[key] = cls

    @classmethod
    def get_constructor_for_key(self, key):
        """Get the constructor for the scaler type matching key."""
        try:
            return ScalerRegistry._scaler_registry[key]
        except KeyError:
            raise RuntimeError(f'Unknown scaler type "{key}".')


class AlphaMapper(metaclass=ScalerRegistry):
    """Base class for all scalers."""

    def __init__(self):
        self._alpha_min = 0.0
        self._alpha_max = 1.0

    def _check_alpha_range(self, alpha):
        if alpha < 0 or alpha > 1:
            raise RuntimeError(f"0 >= alpha >= 1. alpha is {alpha}.")

    def _handle_boundaries(self, alpha):
        if alpha <= self._alpha_min:
            return 1.0
        elif alpha >= self._alpha_max:
            return 0.0
        else:
            return None

    def _check_alpha_min_max(self):
        if (
            self._alpha_min < 0
            or self._alpha_min > 1
            or self._alpha_max < 0
            or self._alpha_max > 1
        ):
            raise RuntimeError(
                "alpha_min and alpha_max must be in range [0, 1]."
                f"alpha_min={self._alpha_min} alpha_max={self._alpha_max}."
            )
        if self._alpha_min >= self._alpha_max:
            raise RuntimeError(
                "alpha_max must be less than alpha_min."
                f"alpha_min={self._alpha_min} alpha_max={self._alpha_max}."
            )


class RestraintScaler(AlphaMapper):
    """Base class for all resraint scaler classes."""

    def __call__(self, alpha: float) -> float:
        raise NotImplementedError("Cannot call base RestraintScaler")


class ConstantScaler(RestraintScaler):
    """This scaler is "always on" and always returns a value of 1.0"."""

    _scaler_key_ = "constant"

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)
        return 1.0


class LinearScaler(RestraintScaler):
    """
    This scaler linearly interpolates from alpha_min to alpha_max.
    """

    _scaler_key_ = "linear"

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a LinearScaler

        Args:
            alpha_min: minimum alpha value
            alpha_max: maximum alpha value
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._delta = alpha_max - alpha_min
        self._check_alpha_min_max()

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)
        scale = self._handle_boundaries(alpha)
        if scale is None:
            scale = 1.0 - (alpha - self._alpha_min) / self._delta
        scale = (1.0 - scale) * (
            self._strength_at_alpha_max - self._strength_at_alpha_min
        ) + self._strength_at_alpha_min
        return scale


class PlateauLinearScaler(RestraintScaler):
    r"""
    A scaler with a plateau shape

    This scaler linearly interpolates between 0 and 1 from alpha_min to
    alpha_one, keeps the value of 1 until alpha_two and then decreases
    linearly until 0 in alpha_max.

    ::

        |    ------   strength alpha_min --> between two and one
        |  /        \
        | /          \ strength alpha_max --> > alpha_max and
        |                                       below alphamin

    """

    _scaler_key_ = "plateau"

    def __init__(
        self,
        alpha_min: float,
        alpha_one: float,
        alpha_two: float,
        alpha_max: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a PlateauLinearScaler

        Args:
            alpha_min: minimum alpha value
            alpha_one: lower range of plateau
            alpha_two: upper range of plateau
            alpha_max: maximum alpha value
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = float(alpha_min)
        self._alpha_one = float(alpha_one)
        self._alpha_two = float(alpha_two)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)
        if alpha <= self._alpha_min:
            scale = self._strength_at_alpha_max
        else:
            if alpha <= self._alpha_one:
                # Decreasing
                scale = 1.0 - (self._alpha_one - alpha) / (
                    self._alpha_one - self._alpha_min
                )
                scale = (
                    scale * (self._strength_at_alpha_min - self._strength_at_alpha_max)
                    + self._strength_at_alpha_max
                )

            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                # Increasing
                scale = 1.0 - (alpha - self._alpha_two) / (
                    self._alpha_max - self._alpha_two
                )
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_max - self._strength_at_alpha_min
                ) + self._strength_at_alpha_min
            else:
                scale = self._strength_at_alpha_max
        return scale


class NonLinearScaler(RestraintScaler):
    """
    A restraint scaler with a non-linear shape
    """

    _scaler_key_ = "nonlinear"

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        factor: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Intialize a NonLinearScaler

        Args:
            alpha_min: minimum alpha value
            alpha_max: maximum alpha value
            factor: controls the non-linear shape, must be >= 0
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()
        if factor < 1:
            raise RuntimeError(f"factor must be >= 1. factor={factor}.")
        self._factor = factor

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)
        scale = self._handle_boundaries(alpha)
        if scale is None:
            delta = (alpha - self._alpha_min) / (self._alpha_max - self._alpha_min)
            norm = 1.0 / (math.exp(self._factor) - 1.0)
            scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
        scale = (1.0 - scale) * (
            self._strength_at_alpha_max - self._strength_at_alpha_min
        ) + self._strength_at_alpha_min
        return scale


class PlateauNonLinearScaler(RestraintScaler):
    """
    Nonlinear scaler with a plateau shape

    This scaler linearly interpolates between 0 and 1 from alpha_min
    to alpha_one, keeps the value of 1 until alpha_two and then decreases
    linearly until 0 in alpha_max.
    """

    _scaler_key_ = "plateaunonlinear"

    def __init__(
        self,
        alpha_min: float,
        alpha_one: float,
        alpha_two: float,
        alpha_max: float,
        factor: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a PlateauNonlinearScaler

        Args:
            alpha_min: minimum alpha value
            alpha_one: lower range of plateau
            alpha_two: upper range of plateau
            alpha_max: maximum alpha value
            factor: controls the non-linear shape, must be >= 0
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = float(alpha_min)
        self._alpha_one = float(alpha_one)
        self._alpha_two = float(alpha_two)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()
        if factor < 1:
            raise RuntimeError(f"factor must be >= 1. factor={factor}.")
        self._factor = factor

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)
        if alpha <= self._alpha_min:
            scale = self._strength_at_alpha_max
        else:
            if alpha <= self._alpha_one:
                delta = (alpha - self._alpha_min) / (self._alpha_one - self._alpha_min)
                norm = 1.0 / (math.exp(self._factor) - 1.0)
                scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_min - self._strength_at_alpha_max
                ) + self._strength_at_alpha_max
            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                delta = (alpha - self._alpha_two) / (self._alpha_max - self._alpha_two)
                norm = 1.0 / (math.exp(self._factor) - 1.0)
                scale = norm * (math.exp(self._factor * (1.0 - delta)) - 1.0)
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_max - self._strength_at_alpha_min
                ) + self._strength_at_alpha_min
            else:
                scale = self._strength_at_alpha_max

        return scale


class PlateauSmoothScaler(RestraintScaler):
    """
    A scaler with a plateau shape

    This scaler linearly interpolates between 0 and 1 from alpha_min
    to alpha_one, keeps the value of 1 until alpha_two and then decreases
    linearly until 0 in alpha_max.
    """

    _scaler_key_ = "plateausmooth"

    def __init__(
        self,
        alpha_min: float,
        alpha_one: float,
        alpha_two: float,
        alpha_max: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a PlateauSmoothScaler

        Args:
            alpha_min: minimum alpha value
            alpha_one: lower range of plateau
            alpha_two: upper range of plateau
            alpha_max: maximum alpha value
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = float(alpha_min)
        self._alpha_one = float(alpha_one)
        self._alpha_two = float(alpha_two)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = strength_at_alpha_min
        self._strength_at_alpha_max = strength_at_alpha_max
        self._check_alpha_min_max()

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)
        if alpha <= self._alpha_min:
            scale = self._strength_at_alpha_max
        else:
            if alpha <= self._alpha_one:
                delta = (alpha - self._alpha_min) / (self._alpha_one - self._alpha_min)
                scale = delta * delta * (3 - 2 * delta)
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_max - self._strength_at_alpha_min
                ) + self._strength_at_alpha_min
            elif alpha <= self._alpha_two:
                scale = self._strength_at_alpha_min
            elif alpha <= self._alpha_max:
                delta = (alpha - self._alpha_two) / (self._alpha_max - self._alpha_two)
                scale = 1 + delta * delta * (2 * delta - 3)
                scale = (1.0 - scale) * (
                    self._strength_at_alpha_max - self._strength_at_alpha_min
                ) + self._strength_at_alpha_min
            else:
                scale = self._strength_at_alpha_max
        return scale


class GeometricScaler(RestraintScaler):
    """
    Scale restraints geometrically
    """

    _scaler_key_ = "geometric"

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        strength_at_alpha_min: float = 1.0,
        strength_at_alpha_max: float = STRENGTH_AT_ALPHA_MAX,
    ):
        """
        Initialize a GeometricScaler

        Args:
            alpha_min: minimum alpha value
            alpha_max: maximum alpha value
            strength_at_alpha_min: strength when alpha <= alpha_min
            strength_at_alpha_max: strength when alpha >= alpha_max
        """
        self._alpha_min = float(alpha_min)
        self._alpha_max = float(alpha_max)
        self._strength_at_alpha_min = float(strength_at_alpha_min)
        self._strength_at_alpha_max = float(strength_at_alpha_max)
        self._delta_alpha = self._alpha_max - self._alpha_min
        self._check_alpha_min_max()

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)

        if alpha < 0 or alpha > 1:
            raise RuntimeError("0 <= alpha <=1 1")

        elif alpha <= self._alpha_min:
            return self._strength_at_alpha_min

        elif alpha <= self._alpha_max:
            frac = (alpha - self._alpha_min) / self._delta_alpha
            delta = math.log(self._strength_at_alpha_max) - math.log(
                self._strength_at_alpha_min
            )
            return math.exp(delta * frac + math.log(self._strength_at_alpha_min))

        else:
            return self._strength_at_alpha_max


class TimeRamp(AlphaMapper):
    """Base class for all time ramp classes."""

    def __call__(self, timestep: int) -> float:
        raise NotImplementedError("Cannot call base TimeRamp directly")


class ConstantRamp(TimeRamp):
    """TimeRamp that always returns 1.0"""

    _scaler_key_ = "constant_ramp"

    def __call__(self, timestep: int) -> float:
        if timestep < 0:
            raise ValueError("Timestep is < 0.")
        return 1.0


class LinearRamp(TimeRamp):
    """TimeRamp that interpolates linearly"""

    _scaler_key_ = "linear_ramp"

    def __init__(
        self, start_time: float, end_time: float, start_weight: float, end_weight: float
    ):
        """
        Initialize a LinearTimeRamp

        Args:
            start_time: time to start ramping up
            end_time: time to finish ramping up
            start_weight: weight when time <= start_time
            end_weight: weight when time >= end_time
        """
        self.t_start = float(start_time)
        self.t_end = float(end_time)
        self.w_start = float(start_weight)
        self.w_end = float(end_weight)

    def __call__(self, timestep: int) -> float:
        if timestep < 0:
            raise ValueError("Timestep is < 0.")
        if timestep < self.t_start:
            return self.w_start
        elif timestep < self.t_end:
            return self.w_start + (self.w_end - self.w_start) * (
                float(timestep) - self.t_start
            ) / (self.t_end - self.t_start)
        else:
            return self.w_end


class NonLinearRamp(TimeRamp):
    """
    TimeRamp that interpolates non-linearly
    """

    _scaler_key_ = "nonlinear_ramp"

    def __init__(
        self,
        start_time: float,
        end_time: float,
        start_weight: float,
        end_weight: float,
        factor: float,
    ):
        """
        Initialize a NonLinearTimeRamp

        Args:
            start_time: time to start ramping up
            end_time: time to finish ramping up
            start_weight: weight when time <= start_time
            end_weight: weight when time >= end_time
            factor: controls the shape of the non-linear ramp, must be >= 1
        """
        if end_time <= start_time:
            raise ValueError("end_time must be > start_time")
        if factor < 1.0:
            raise ValueError("factor myst be > 1.0")

        self.t_start = float(start_time)
        self.t_end = float(end_time)
        self.w_start = float(start_weight)
        self.w_end = float(end_weight)
        self.factor = float(factor)

    def __call__(self, timestep: int) -> float:
        if timestep < 0:
            raise ValueError("timestep is < 0.")

        if timestep < self.t_start:
            return self.w_start
        elif timestep < self.t_end:
            # we scale differently depending on if we are ramping up or down
            # we change more slowly at lower values and more rapidly at
            # higher values
            #
            # this is for scaling up
            if self.w_end > self.w_start:
                delta = 1.0 - (float(timestep) - self.t_start) / (
                    self.t_end - self.t_start
                )
                norm = 1.0 / (math.exp(self.factor) - 1.0)
                scale = norm * (math.exp(self.factor * (1.0 - delta)) - 1.0)
                return scale * (self.w_end - self.w_start) + self.w_start
            # this is for scaling down
            else:
                delta = (float(timestep) - self.t_start) / (self.t_end - self.t_start)
                norm = 1.0 / (math.exp(self.factor) - 1.0)
                scale = norm * (math.exp(self.factor * (1.0 - delta)) - 1.0)
                return (1.0 - scale) * (self.w_end - self.w_start) + self.w_start
        else:
            return self.w_end


class TimeRampSwitcher(TimeRamp):
    """
    Switches between two TimeRamp objects.

    Class first_ramp before switching time. At the switching
    time it switches to second_ramp, which it uses thereafter.
    """

    _scaler_key_ = "ramp_switcher"

    def __init__(
        self, first_ramp: TimeRamp, second_ramp: TimeRamp, switching_time: float
    ):
        """
        Initialize a TimeRampSwitcher

        Args:
            first_ramp: active when time < switching_time
            second_ramp: active when time >= switching_time
            switching_time: when to switch
        """
        self.first_ramp = first_ramp
        self.second_ramp = second_ramp
        self.switching_time = switching_time

    def __call__(self, timestep: int) -> float:
        if timestep < self.switching_time:
            return self.first_ramp(timestep)
        else:
            return self.second_ramp(timestep)


class Positioner(AlphaMapper):
    """Base class for all positioner classes."""

    def __call__(self, alpha: float) -> float:
        raise NotImplementedError("Cannot call base positioner")


class ConstantPositioner(Positioner):
    """Always returns the supplied value."""

    _scaler_key_ = "constant_positioner"

    def __init__(self, value: u.Quantity) -> None:
        self._value = strip_unit(value, u.nanometer)

    def __call__(self, alpha: float) -> float:
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if alpha > 1:
            raise ValueError("alpha must be <= 1")

        return self._value


class LinearPositioner(Positioner):
    """
    Position restraints linearly within a range
    """

    _scaler_key_ = "linear_positioner"

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        pos_min: u.Quantity,
        pos_max: u.Quantity,
    ) -> None:
        """
        Initialize a LinearPositioner

        Args:
            alpha_min: minimum alpha value
            alpha_max: maximum alpha value
            pos_min: value at alpha_min
            pos_max: value at alpha_max
        """
        if alpha_max <= alpha_min:
            raise ValueError("alpha_max must be > alpha_min")

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.pos_min = strip_unit(pos_min, u.nanometer)
        self.pos_max = strip_unit(pos_max, u.nanometer)

    def __call__(self, alpha: float) -> float:
        if alpha < 0:
            raise ValueError("alpha was < 0")
        if alpha > 1:
            raise ValueError("alpha was > 1")
        if alpha < self.alpha_min:
            return self.pos_min
        elif alpha < self.alpha_max:
            delta = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)
            return delta * (self.pos_max - self.pos_min) + self.pos_min
        else:
            return self.pos_max


class BlurScaler(AlphaMapper):
    """Base class for all blur scaler classes."""

    def __call__(self, alpha: float) -> float:
        raise NotImplementedError("Cannot call base BlurScaler")


class LinearBlurScaler(BlurScaler):
    """
    This scaler linearly interpolates from alpha_min to alpha_max.
    """

    _scaler_key_ = "linear_blur"

    def __init__(
        self,
        alpha_min: float,
        alpha_max: float,
        min_blur: float,
        max_blur: float,
        num_replicas: int,
    ):
        super().__init__()
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._min_blur = min_blur
        self._max_blur = max_blur
        self._delta = alpha_max - alpha_min
        self._num_replicas = num_replicas
        self._check_alpha_min_max()

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)
        blur = self._handle_boundaries(alpha)
        if blur is None:
            blur = self._min_blur + (self._max_blur - self._min_blur) * (alpha - self._alpha_min) / self._delta
        else:
            blur = (1.0 - blur) * (self._max_blur - self._min_blur) + self._min_blur
        return blur

class ConstantBlurScaler(BlurScaler):
    """This scaler is "always on" and always returns a value of 1.0"."""

    _scaler_key_ = "constant_blur"
    def __init__(self, blur: float, num_replicas: int) -> None:
        self.blur = blur
        self._num_replicas = num_replicas

    def __call__(self, alpha: float) -> float:
        self._check_alpha_range(alpha)
        return self.blur
