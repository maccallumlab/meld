#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import math
from collections import namedtuple
from typing import List, Optional, Union

import numpy as np  # type: ignore

# named tuple to hold the results
AdaptationRequired = namedtuple("AdaptationRequired", "adapt_now reset_now")


class AdaptationPolicy:
    """
    Defines an adaptation policy

    Repeat adaptation on a regular schedule with an optional burn-in and
    increasing adaptation times.
    """

    def __init__(
        self,
        growth_factor: float,
        burn_in: int,
        adapt_every: int,
        stop_after: Optional[int] = None,
    ) -> None:
        """
        Initialize an AdaptationPolicy

        Args:
            growth_factor: increase adapt_every by a factor of growth_factor
                every adaptation
            burn_in: number of steps to ignore at the beginning
            adapt_every: how frequently to adapt (in picoseconds)
            stop_after: when to stop adapting
        """
        self.growth_factor = growth_factor
        self.burn_in: Optional[int] = burn_in
        self.adapt_every = adapt_every
        self.next_adapt = adapt_every + burn_in
        self.stop_after = stop_after

    def should_adapt(self, step: int) -> AdaptationRequired:
        """
        Determine if adaptation is required

        Args:
            step: the current simulation step
        Returns:
            a :class:`AdaptationPolicy.AdaptationRequired` object
                indicating if adaptation or resetting is necessary

        """
        if self.stop_after is not None:
            if step > self.stop_after:
                return AdaptationRequired(False, False)

        if self.burn_in:
            if step >= self.burn_in:
                self.burn_in = None
                result = AdaptationRequired(False, True)
            else:
                result = AdaptationRequired(False, False)
        else:
            if step >= self.next_adapt:
                result = AdaptationRequired(True, True)
                self.adapt_every = int(self.growth_factor * self.adapt_every)
                self.next_adapt += self.adapt_every
            else:
                result = AdaptationRequired(False, False)
        return result


class _AcceptanceCounter:
    def __init__(self, n_replicas: int) -> None:
        self.n_replicas = n_replicas
        self.successes = np.zeros(self.n_replicas - 1)
        self.attemps = np.zeros(self.n_replicas - 1)
        self.reset()

    def reset(self) -> None:
        """
        Reset statistics
        """
        self.successes = np.zeros(self.n_replicas - 1)
        self.attempts = np.zeros(self.n_replicas - 1)

    def update(self, i: int, accepted: bool) -> None:
        """
        Update statistics

        Args:
            i: lower replica index of pair with attempted swap
            accepted: :const:`True` if swap was successful
        """
        assert i in range(self.n_replicas - 1)
        self.attempts[i] += 1
        if accepted:
            self.successes[i] += 1

    def get_acceptance_probabilities(self) -> np.ndarray:
        """
        Get acceptance probabilities
        """
        return self.successes / (self.attempts + 1e-9)


class NullAdaptor(_AcceptanceCounter):
    """
    Adaptor that does nothing
    """

    def __init__(self, n_replicas: int) -> None:
        """
        Initialize NullAdaptor

        Args:
            n_replicas: number of replicas
        """
        _AcceptanceCounter.__init__(self, n_replicas)
        self.reset()

    def update(self, i: int, accepted: bool) -> None:
        _AcceptanceCounter.update(self, i, accepted)

    def adapt(self, previous_lambdas: List[float], step: int) -> List[float]:
        return previous_lambdas

    def reset(self) -> None:
        _AcceptanceCounter.reset(self)


class EqualAcceptanceAdaptor(_AcceptanceCounter):
    """
    Adaptor based on making acceptance rates uniform
    """

    accept_probs: np.ndarray
    t_lens: np.ndarray

    def __init__(
        self,
        n_replicas: int,
        adaptation_policy: AdaptationPolicy,
        min_acc_prob: float = 0.1,
    ) -> None:
        """
        Initialize EqualAcceptanceAdaptor

        Args:
            n_replicas: number of replicas
            adaptation_policy: policy to use to decide when to adapt
            min_acc_prob: floor on the acceptance probability used for adaptation
        """
        _AcceptanceCounter.__init__(self, n_replicas)

        self.adaptation_policy = adaptation_policy
        self.min_acc_prob = min_acc_prob
        self.reset()

    def update(self, i: int, accepted: bool) -> None:
        """
        Update statistics

        Args:
            i: lower replica index of pair with attempted swap
            accepted: :const:`True` if swap was successful
        """
        _AcceptanceCounter.update(self, i, accepted)

    def adapt(self, previous_lambdas: List[float], step: int) -> List[float]:
        """
        Compute new optimal values of lambda.

        Args:
            previous_lambdas: the previous lambda values
            step: the current simulation step

        Returns:
            the new, optimized lambda values
        """
        should_adapt = self.adaptation_policy.should_adapt(step)

        if should_adapt.adapt_now:
            self._compute_accept_probs()
            self._compute_t_len()

            # put the t_lens on a grid
            lambda_grid = np.linspace(0.0, 1.0, 5000)
            t_lens = np.interp(lambda_grid, previous_lambdas, self.t_lens)

            # compute the desired t_lens based on equal spacing
            even_spacing = np.linspace(0, t_lens[-1], self.n_replicas)

            # compute the values of lambda that will give the desired evenly
            # spaced t_lens
            new_lambdas = list(np.interp(even_spacing[1:-1], t_lens, lambda_grid))
            new_lambdas = [0.0] + new_lambdas + [1.0]
        else:
            new_lambdas = previous_lambdas

        if should_adapt.reset_now:
            self.reset()

        return new_lambdas

    def reset(self) -> None:
        """
        Forget about any previous updates.

        Resets all internal counters and statistics to zero.
        """
        _AcceptanceCounter.reset(self)

    def _compute_accept_probs(self) -> None:
        # default to 50 percent if there hasn't been a trial
        self.successes[self.attempts == 0] = 1.0
        self.attempts[self.attempts == 0] = 2.0

        self.accept_probs = self.successes / self.attempts

        # set minimum percentage
        index = self.accept_probs < self.min_acc_prob
        self.accept_probs[index] = self.min_acc_prob

    def _compute_t_len(self) -> None:
        # compute the t_len between adjacent pairs
        delta_ts = [math.sqrt(-2.0 * math.log(acc)) for acc in self.accept_probs]

        # compute a running total
        t_lens = [0.0]
        total = 0.0
        for dt in delta_ts:
            total += dt
            t_lens.append(total)
        self.t_lens = np.array(t_lens)


Adaptor = Union[NullAdaptor, EqualAcceptanceAdaptor]


class SwitchingCompositeAdaptor:
    """
    An adaptor that switches between two strategies at a specified time
    """

    def __init__(
        self, switching_time: int, first_adaptor: Adaptor, second_adaptor: Adaptor
    ) -> None:
        """
        Initialize a SwitchingCompositeAdaptor

        Args:
            switching_time: when to switch from first_adaptor to second_adaptor
            first_adaptor: adaptor before switching_time
            second_adaptor: adaptor after switching_time
        """
        self.switching_time = switching_time
        self.first_adaptor = first_adaptor
        self.second_adaptor = second_adaptor

    def update(self, i: int, accepted: bool) -> None:
        """
        Update statistics

        Args:
            i: lower replica index of pair with attempted swap
            accepted: :const:`True` if swap was successful
        """
        self.first_adaptor.update(i, accepted)
        self.second_adaptor.update(i, accepted)

    def adapt(self, previous_lambdas: List[float], step: int) -> List[float]:
        """
        Compute new optimal values of lambda.

        Args:
            previous_lambdas: the previous lambda values
            step: the current simulation step

        Returns:
            the new, optimized lambda values
        """
        lambdas_from_first = self.first_adaptor.adapt(previous_lambdas, step)
        lambdas_from_second = self.second_adaptor.adapt(previous_lambdas, step)
        if step <= self.switching_time:
            return lambdas_from_first
        else:
            return lambdas_from_second

    def reset(self) -> None:
        """
        Forget about any previous updates.

        Resets all internal counters and statistics to zero.
        """
        self.first_adaptor.reset()
        self.second_adaptor.reset()

    def get_acceptance_probabilities(self) -> np.ndarray:
        return self.first_adaptor.get_acceptance_probabilities()
