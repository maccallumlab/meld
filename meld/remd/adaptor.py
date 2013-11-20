import numpy as np
from scipy import interpolate
import math
from collections import namedtuple


class AcceptanceCounter(object):
    '''
    Class to keep track of acceptance rates.
    '''
    def __init__(self, n_replicas):
        self.n_replicas = n_replicas
        self.successes = None
        self.attempts = None
        self.reset()

    def reset(self):
        self.successes = np.zeros(self.n_replicas - 1)
        self.attempts = np.zeros(self.n_replicas - 1)

    def update(self, i, accepted):
        assert i in range(self.n_replicas - 1)
        self.attempts[i] += 1
        if accepted:
            self.successes[i] += 1

    def get_acceptance_probabilities(self):
        return self.successes / (self.attempts + 1e-9)


class NullAdaptor(AcceptanceCounter):
    def __init__(self, n_replicas):
        AcceptanceCounter.__init__(self, n_replicas)
        self.reset()

    def update(self, i, accepted):
        AcceptanceCounter.update(self, i, accepted)

    def adapt(self, previous_lambdas, step):
        return previous_lambdas

    def reset(self):
        AcceptanceCounter.reset(self)


class EqualAcceptanceAdaptor(AcceptanceCounter):
    '''
    Adaptor based on making acceptance rates uniform.
    '''
    def __init__(self, n_replicas, adaptation_policy, min_acc_prob=0.1):
        '''
        Initialize adaptor

        Parameters
            n_replicas -- number of replicas
            min_acc_prob -- all acceptence probabilities below this value will be raised to this value

        '''
        AcceptanceCounter.__init__(self, n_replicas)

        self.adaptation_policy = adaptation_policy
        self.min_acc_prob = min_acc_prob
        self.accept_probs = None
        self.t_lens = None
        self.reset()

    def update(self, i, accepted):
        '''
        Update adaptor with exchange.

        Parameters
            i -- index of first replica; second replica is i+1
            accepted -- True if the exchange was accepted

        '''
        AcceptanceCounter.update(self, i, accepted)

    def adapt(self, previous_lambdas, step):
        '''
        Compute new optimal values of lambda.

        Parameters
            previous_lambdas -- a list of the previous lambda values
            step -- the current simulation step
        Returns
            a list of the new, optimized lambda values

        '''
        should_adapt = self.adaptation_policy.should_adapt(step)

        if should_adapt.adapt_now:
            self._compute_accept_probs()
            self._compute_t_len()

            # put the t_lens on a grid
            lambda_grid = np.linspace(0., 1., 5000)
            t_lens = np.interp(lambda_grid, previous_lambdas, self.t_lens)

            # compute the desired t_lens based on equal spacing
            even_spacing = np.linspace(0, t_lens[-1], self.n_replicas)

            # compute the values of lambda that will give the desired evenly spaced
            # t_lens
            new_lambdas = np.interp(even_spacing[1:-1], t_lens, lambda_grid)
            new_lambdas = [x for x in new_lambdas]
            new_lambdas = [0.] + new_lambdas + [1.]
        else:
            new_lambdas = previous_lambdas

        if should_adapt.reset_now:
            self.reset()

        return new_lambdas

    def reset(self):
        '''
        Forget about any previous updates.

        Resets all internal counters and statistics to zero.

        '''
        AcceptanceCounter.reset(self)
        self.accept_probs = None
        self.t_lens = None

    def _compute_accept_probs(self):
        # default to 50 percent if there hasn't been a trial
        self.successes[self.attempts == 0] = 1.
        self.attempts[self.attempts == 0] = 2.

        self.accept_probs = self.successes / self.attempts

        # set minimum percentage
        self.accept_probs[self.accept_probs < self.min_acc_prob] = self.min_acc_prob

    def _compute_t_len(self):
        # compute the t_len between adjacent pairs
        delta_ts = [math.sqrt(-2.0 * math.log(acc)) for acc in self.accept_probs]

        # compute a running total
        t_lens = [0.]
        total = 0.
        for dt in delta_ts:
            total += dt
            t_lens.append(total)
        self.t_lens = t_lens


class FluxAdaptor(AcceptanceCounter):
    def __init__(self, n_replicas, adaptation_policy, smooth_factor=0.5):
        AcceptanceCounter.__init__(self, n_replicas)
        self.adaptation_policy = adaptation_policy
        assert smooth_factor > 0, 'A small, positive smoothing factor is required.'
        self.smooth_factor = smooth_factor

        self.up_state = None
        self.down_state = None
        self.n_up = None
        self.n_down = None

        self.reset()

    def update(self, i, accepted):
        # update the acceptance probabilities
        AcceptanceCounter.update(self, i, accepted)

        if accepted:
            # swap the states
            self.up_state[[i, i + 1]] = self.up_state[[i + 1, i]]
            self.down_state[[i, i + 1]] = self.down_state[[i + 1, i]]

            # set the values at the end
            self.up_state[0] = 1
            self.down_state[0] = 0
            self.up_state[-1] = 0
            self.down_state[-1] = 1

            # increment the counts
            self.n_up += self.up_state
            self.n_down += self.down_state

    def adapt(self, previous_lambdas, step):
        should_adapt = self.adaptation_policy.should_adapt(step)

        if should_adapt.adapt_now:
            f = self._compute_f()
            f = self._make_f_monotonic(f)
            f = self._apply_smoothing(f)
            gx, gy = self._compute_g(f, previous_lambdas)
            new_lambdas = self._compute_new_lambdas(gx, gy).tolist()
        else:
            new_lambdas = previous_lambdas

        if should_adapt.reset_now:
            self.reset()

        return new_lambdas

    def reset(self):
        AcceptanceCounter.reset(self)
        self.up_state = np.zeros(self.n_replicas)
        self.up_state[0] = 1
        self.down_state = np.zeros(self.n_replicas)
        self.down_state[-1] = 1
        self.n_up = np.zeros(self.n_replicas)
        self.n_down = np.zeros(self.n_replicas)

    def _compute_f(self):
        total = self.n_up + self.n_down
        total[total == 0] = 1  # prevent division by zero if we have no samples
        return self.n_down / total

    def _make_f_monotonic(self, f):
        # compute the derivative and force it to be positive
        diff = np.zeros_like(f)
        diff[1:] = f[1:] - f[:-1]
        diff[diff < 0] = 0

        # integrate and rescale into the correct range
        f = np.cumsum(diff)
        f = f / f[-1]
        return f

    def _apply_smoothing(self, f):
        smooth = np.linspace(0, 1, self.n_replicas)
        return (1.0 - self.smooth_factor) * f + self.smooth_factor * smooth

    def _compute_g(self, f, previous_lambdas):
        gy = np.array(previous_lambdas)
        gx = f
        return gx, gy

    def _compute_new_lambdas(self, gx, gy):
        f = interpolate.interp1d(gx, gy)
        samples = np.linspace(0, 1, self.n_replicas)
        return f(samples)


class SwitchingCompositeAdaptor(object):
    def __init__(self, switching_time, first_adaptor, second_adaptor):
        self.switching_time = switching_time
        self.first_adaptor = first_adaptor
        self.second_adaptor = second_adaptor

    def update(self, i, accepted):
        self.first_adaptor.update(i, accepted)
        self.second_adaptor.update(i, accepted)

    def adapt(self, previous_lambdas, step):
        lambdas_from_first = self.first_adaptor.adapt(previous_lambdas, step)
        lambdas_from_second = self.second_adaptor.adapt(previous_lambdas, step)
        if step <= self.switching_time:
            return lambdas_from_first
        else:
            return lambdas_from_second

    def reset(self):
        self.first_adaptor.reset()
        self.second_adaptor.reset()

    def get_acceptance_probabilities(self):
        return self.first_adaptor.get_acceptance_probabilities()


class AdaptationPolicy(object):
    '''
    Repeat adaptation on a regular schedule with an optional burn-in and increasing adaptation times.
    '''
    # named tuple to hold the results
    AdaptationRequired = namedtuple('AdaptationRequired', 'adapt_now reset_now')

    def __init__(self, growth_factor, burn_in, adapt_every):
        '''
        Initialize a repeating adaptation scheduler

        Parameters:
            growth_factor -- increase adapt_every by a factor of growth_factor every adaptation
            burn_in -- number of steps to ignore at the beginning
            adapt_every -- how frequently to adapt (in picoseconds)

        '''
        self.growth_factor = growth_factor
        self.burn_in = burn_in
        self.adapt_every = adapt_every
        self.next_adapt = adapt_every + burn_in

    def should_adapt(self, step):
        '''
        Is adaptation required?

        Parameters:
            step -- the current simulation step
        Returns:
            an AdaptationRequired object indicating if adaptation or resetting is necessary

        '''
        if self.burn_in:
            if step >= self.burn_in:
                self.burn_in = None
                result = self.AdaptationRequired(False, True)
            else:
                result = self.AdaptationRequired(False, False)
        else:
            if step >= self.next_adapt:
                result = self.AdaptationRequired(True, True)
                self.adapt_every *= self.growth_factor
                self.next_adapt += self.adapt_every
            else:
                result = self.AdaptationRequired(False, False)
        return result
