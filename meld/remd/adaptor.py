import numpy
import math
from collections import namedtuple


class EqualAcceptanceAdaptor(object):
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
        self.n_replicas = n_replicas
        self.adaptation_policy = adaptation_policy
        self.min_acc_prob = min_acc_prob
        self.success = None
        self.attempts = None
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
        assert i in range(self.n_replicas - 1)
        self.attempts[i] += 1
        if accepted:
            self.success[i] += 1

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
            lambda_grid = numpy.linspace(0., 1., 5000)
            t_lens = numpy.interp(lambda_grid, previous_lambdas, self.t_lens)

            # compute the desired t_lens based on equal spacing
            even_spacing = numpy.linspace(0, t_lens[-1], self.n_replicas)

            # compute the values of lambda that will give the desired evenly spaced
            # t_lens
            new_lambdas = numpy.interp(even_spacing[1:-1], t_lens, lambda_grid)
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
        self.success = numpy.zeros(self.n_replicas - 1)
        self.attempts = numpy.zeros(self.n_replicas - 1)
        self.accept_probs = None
        self.t_lens = None

    def _compute_accept_probs(self):
        # default to 50 percent if there hasn't been a trial
        self.success[self.attempts == 0] = 1.
        self.attempts[self.attempts == 0] = 2.

        self.accept_probs = self.success / self.attempts

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
