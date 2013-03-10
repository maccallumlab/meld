import numpy
import math


class EqualAcceptanceAdaptor(object):
    '''
    Adaptor based on making acceptance rates uniform.
    '''
    def __init__(self, n_replicas, min_acc_prob=0.1):
        '''
        Initialize adaptor

        Parameters
            n_replicas -- number of replicas
            min_acc_prob -- all acceptence probabilities below this value will be raised to this value

        '''
        self.n_replicas = n_replicas
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

    def adapt(self, previous_lambdas):
        '''
        Compute new optimal values of lambda.

        Parameters
            previous_lambdas -- a list of the previous lambda values
        Returns
            a list of the new, optimized lambda values

        '''
        self._compute_accept_probs()
        self._compute_t_len()

        # put the t_lens on a grid
        alpha_grid = numpy.linspace(0., 1., 5000)
        t_lens = numpy.interp(alpha_grid, previous_lambdas, self.t_lens)

        # compute the desired t_lens based on equal spacing
        even_spacing = numpy.linspace(0, t_lens[-1], self.n_replicas)

        # compute the values of lambda that will give the desired evenly spaced
        # t_lens
        new_alphas = numpy.interp(even_spacing[1:-1], t_lens, alpha_grid)
        new_alphas = [x for x in new_alphas]
        return [0.] + new_alphas + [1.]

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
