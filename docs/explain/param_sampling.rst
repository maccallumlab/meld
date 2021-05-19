.. _parameter-sampling-background:

===============================
Bayesian Sampling of Parameters
===============================

MELD has the ability to perform Bayesian sampling over
parameters, which allows for the value of those parameters
to "be decided by the system".

Formally, MELD samples the joint posterior distribution of
structures :math:`x` and parameters :math:`y` given some
data :math:`D`.

.. math::
    p(x, y | D) \propto p(x) p(y) p(D | x, y)

where, :math:`p(x, y | D)` is the posterior distribution that we
are sampling over, :math:`p(D | x , y)` is the likelihood function,
and :math:`p(x)` and :math:`p(y)` are the priors over the structures
:math:`x` and parameters :math:`y`.

Typically, we are interested in the marginal distribution :math:`p(x|D)`,
which is obtained by integrating out :math:`y`.

Practical details can be found at :ref:`parameter-sampling-howto`.
