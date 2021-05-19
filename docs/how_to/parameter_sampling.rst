.. _parameter-sampling-howto:

=============================
How to use parameter sampling
=============================

Meld supports the Bayesian sampling of parameters.
See :ref:`parameter-sampling-background` for background.

MELD currently only supports parameter sampling for the
number of active restraints in a group and for the number
of active groups in a collection. In future other sampled
parameters will be added.

Types of parameters
-------------------

MELD supports both :class:`ContinuousParameter` and :class:`DiscreteParameter`
parameters. Continuous parameters can take on any floating point value.
Discrete parameters are restricted to integer values.

Currently, MELD only supports parameter sampling for groups and collections,
which use :class:`DiscreteParameter`, so only these will be discussed.

Priors
------

A prior encodes the prior beleif about the likely values of a parameter.
The key quantity is the log of the prior probability, which MELD treats
as an energy in units of :math:`k_BT`.

.. note::
    This means that the prior has the same strength regardless of
    temperature, whereas the physics-based prior (i.e. the energy)
    becomes flatter with increasing temperature.

There are two types of priors. A :class:`DiscreteUniformPrior` encodes a flat
prior with no prefered value. A :class:`DiscreteExponentialPrior` encodes
a prior that exponentially favors low or high values, depending on the sign
of the parameter ``k`` that controls how strong this preference is.

Samplers
--------

Each parameter must have a sampler, which determines the minimum and maximum allowed
values. The sampler also sets the step size of Monte Carlo moves for this parameter.

Monte Carlo Steps
-----------------

After each round of molecular dynamics steps, the parameters are updated using a series
of Monte Carlo trials. The number of trials is controlled by the :code:`param_mcmc_steps`
option of the :class:`RunOption` object.

Puting it together
------------------

Assuming we have a MELD system called :code:`system`, we can create a new parameter:

.. code-block:: python
    :linenos:

    from meld.system import param_sampling

    prior = param_sampling.DiscreteExponentialPrior(k=1.0)
    sampler = param_sampling.DiscreteSampler(50, 100, 5)
    param = system.param_sampler.add_discrete_parameter("param", 75, prior, sampler)

**Line 3**: Creates a prior that favors higher values with :math:`-1 k_BT` energy
contribution for each unit of increase in the parameter.

**Line 4**: Creates a sampler with a minimum value of 50 and a maximum value of 100,
both inclusive. It uses a step size of 5, so that random moves are attemped
:math:`\pm 5` from the current value.

**Line 5**: Creates the parameter. Each parameter must have a unique name, :code:`"param"`
in this case. We must also specify the initial value, here :code:`75`. The resulting
object, :code:`param`, can then be passed to other places in the MELD code base.
Currently this can only be used as :code:`num_active` when creating a group or collection.