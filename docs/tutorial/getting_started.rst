=========================
Getting Started with MELD
=========================

Running your first REMD run
===========================

In essence MELD provides a customizable way of running REMD. We can decide to
change temperature, hamiltonian or both along the replica. In this tutorial you
will learn the basic elements to run your first Replica exchange molecular
Dynamics. We will create a setup.py script with all the python commands to setup
a simulation.

Loading Some Helpful Modules
----------------------------
.. code-block:: python

    import numpy as np
    import meld
    from meld import unit as u


Setting up the system
---------------------

You can setup your protein either from a pdb file or from sequence.

.. code-block:: python

    # Option 1: from a pdb file
    protein = meld.AmberSubSystemFromPdbFile("example.pdb")

    # Option 2: from sequence in a file
    protein = meld.AmberSubSystemFromSequence("NALA ALA CALA")        

    # Option 3: from a sequence
    sequence = meld.get_sequence_from_AA1(filename='sequence.dat')
    protein = meld.AmberSubSystemFromSequence(sequence)

Once we have the protein system we have to specify a force field. Current
options are ``ff12sb``, ``ff14sbside``, or ``ff14sb``. We can also chose the
implicit solvent model, in this case ``gbNeck2``.

We also need to setup various options about how the simulation will be run, including
the timestep and any nonbonded cutoffs. By setting ``use_big_testep``, we will be using
a 3.5 fs timestep with hydrogen mass repartitioning.

.. code-block:: python

    build_options = meld.AmberOptions(
        forcefield="ff14sbside",
        implicit_solvent_model="gbNeck2",
        use_big_timestep=True,
        cutoff=1.8*u.nanometer
    )
    builder = system.AmberSystemBuilder(build_options)

Now we generate the system:

.. code-block:: python

    system = builder.build_system([protein]).finalize()


At this point we can start defining different ways to setup the replica ladders.
In MELD we have a parameter called alpha, with values between [0,1] which map on
to the replica ladder. The lowest replica will always have ``alpha=1.0``, whereas
the highest replica will have ``alpha=1.0``. The intermediate replcias will
initial linearly interpolate between these values and will be modified
adaptively during the simualtion to attain approximately equal average
acceptance probabilities between adjacent replicas.

The various properties of the system, e.g. temperatures or force constants, are
set based on ``alpha`` through the use of scalers.

We will use a geometric scaling of the temperatures. As an initial example let
us setup a temparature replica ladder that will cover the whole replica space
going from 300K to 450K.

.. code-block:: python

   system.temperature_scaler = meld.GeometricTemperatureScaler(0, 1.0, 300., 450.)

Next, we will specify some options for the system. The ``timesteps`` parameter
controls the number of molecular dynamics steps between exchanges. Since we set
``use_big_timestep`` the value of ``14_286`` corresponds to 50 ps between exchanges.
The system will be minimized for ``minimize_steps`` before the simulation starts.

.. code-block:: python

   options = meld.RunOptions(
       timesteps = 14_286
       minimize_steps = 20_000,
    )


Setting up replica exchange
---------------------------

Next, we need to setup replica exchange. The main parameters to set are: the
total number of rounds of simulation to run ``n_steps``, the number of replicas
``n_replicas``, the number of trials per replica exchange step ``n_trials``
(defaults to ``n_replicas**2``). There are also various options for setting
adaptation of ``alpha``, but the defaults should generally work fine.

In this case, we're doing 5000 steps of replica exchange, each with 50 ps of
molecular dynamics, for a total of 250 ns.

.. code-block:: python

    remd = meld.setup_replica_exchange(system, n_replicas=4, n_steps=5000)


Storing simulation output
-------------------------

Finally, we need to store all of this setup information to disk in preparation for running
the calculation

.. code-block:: python

    meld.setup_data_store(system, options, remd)


Full setup script
-----------------

The full setup script should be saved as something like ``setup_simulation.py``.

.. code-block:: python

    import numpy as np
    import meld
    from meld import unit as u

    protein = meld.AmberSubSystemFromSequence("NALA ALA CALA")        

    build_options = meld.AmberOptions(
        forcefield="ff14sbside",
        implicit_solvent_model="gbNeck2",
        use_big_timestep=True,
        cutoff=1.8 * u.nanometer
    )
    builder = system.AmberSystemBuilder(build_options)

    system = builder.build_system([protein]).finalize()
    system.temperature_scaler = meld.GeometricTemperatureScaler(0, 1.0, 300. * u.kelvin, 450. * u.kelvin)

    options = meld.RunOptions(
        timesteps = 14_286
        minimize_steps = 20_000,
    )
    remd = meld.setup_replica_exchange(system, n_replicas=4, n_steps=5000)
    meld.setup_data_store(system, options, remd)


Running the system
------------------

After executing ``python setup_simulation.py`` you should get a Data directory
with all the files needed to run MELD. Use your queing system to submit an MPI
job with the number of replicas you have indicated. Currently, we need one GPU
for each replica.

.. code-block:: shell

    aprun -n 2 -N 1 launch_remd --debug

For debugging purposes, it is also possible to run locally on a laptop or
workstation, although this can be very slow.

.. code-block:: shell

    launch_remd_multiplex --debug --platform CUDA

You can replace ``CUDA`` with ``Reference`` if you do not have a GPU, athough this
will be quite slow.