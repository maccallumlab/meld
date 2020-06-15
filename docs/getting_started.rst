=========================
Getting Started with MELD
=========================

Running your first REMD run
===========================

In essence MELD provides a very customizable way of running REMD. We can decide to change temperature, hamiltonian or both along the replica
ladder and the way we do it is very customizable. The minimum number of replicas you need to run MELD is 2. In this tutorial you will learn the basic elements to run your first Replica exchange molecular Dynamics. We will create a setup.py script with all the python commands to setup a simulation.

Loading Some Helpful Modules
--------------------
import numpy as np
from meld.remd import ladder, adaptor, leader
from meld import comm, vault
from meld import system
from meld import parse


Setting up the system
---------------------

MELD runs can start from a PDB file, fasta sequence file with no header or sequence chain:
::

   p = protein.ProteinMoleculeFromSequence("NALA ALA CALA")        
   
   p = system.ProteinMoleculeFromPdbFile("example.pdb")

   sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
   p = system.ProteinMoleculeFromSequence(sequence)

Once we have the protein system we have to specify a force field. Current options are ff12sb, ff14sbside (ff99backbone) or ff14sb:
::
   b = system.SystemBuilder(forcefield="ff14sbside")

Now we generate the topoloty/coordinate files to start simulations:
::
   s = b.build_system_from_molecules([p])


At this point we can start defining different ways to setup the replica ladders. In MELD we have a parameter called alpha, with values between [0,1] which map on to the replica ladder. 1 will correspond to the highest replican and 0 to the lowest replica. Given an alpha value we can map all the restraints and the temperature that replica should have. We will use a geometric scaling of the temperatures. As an initial example let us setup a temparature replica ladder that will expand the whole replica space going from 300K to 450K.
::

   s.temperature_scaler = system.GeometricTemperatureScaler(0, 1.0, 300., 450.)

Next, we will specify some options for the system. This is where we can decide things like timestep, solvent model or minimizer to use. Distances are in nm. So a 1.8 cutoff is 1.8nm cutoff for non-bonded. The use_bigger_timestep keyword allows us to use HydrogenMassRepartitioning to increase timesteps to 4.5fs and the timesteps keyword tells us the steps to take before REMD exchange attempts (in this case 4.5*11111fs ~ 50ps). We will minimized for 20000 steps using OpenMM's minimizer and will use the GB neck 2 implicit solvent model:
::


   options = system.RunOptions()

   options.use_bigger_timestep = True
   options.cutoff = 1.8
   options.timesteps = 11111
   options.minimize_steps = 20000
   options.implicit_solvent_model = 'gbNeck2'

Storing simulation output
-------------------------

By default, the setup script will generate a Data directory with all the information needed to run the system. To generate the system we will need to specify the size of the system (number of atoms), the number of replicas to use and how often to generate writing blocks. Writing blocks is related to how we store the information. A block_size of 1 would mean that everytime there is a swap attempt (50ps in this example) we would save the information of the system (coordinates, velocities, ...) in a new block. We typically write in blocks of 100 (5ns). This block system is useful to ensure there is no corruption in the system. When restarting MELD, we always restart from the last complete BLOCK:
::

    N_REPLICAS = 2
    BLOCK_SIZE = 100
    # create a store
    store = vault.DataStore(s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

Replica ladder properties
-------------------------
Here is where we specify which replicas are going to exchange with which, and how many swap attempts we are going to try. In this example, replicas will attempt exchanges with replicas that are adjacent to them. After each trial, we will update replica information and will repeat this for n_trials attemps. This means that a given conformation could scale up and down more than one position during a swapping attempt.
We can choose and adaptor policy for the REMD ladder. In essence, this alows to change the alpha values on the flight to improve a certain criteria. In this example, we want all replicas to exchange at the same ratio. This means that if there are bottlenecks in the Replica ladder, the system will try to put replicas that are exchanging too infrequently closer to each other while separating those that are exchanging too frequently.

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=100)
    policy = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)

    remd_runner = leader.LeaderReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

Initialize the communicators and starting replica conformations
---------------------------------------------------------------
    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS)
    store.save_communicator(c)


    def gen_state(s, index):
        #Start from same conformation, no initial velocicities
        pos = s._coordinates
        pos = pos - np.mean(pos, axis=0)
        vel = np.zeros_like(pos)
        #Set position in replica ladder -- initially spaced equally
        alpha = index / (N_REPLICAS - 1.0)
        s._box_vectors=np.array([0.,0.,0.])
        energy = 0
        return system.SystemState(pos, vel, alpha, energy,s._box_vectors)

    # create and save the initial states
    states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

Running the system
------------------
After executing python setup.py you should get a Data directory with all the files needed to run MELD. Use your queing system to submit an mpi job with the number of replicas you have indicated. Currently, we need one GPU for each replica.
::
    aprun -n 2 -N 1 launch_remd --debug

