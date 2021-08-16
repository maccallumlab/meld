===========================
Analysis of MELD simulation
===========================

In this tutorial, we describe analysis tools available in MELD to extract 
information out of simulation after it's finished.

Restart simulation
------------------
The simulation might stop in the middle and we would like to restart it 
from where the simulation failed. This can be done with 

.. code-block:: python

    prepare_restart --prepare-run

The simulation will restart from the nearest block saved in :code:`Data/`.


Extract trajectory
------------------
After simulation is done, we can use :code:`extract_trajectory` to extract frames 
saved in :code:`Data/`. The options can be seen from :code:`extract_trajectory --help`

.. code-block:: python

    usage: extract_trajectory [-h] {extract_traj,extract_traj_dcd,extract_last,extract_random,follow_structure,follow_dcd} ...

    Extract frames from a trajectory.

    positional arguments:
    {extract_traj,extract_traj_dcd,extract_last,extract_random,follow_structure,follow_dcd}
        extract_traj        extract a trajectory for one replica
        extract_traj_dcd    extract a trajectory for one replica
        extract_last        extract the last frame for each replica from a trajectory
        extract_random      extract random frames from a trajectory for reseeding
        follow_structure    follow a structure replica through the ladder
        follow_dcd          follow a structure replica through the ladder

    optional arguments:
    -h, --help            show this help message and exit

e.g. extracting frame 1 to 1000 on replica 0 in :code:`.dcd` format. 

.. code-block:: python
    
    extract_trajectory extract_traj_dcd --start 1 --end 1000 --replica 0 trajectory.00.dcd 

Visulize replica exchange
-------------------------
It's important to check the exchange between replicas along simulation. We provide
several ways to extract such information. The two of them used mostly are:  

.. code-block:: python
    
    analyze_remd visualize_trace

which will show plot like the following:

.. image:: compare_trace.png

Each color represents where each replica is along simulation. The left plot shows 
better exchanges because replicas got exchanged frequently among replica ladders. This 
can also be seen from:

.. code-block:: python
    
    analyze_remd visualize_fup

which will show plot like the following:

.. image:: compare_fup.png

The x-axis indicates all 30 replicas and y-axis represents the probability of going 
up (black) or down (:red:`red`) along replica ladders. The right plot reflects higher
probability of going up, which shows worse exchange than the left one.

Extract representative
----------------------
Once simulation is done, we want to see what is the representative structure
among all conformations it sampled. This requires a similarity measure like RMSD between 
selected atoms of conformations and usually use clustering tools available in open source 
packages such as :code:`scikit-learn` and :code:`cpptraj` to group conformations with high similarity.
Here we provide a rather simple but effective tool :code:`density_rank` to extract the representative among selected
samples. The calculation is based on the contacts formed between selected atoms and the assumption 
is that the representative should be the conformation having more contacts formed and got sampled
more times such as bound and folded state. 

The full description can be seen from :code:`density_rank --help`

.. code-block:: python

    usage: density_rank [-h] [-traj path [path ...]] [-top path] [-start N [N ...]] [-end N [N ...]] [-sieve N [N ...]]
                        [-inter res_0 res_1 skip_0 res_2 res_3 skip_1 [res_0 res_1 skip_0 res_2 res_3 skip_1 ...]] [-inter_cutoff cutoff [cutoff ...]]
                        [-intra res_0 res_1 skip [res_0 res_1 skip ...]] [-intra_cutoff cutoff [cutoff ...]] [-extract_traj density range [density range ...]]

    optional arguments:
    -h, --help            show this help message and exit
    -traj path [path ...] path of trajectories
    -top path             path of topology
    -start N [N ...]      select start frame of each trajectory
    -end N [N ...]        select end frame of each trajectory
    -sieve N [N ...]      skip every N frames of each trajectory

    -inter res_0 res_1 skip_0 res_2 res_3 skip_1 [res_0 res_1 skip_0 res_2 res_3 skip_1 ...]
                            calculate contact in range [res_0:res_1:skip_0] and [res_2:res_3:skip_1] with inter_cutoff, multiple ranges are allowed, total length should be
                            multiple of 6

    -inter_cutoff cutoff [cutoff ...] inter_contact cutoff, unit in nm

    -intra res_0 res_1 skip [res_0 res_1 skip ...]
                            calculate contact in range [res_0:res_1:skip] with intra_cutoff, multiple ranges are allowed, total length should be multiple of 3
    
    -intra_cutoff cutoff [cutoff ...] intra_contact cutoff, unit in nm

    -extract_traj density_range [density_range ...] extract samples with specified density range, default not extracting.



Here are a couple of examples:

For extracting representative in binding simulation, we usually define the contacts
between selected residues in host and ligand. In addition, a cutoff needs to be set, which can be estimated
from sampled conformations.

.. code-block:: python
    
    density_rank -traj trajectory.00.dcd trajectory.01.dcd trajectory.02.dcd -top topol.prmtop -start 500 500 600 -end 900 800 700 -sieve 2 2 2 -inter 0 67 2 67 88 2 -inter_cutoff 0.7

This will process contacts between residues 1-67 and residues 68-89 with cutoff 0.7 nm every 2 frames among 500 to 900 of :code:`trajectory.00.dcd`, 500 to 800 of :code:`trajectory.01.dcd` and 600 to 700 of :code:`trajectory.02.dcd`.

The output files are :code:`density.npy` (density value of each conformation) with associated plot :code:`density_rank.png` and the pdb file :code:`top_density.pdb` of conformation with 
highest density as representative of selected trajectory set.

For extracting representative in folding simulation, we usually define the the intra-contacts among selected residues in the molecule.

.. code-block:: python
    
    density_rank -traj trajectory.00.dcd -top topol.prmtop -start 500 -end 9000 -sieve 2 -intra 1 168 2  -intra_cutoff 0.6

This will process pairwise contacts in residue set 1-168 with cutoff 0.6 nm every 2 frames among 500 to 9000 of :code:`trajectory.00.dcd`.
