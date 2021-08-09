===========================
Analysis of MELD simulation
===========================

In this tutorial, we describe analysis tools available in MELD to extract 
information out of simulation after it's finished.

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

