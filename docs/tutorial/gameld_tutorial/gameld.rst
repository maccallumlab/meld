========================================================
GaMELD: Accelerating Biomolecular Landscape Exploration.
========================================================

Gaussian accelerated MELD (GaMELD) combines the strengths of Gaussian Accelerated Molecular Dynamics (GaMD) and Modeling Employing Limited Data (MELD) to navigate complex energy landscapes.

Reference: M. Caparotta, A. Perez, When MELD meets GaMD: Accelerating Biomolecular Landscape Exploration.

Introduction
-------------
    In this tutorial we use GaMELD to accelerate the folding of Ubiquitin, PDB ID: `1UBQ <https://www.rcsb.org/structure/1UBQ>`_.

Pre-requisites
---------------
    Besides installing `MELD <https://github.com/maccallumlab/meld>`_, you´ll also need to install `GaMD-OpenMM <https://github.com/MiaoLab20/gamd-openmm>`_.

Input
------
    File *sequence.dat*: Amino acids sequence from the `FASTA sequence <https://www.rcsb.org/fasta/entry/1UBQ/display>`_ in the PDB.

    File *ss.dat*: Secondary structure prediction. We used the `PsiPred Server <http://bioinf.cs.ucl.ac.uk/psipred/>`_.

Configuration
--------------
    File *setup_CPI_gameld.py*.

    This file largely comprises the same code as a `previous work <https://doi.org/10.1073/pnas.1515561112>`_ that implements CPIs (Coarse Physical Insights) in MELD. Below, we emphasize the significant changes in the configuration for utilizing GAMELD. For a detailed explanation of the parameters specific to GaMD, please refer to `this article <https://doi.org/10.1021/acs.jpcb.2c03765>`_.

    .. code-block:: python

        N_REPLICAS = 9  # Number of replicas
        N_STEPS = 100000
        GAMD: bool = True   # Apply GaMD? True/False
        TIMESTEPS = 2500
        CONVENTIONAL_MD_PREP = 100
        CONVENTIONAL_MD = 1000
        GAMD_EQUILIBRATION_PREP = 900
        GAMD_EQUILIBRATION = 9000

        ...

        build_options = meld.AmberOptions(
            forcefield="ff14sbside",
            implicit_solvent_model="gbNeck2",
            cutoff=1.8*u.nanometer,
            enable_gamd=GAMD, 
            boost_type_str="upper-total", # Implemented modes: upper-dual, lower-dual, upper-total, lower-total, lower-dihedral and upper-dihedral.
            conventional_md_prep=CONVENTIONAL_MD_PREP * TIMESTEPS,
            conventional_md=CONVENTIONAL_MD * TIMESTEPS,
            gamd_equilibration_prep=GAMD_EQUILIBRATION_PREP * TIMESTEPS,
            gamd_equilibration=GAMD_EQUILIBRATION * TIMESTEPS,
            total_simulation_length=N_STEPS * TIMESTEPS,
            averaging_window_interval=TIMESTEPS,
            sigma0p=6.0,
            sigma0d=6.0,
            random_seed=0,
            friction_coefficient=1.0
        )

    Another important consideration is to add the *enable_gamd* flag in the *RunOptions* function:

    .. code-block:: python

            # create the options
            options = meld.RunOptions(timesteps=TIMESTEPS, minimize_steps=20000, 
                n_replicas = N_REPLICAS, enable_gamd=GAMD) 

Build and execute it just like any other MELD simulation.

Results
--------
    Superposed experimental (red) vs best predicted (blue) structure (RMSD 2.5 A).

    .. image:: native_vs_predicted.png
        :width: 350
