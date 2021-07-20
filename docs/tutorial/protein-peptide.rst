
=========================
Simulating protein-peptide complex and predicting native state
=========================
Introduction
=========================

Here we will learn how to setup a protein-peptide system and run simulation and analyze the trajectory to predict the native complex structure. For that we will start the simulation from unbound conformation of both receptor (protein) and ligand (peptide) and allow data and force field to guide to the native bound conformation of the complex. Here we will work with a known protein complex of ET domain of BRD3 protein with MLV peptide (RCSB PDB code 7jq8). Unbound form of BRD3 ET protein is also available as PDB code 7jmy.

Setup of the starting system
============================
The protein sequence is

    HHHHHHSHMGKQASASYDSEEEEEGLPMSYDEKRQLSLDINRLPGEKLGRVVHIIQSREPSLRDSNPDEIEIDFETLKPTTLRELERYVKSCLQKK
where first 28 residues are purification tag. So we will exclude that in the simulation and we have 

    SYDEKRQLSLDINRLPGEKLGRVVHIIQSREPSLRDSNPDEIEIDFETLKPTTLRELERYVKSCLQKK
and the peptide sequence is 


































MELD runs can start from a PDB file, fasta sequence file with no header or sequence chain:

.. code-block:: python

    p = subsystem.SubSystemFromSequence("NALA ALA CALA")
    
    p = system.SubSystemFromPdbFile("example.pdb")
    
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    
    p = system.SubSystemFromSequence(sequence)
    
Once we have the protein system we have to specify a force field. Current options are ff12sb, ff14sbside (ff99backbone) or ff14sb:
