
=========================
Simulating protein-peptide complex and predicting native state
=========================
Introduction
=========================

Here we will learn how to setup a protein-peptide system and run simulation and analyze the trajectory to predict the native complex structure. For that we will start the simulation from unbound conformation of both receptor (protein) and ligand (peptide) and allow data and force field to guide to the native bound conformation of the complex. Here we will work with a known protein complex of ET domain of BRD3 protein with MLV peptide (RCSB PDB code 7jq8). Unbound form of BRD3 ET protein is also available as PDB code 7jmy. Here this tutorial is done with meld/0.4.20 version. 

Setup of the starting system
============================
The protein sequence is

    HHHHHHSHMGKQASASYDSEEEEEGLPMSYDEKRQLSLDINRLPGEKLGRVVHIIQSREPSLRDSNPDEIEIDFETLKPTTLRELERYVKSCLQKK
where first 28 residues are purification tag. So we will exclude that in the simulation and we have 

    SYDEKRQLSLDINRLPGEKLGRVVHIIQSREPSLRDSNPDEIEIDFETLKPTTLRELERYVKSCLQKK
and the peptide sequence is 
    
    SRLTWRVQRSQNPLKIRLTREAP
Since we want to start from a free peptide conformation, we generate a minimized PDB file for the peptide in an extended conformation. We can use *setup_from_random.sh* script in this directory for this purpose. This script uses 'setup_system' function to generate tleap input. Then tleap generates topology and initial coordinate which we minimize using amber to generate a minimized pdb for the peptide. The 'setup_system' script follows as:

.. code-block:: python

    def setup_system():
        # load the sequence
        sequence = parse.get_sequence_from_AA1(filename='sequence.dat')  # above mentioned peptide sequence is in sequence.dat file
        n_res = len(sequence.split())
        # build the system
        p = system.ProteinMoleculeFromSequence(sequence)
        b = system.SystemBuilder(forcefield="ff14sbside")   # we use ff14SB side and ff99SB backbone forcefield  
        s = b.build_system_from_molecules([p])            # build the pdb file

We will combine this minimized peptide pdb and unbound protein pdb (minimized as well) together. One thing we have to make sure is that peptide is atleast 30 angstrom far away from the receptor. There are several ways to shift coordinate of either receptor or peptide. Using *change_coor.py* script is one of the ways. Finally we have *minimized_complex.pdb*. A ascreenshot of the starting system from vmd is given below.

.. image:: https://github.com/arupmondal835/meld/tree/master/docs/tutorial/start.png 

Prepare the input restraint file for this system:
=================================================
As we already learned, MELD is based on Bayesian frameworks, it uses data coming from all sort of sources and an atomistic force field. Data plays a major role here, it helps to limit the conformational landscape and help to find minima faster. The principle of MELD is explained in detailed somewhere else. 
For this particular example we will use distance restraints between protein -peptide residue pairs. Since the structure is known, we use the native complex structure to determine protein-peptide CA pairs which are within 8 angstrom from each other and use those in the simulation to guide the binding. This protein-peptide restrain file is added here as *protein_pep_all.dat* which is calculated using popular python library MDTraj and the script is also given here as *pdb_contacts.py*. Few restraints from this file are shown here: 
    5 CA 82 CA 0.5846176743507385

    6 CA 81 CA 0.5934389233589172

    6 CA 82 CA 0.5739095211029053

    9 CA 81 CA 0.6932587623596191

    ...
    
    ...
    
Here, residue 1 to 68 is corresponding to the protein and 69 to 91 for peptide. The first column is residue number of protein and second column is atom in protein residue, 3rd and 4th column are respectively residue number and atoms for peptide and the 5th column is the distance between those two atom in nanometer. To be specific, the 1st row means CA of 5th residue should be 7.5 angstrom away from CA of 82nd residue (which is 82-68=14th residue in peptide) in the bound conformation. These restraints are mere for tutorial purposes, for real system we need to get data from experiment or statistical analysis as complex structure will be unknown. Also one gap between each restraints are importants for this particular simulation setup as we are defining all these restraints as a collection, and inside collection we have groups seperated by blank line and in each group we have restriants. Here each group only has one restriant.

We are using unbound protein conformation in our simualtion, the protein will probably go through conformational change upon complex formation- but we can expect it keep its fold intact. For this, we calculate interprotein residue pairs within 8 angstrom and put distance restraints on them in a similar way to peptide. We can use similar script for this purpose as well and it generates *protein_contacts.dat* file. 
 




    
































