=========================
Refining AF2 prediction with MELD using Cryo-EM density map. 
=========================
Introduction
=========================
Here, in this tutorial, we will show how to refine an AF2 prediction with MELD simulation. We will use Cryo-EM map to extract the best strucutre from MELD generated ensemble. For this particular case we have choose heat shock protein system with known experimental structure *2xhx.pdb* (RCSB-PDB code 2xhx). This is a 213 residue protein starting from res 11 to res 223. AF2 is used first to predict the folded stucture from sequence (thanks to John from Singharoy Group at ASU and Dr. Rowley at Carleton University) and protocol can be found elsewhere. The provided prediction (*AF-P07900-F1-model_v1_short.pdb*) has residue 16 to 223 and it mostly looks pretty good except two region, shown below:

.. image:: both-sidebyside.png
     :width: 450

On the left we have the native structure and on the right we have ALphaFold2 prediction. The part matching well with expermental structure is shown in Red color and the parts that do not match, are shown in Cyan. From residue 109 to 136 the loop has different secondary structure and the orientation is incorrect and at the terminal from res 16 to 40, the prediction is missing the beta sheet pairing. Rest of the protein match well:

.. image:: both-overlap.png
    :width: 450
Here we are only showing the part that matches well. In blue we have native and in orange we have the prediction.

We looked at the per residue confidence score for the model prediction AF2. The scores correlate well with the quality of the predicted structure:

.. image:: confidence.png
    :width: 450
Here, as score increases color changes from blue--> white --> red. In other words blue represent the lower score and Red represent the higher score. Notice, the region which do not match with native, mostly have blue, white or faded red color. Also some hehix-turn have faded red color. 

To refine/remodel these region, we run MELD simulation starting from the AF2 prediction. We put cartesian restraint on the coordinate of CA atom in the region where the confidence score is higher that 90% and no restraint for less that 90%. We can find this score in the second last column in *AF-P07900-F1-model_v1_short.pdb* file.



==========================================
Starting conformation for MELD simulation
==========================================

To run MELD simulation, first we need to install MELD in the cluster. Instruction for Conda installation of MELD is given in the github page. Once we have MELD installed successfully, we then need to minimize AF2 predicted model with AMBER. Before that, we used tleap to generate topology and coordinate file using *tleap.in* file. We execute this as:

*tleap -f tleap.in*

This returns *system.top* and *system.mdcrd* files and we minimize the coordinate with *minimize_GPU.sh*. This returns minimized coordinate *eq0_template.rst* which can be converted to a pdb file:

*ambpdb -p system.top -c eq0_template.rst >minmized.pdb*

The *minimized.pdb* is moved to *TEMPLATES/* directory.

Now as we have the starting conformation, we are reasy to submit MELD simulation.

===============================
Configuring the MELD simulation
===============================

To configure the MELD simulation we use a python script *setup_MELD.py*

First we need to import the libraries:


.. code-block:: python
    
    import numpy as np
    from meld.remd import ladder, adaptor, leader
    from meld import comm, vault
    from meld import system
    from meld import parse
    import meld.system.montecarlo as mc
    from meld.system.restraints import LinearRamp,ConstantRamp
    from collections import namedtuple
    import glob as glob
