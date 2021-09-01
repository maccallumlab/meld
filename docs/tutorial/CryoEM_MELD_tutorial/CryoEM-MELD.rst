=========================
Refining AF2 prediction with MELD using Cryo-EM density map. 
=========================
Introduction
=========================
Here, in this tutorial, we will show how to refine an AF2 prediction with MELD simulation. We will use Cryo-EM map to extract the best strucutre from MELD generated ensemble. For this particular case we have choose heat shock protein system with known experimental structure (RCSB-PDB code 2xhx). This is a 213 residue protein starting from res 11 to res 223. AF2 is used first to predict the folded stucture from sequence (thanks to John from Singharoy Group at ASU and Dr. Rowley at Carleton University) and procol can be found elsewhere. The prediction has res 16 to 223 and it mostly looks pretty good except two region, shown below:

.. image:: Screenshot from 2021-09-01 12-07-25.png
     :width: 450

On the left we have the native structure and on the right we have ALphaFold2 prediction. The part matching well with expermental structure is shown in Red color and the parts that do not match, are shown in Cyan. From residue 109 to 136 the loop has different secondary structure and the orientation is incorrect and at the terminal from res 16 to 40, the prediction is missing the beta sheet pairing. Rest of the protein match well:

.. image:: Screenshot from 2021-09-01 12-00-34.png
    :width: 450
Here we are only showing the part that matches well. In blue we have native and in orange we have the prediction.

We looked at the per residue confidence score for the model prediction AF2. The scores correlate well with the quality of the predicted structure:
.. image:: Screenshot from 2021-09-01 12-09-06.png
    :width: 450
Here, as score increases color changes from blue--> white --> red. In other words blue represent the lower score and Red represent the higher score. Notice, the region which do not match with native, mostly have blue, white or faded red color. Also some hehix-turn have faded red color. 
