
MELD runs can start from a PDB file, fasta sequence file with no header or sequence chain:
24
​
25
.. code-block:: python
26
​
27
    p = subsystem.SubSystemFromSequence("NALA ALA CALA")        
28
   
29
    p = system.SubSystemFromPdbFile("example.pdb")
30
​
31
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
32
    p = system.SubSystemFromSequence(sequence)
33
​
34
Once we have the protein system we have to specify a force field. Current options are ff12sb, ff14sbside (ff99backbone) or ff14sb:
