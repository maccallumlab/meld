
MELD runs can start from a PDB file, fasta sequence file with no header or sequence chain:

.. code-block:: python

    p = subsystem.SubSystemFromSequence("NALA ALA CALA")
    
    p = system.SubSystemFromPdbFile("example.pdb")
    
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    
    p = system.SubSystemFromSequence(sequence)
    
Once we have the protein system we have to specify a force field. Current options are ff12sb, ff14sbside (ff99backbone) or ff14sb:
