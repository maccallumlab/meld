#!/bin/bash
###add headers




cat<<EOF>setup_random.py
#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from meld import system
from meld import parse

def setup_system():
    # load the sequence
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    n_res = len(sequence.split())
    
    # build the system
    p = system.ProteinMoleculeFromSequence(sequence)
    b = system.SystemBuilder(forcefield="ff14sbside")
    s = b.build_system_from_molecules([p])
setup_system()
EOF

python  setup_random.py > tleap.in
tleap -f tleap.in

cat<<EOF>minimize.in
Stage 1 - minimisation of 1sr protein
 &cntrl
  imin=1, maxcyc=1000, ncyc=500,
  cut=999., rgbmax=999.,igb=8, ntb=0,
  ntpr=100,
 /
EOF

pmemd -O \
    -i minimize.in \
    -o minimize.out \
    -c system.mdcrd \
    -ref system.mdcdr \
    -r eq0.rst \
    -p system.top \
    -e eq0.ene \
    -x eq0.netcdf

cpptraj system.top<<EOF>&err
trajin eq0.rst
trajout min_peptide.pdb pdb
go
EOF

