#! /usr/bin/env python

#ATOM      3  CA  ARG     1       3.970   2.846  -0.000  1.00  0.00
#ATOM    142 HH22 ARG     6      26.753   9.204  -4.004  1.00  0.00


import sys

fi = open(sys.argv[1],'r')

all_lines = []
for line in fi:
    if "ATOM" in line:
        beg = line[0:30]
        xyz = line[30:54]
        end = line[54:]
        (x,y,z) = xyz.split()
        x = float(x) + 30.
        y = float(y) + 30.
        z = float(z) + 30.
        all_lines.append('{}{:8.3f}{:8.3f}{:8.3f}{}'.format(beg,x,y,z,end))
    else:
        all_lines.append(line)

    xyz = "xxxxxxxxyyyyyyyyzzzzzzzz"
    #print(beg)
    #print(xyz)
    #print(end)
    #print(line)

with open('pep_shifted.pdb','w') as fo:
    for i in all_lines:
        fo.write(i)
