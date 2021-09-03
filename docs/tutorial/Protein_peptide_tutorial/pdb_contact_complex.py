import mdtraj as md
from matplotlib import pyplot as plt
import sys
import numpy as np

traj=md.load_pdb('native_complex.pdb')
pair=[]
for i in range(68):
    for j in range(68,91):
        pair.append([i,j])

dist = md.compute_contacts(traj, pair, scheme='CA')
print(len(dist[1]))
dis=np.reshape(dist[0],(len(dist[1]),1))

dist_list=dis.tolist()

file = open("protein_pep_all.dat","w")
accpt=[]
for i in zip(pair, dist_list):
    j  = [val for sublist in i for val in sublist]
    
    if j[2] <= 0.8:
        accpt.append([j[0],j[1],j[2]])
        #file.write('%d %d\n'%(j[0], j[1]))
        file.write( "{} CA {} CA {}\n".format(j[0]+1, j[1]+1,float(j[2])))
        file.write("\n")
#print(accpt)
