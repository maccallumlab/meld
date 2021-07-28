#! /usr/bin/env python

"""
Process distograms, psi, and phi data.

This file screens every histogram with certain range under certain threshold,
for each histograms: if sum(i,i+c) > probability threshold(0.8), it will be selected to construct flat-bottom harmonic potential.
\              /
 \            /
  \          / (linear restraint)
   (        )
    (      )   (harmonic restraint) 
     ------    (zero restraint)
     i    i+c

""" 

import numpy as np
import matplotlib
from matplotlib import pyplot
import sys
import pickle
import argparse

parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.parse_args()


disto = np.load('distogram.npy')
hmin = 2.3
hmax = 22

seq=open('sequence.fa','r')
seq=seq.readlines()
sequence=seq[1][:-1]
L = len(sequence)

(bins, length,length2) = disto.shape
if length2 != L:
  raise Exception("sequence length doesn't match distogram, please check the sequence.fa file.")
#window width for evaluating histograms
#the histograms are already integrated so that adding y values gives the integral under the curve
window =  5.
tight_window = 3.
#threshold distance use: we use a histogram if in window the probability is higher than treshold
threshold = 0.8

#Find spacing between bins: 
#CHECK: the last point in histogram is density beyond cutoff distance, is this included for spacing or should be removed?? bins-1
spacing = (hmax - hmin) /(bins-1)
n_bins = int(window/spacing)
tight_n_bins = int(tight_window/spacing)

pyplot.figure()
f, (ax1, ax2) = pyplot.subplots(1, 2, sharey=True)



def analyze_dihed(histo,n_bins,threshold):
    val = []
    for i in range(len(histo)):
        val.append(np.sum(histo[i:i+n_bins+1]))
    val = np.array(val)
    max_index = np.argsort(val)[-1]
    max_val = val[max_index]
    #print(val)
    if max_val > threshold:
        #print(max_val,max_index)
        return max_index
    else:
        return None
        


def analyze_disto(histo,n_bins,threshold):
    val = []
    for i in range(len(histo)):
        val.append(np.sum(histo[i:i+n_bins+1]))
    val = np.array(val)
    max_index = np.argsort(val)[-1]
    max_val = val[max_index]
    #print(val)
    if max_val > threshold:
        #print(max_val,max_index)
        return max_index
    else:
        return None
        

fo = open('tight_contacts.dat','w')
tight_contacts=[]
for i in range(L):
    for j in range(i+4,L):
        value = analyze_disto(disto[:-1,i,j],tight_n_bins,threshold)
        if value:
            n_i = 'CB'
            n_j = 'CB'
            if sequence[i] is 'G':
                n_i = 'CA'
            if sequence[j] is 'G':
                n_j = 'CA'
            #amber numbering
            fo.write("{} {} {} {} {}\n".format(i+1,n_i,j+1,n_j,hmin + value*spacing))
            fo.write('\n')
            tight_contacts.append([i,j])
            ax2.plot(disto[:,i,j])
        else:
            ax1.plot(disto[:,i,j])

fo.close

fo = open('contacts.dat','w')
 
for i in range(L):      
    for j in range(i+4,L):
        value = analyze_disto(disto[:-1,i,j],n_bins,threshold)
        if value:       
            n_i = 'CB'  
            n_j = 'CB'  
            if sequence[i] is 'G':
                n_i = 'CA'
            if sequence[j] is 'G':
                n_j = 'CA'
            #amber numbering
            if [i,j] not in tight_contacts:
                fo.write("{} {} {} {} {}\n".format(i+1,n_i,j+1,n_j,hmin + value*spacing))
                fo.write('\n')
                ax2.plot(disto[:,i,j])
        else:           
            ax1.plot(disto[:,i,j])
 
fo.close

pyplot.savefig('distograms.png')
        

f0 = open('tight_phi.dat','w')
f1 = open('phi.dat','w')
f2 = open('tight_psi.dat','w')
f3 = open('psi.dat','w')
phi = np.load('phi.npy')
psi = np.load('psi.npy')
hmin = -180
hmax = 180
(dbins, length) = phi.shape
window =50
tight_window = 30
spacing = (hmax - hmin) /dbins
n_bins = int(window/spacing)
tight_n_bins = int(tight_window/spacing)

tight_phi=[]
for i in range(1,L):
        value = analyze_disto(phi[:,i],tight_n_bins,threshold)
        if value:
            #amber numbering
            f0.write("{} {}\n".format(i+1,hmin + value*spacing))
         #   fo.write('\n')
            tight_phi.append(i)
            ax2.plot(disto[:,i])
        else:
            ax1.plot(disto[:,i])
f0.close 
for i in range(1,L):
        value = analyze_disto(phi[:,i],n_bins,threshold)
        if value:
            #amber numbering
            if i not in tight_phi:
                f1.write("{} {}\n".format(i+1,hmin + value*spacing))
         #   fo.write('\n')
                ax2.plot(disto[:,i])
        else:
            ax1.plot(disto[:,i])

f1.close
pyplot.savefig('phi.png')
        
tight_psi=[] 
for i in range(0,L-1):                                                                             
        value = analyze_disto(psi[:,i],tight_n_bins,threshold)
        if value:
            #amber numbering
            f2.write("{} {}\n".format(i+1,hmin + value*spacing,window))
        #    f2.write('\n')
            tight_psi.append(i)
            ax2.plot(psi[:,i])
        else:
            ax1.plot(psi[:,i])
 
f2.close


for i in range(0,L-1):
        value = analyze_disto(psi[:,i],n_bins,threshold)
        if value:
            #amber numbering
            if i not in tight_psi:
                f3.write("{} {}\n".format(i+1,hmin + value*spacing,window))
        #    f2.write('\n')
                ax2.plot(psi[:,i])
        else:
            ax1.plot(psi[:,i])

f3.close
pyplot.savefig('psi.png')
        

