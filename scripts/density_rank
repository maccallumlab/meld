#!/usr/bin/env python     
# encoding: utf-8

#'''
#traj_loc,pdb_loc,traj_start,traj_end,sieve_traj,sieve_res,residues_0,residues_1,residues_2,residues_3
#'''

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import cm
import mdtraj as md
import itertools
import time

def parse_args():                              #in line argument parser with help 
    parser = argparse.ArgumentParser()
    parser.add_argument('-traj',type=str,metavar='path',help='path of trajectory',nargs='+')
    parser.add_argument('-top',type=str,metavar='path',help='path of topology')
    parser.add_argument('-start',metavar='N',type=int,help='start frame',nargs='+')
    parser.add_argument('-end',metavar='N',type=int,help='end frame',nargs='+')
    parser.add_argument('-sieve',metavar='N',type=int,help='skip every N frames',nargs='+')
    parser.add_argument('-inter',metavar='res_0 res_1 skip_0 res_2 res_3 skip_1',type=int,default=0,help='calculate contact in range [res_0:res_1:skip_0]  and [res_2:res_3:skip_1] with inter_cutoff, multiple ranges are allowed, total length should be multiple of 6',nargs='+')
    parser.add_argument('-inter_cutoff',metavar='cutoff',type=float,help='inter_contact cutoff, unit in nm',nargs='+')
    parser.add_argument('-intra',metavar='res_0 res_1 skip',type=int,default=0,help='calculate contact in range [res_0:res_1:skip] with intra_cutoff, multiple ranges are allowed, total length should be multiple of 3',nargs='+')
    parser.add_argument('-intra_cutoff',metavar='cutoff',type=float,help='intra_contact cutoff, unit in nm',nargs='+')
    parser.add_argument('-extract_traj',metavar='density range',type=float,default=0,required=False,help='extract samples with specified density range, default not extracting.',nargs='+')
    return parser.parse_args()
 
args=parse_args()
traj_loc = args.traj
pdb_loc = args.top
traj_start = args.start
traj_end = args.end
sieve_traj = args.sieve
inter = args.inter
inter_cutoff = args.inter_cutoff
intra = args.intra
intra_cutoff = args.intra_cutoff
extract_traj = args.extract_traj

def feature(traj_loc, pdb_loc, traj_start=traj_start, traj_end=traj_end, sieve_traj=sieve_traj, 
            inter_ct=inter, inter_cf = inter_cutoff,intra_ct=intra,intra_cf=intra_cutoff):
    '''
    Contact calculation.
    ---
    Input:
    traj_loc, pdb_loc: trajectory and pdb file location.
    sieve_res: calculate contact fingerprint every "sieve_res" (e.g. every two residue),
    sieve_traj: calculate fingerprints every "sieve_traj" sample.
    threshold: criterion for every "sieve_res" contact, within threshold 1 and 0 otherwise.
    
    Output:
    inp: contact fingerprints for selected samples
    '''
    num_traj=len(traj_loc)
    traj_len=[md.load_dcd(traj_loc[i],top=pdb_loc).n_frames for i in range(num_traj)] 
    for i in range(num_traj):
        try:
            traj_end[i] <= traj_len[i]
        except:
            sys.stderr.write(f'traj {i} range is larger than its actual length')
            sys.exit(1)
    traj = md.load_dcd(traj_loc[0],top=pdb_loc)[traj_start[0]:traj_end[0]:sieve_traj[0]]
    if num_traj>1:
        if len(traj_start) == 1 and len(traj_end) == 1 and len(sieve_traj) == 1:
            for i in range(1,num_traj):
                traj += md.load_dcd(traj_loc[i],top=pdb_loc)[traj_start[0]:traj_end[0]:sieve[0]]
        elif len(traj_start) == num_traj and len(traj_end) == num_traj and len(sieve_traj) == num_traj: 
            for i in range(1,num_traj):
                traj += md.load_dcd(traj_loc[i],top=pdb_loc)[traj_start[i]:traj_end[i]:sieve_traj[i]]
        else:
            sys.stderr.write('The number of traj ranges and sieves should be 1 or equal to the number of trajs')  
            sys.exit(1)
    topfile=traj.top
    residues = np.arange(0,topfile.n_residues)
    pairs=[]
    tmp_inp=[]
    if inter_ct != 0:
        try:
            len(inter_ct) % 6 == 0
        except:
            sys.stderr.write('inter should be divisible by 6')
            sys.exit(1)  
        for inters_index,inters in enumerate([inter_ct[i:i + 6] for i in range(0, len(inter_ct), 6)]):
            tmp_pairs = []
            print(f'select contacts between [{inters[0]}:{inters[1]}:{inters[2]}] and [{inters[3]}:{inters[4]}:{inters[5]}]')
            for i,r1 in enumerate(range(inters[0],inters[1],inters[2])):
                for r2 in range(inters[3],inters[4],inters[5]):
                    tmp_pairs.append([r1,r2])
            tmp_inp.append((md.compute_contacts(traj,tmp_pairs,scheme='closest-heavy',periodic=False)[0]<inter_cf[inters_index]).astype(np.int))
    if intra_ct != 0:
        try:
            len(intra_ct) % 3 == 0
        except:
            sys.stderr.write('intra should be divisible by 3')
            sys.exit(1)  
        for intras_index,intras in enumerate([intra_ct[i:i + 6] for i in range(0, len(intra_ct), 6)]):
            tmp_pairs = []
            print(f'select contacts in range [{intras[0]}:{intras[1]}:{intras[2]}]')
            for i,r1 in enumerate(range(intras[0],intras[1])):
                for r2 in range(intras[0],intras[1])[i+1::intras[2]]:
                    tmp_pairs.append([r1,r2])
            tmp_inp.append((md.compute_contacts(traj,tmp_pairs,scheme='closest-heavy',periodic=False)[0]<intra_cf[intras_index]).astype(np.int))
                                                                                    
    inp = np.hstack(tmp_inp)
    return inp,traj


def binary_simi_matrix(inp,simi_scale='scaled',scale=0,batch_size=500000):
    '''
    Binary similarity matrix calculation.
    ---
    Input:
    inp: sample contact fingerprints with size np.array((n,m)). n is number of samples 
         and m is the length of each fingerprint.
    simi_scale: select which scale index for similarity calculation.
                default: simply add all 1 and all 0 together
    batch_size: calculate simi_matrix in batches if number of samples are too large.
                default=1000000
    
    Output:
    simi_matrix: similarity matrix with size np.array((n,n)).
    '''
    all_start=time.time()
    all_input = list(itertools.combinations(inp, 2))
    batch_size=batch_size
    inp_sliced=[all_input[i*batch_size:(i+1)*batch_size] for i in range(int(len(all_input)/batch_size))]
    if int(len(inp_sliced)) < len(all_input)/batch_size:
        inp_sliced.append(all_input[len(inp_sliced)*batch_size:])
    for i in range(len(inp_sliced)):
        temp_start = time.time()
        temp_c = np.zeros((int(len(inp_sliced[i])),3))
        temp_input = np.array(inp_sliced[i])
        product = temp_input.reshape(-1,2,temp_input.shape[-1]).sum(1)
        for row in range(3):
            temp_c[:,row] = np.sum(product==row,axis=1)
        if i == 0:
            all_c = temp_c
        else:
            all_c = np.concatenate((all_c,temp_c),axis=0)
    all_end = time.time() 
    all_time = all_end - all_start 
    ###calculate similarity
    if simi_scale == 'no_scaled':
        simi = all_c[:,0]+all_c[:,2]
        dis_simi = all_c[:,1]
    elif simi_scale == "Faith":
        all_simi = all_c[:,0]+0.5*all_c[:,2]
        denominate = all_c[:,0]+all_c[:,1]+all_c[:,2]
        simi = all_simi/denominate
    elif simi_scale == 'scaled':
        simi = scale*all_c[:,0]+all_c[:,2]
        dis_simi = all_c[:,1]
    simi_matrix = np.zeros((len(inp),len(inp)))
    dis_simi_matrix = np.zeros((len(inp),len(inp)))
    indices = np.triu_indices(len(inp),k=1)
    indices = (indices[1],indices[0])
    simi_matrix[indices] = simi
    dis_simi_matrix[indices] = dis_simi

    return simi_matrix, dis_simi_matrix, all_time


def gbs_plot(simi_matrix,save=False):
    fliped_simi_matrix = np.fliplr(simi_matrix)
    mirror_binary_simi=simi_matrix+np.rot90(fliped_simi_matrix)
    sum_mirror_binary_simi = np.sum(mirror_binary_simi.T,axis=1)
    sum_mirror_binary_simi_index = np.vstack((sum_mirror_binary_simi,range(len(sum_mirror_binary_simi)))).T
    nsmbs = sum_mirror_binary_simi/sum_mirror_binary_simi_index[:,0].max()
    max_g_simi=np.where(sum_mirror_binary_simi_index[:,0]==sum_mirror_binary_simi_index[:,0].max())
    return nsmbs,mirror_binary_simi,max_g_simi[0][0]#,fig
    
def main():
    if len(sys.argv) <= 1:
      print('no INPUT!')
      sys.exit(1)
    inp,traj = feature(traj_loc, pdb_loc, traj_start=traj_start,traj_end=traj_end, sieve_traj=sieve_traj, \
            inter_ct=inter, inter_cf = inter_cutoff,intra_ct=intra,intra_cf=intra_cutoff)
    print('(traj_len,contact_len): ',inp.shape)
    simi_matrix, _, all_time =  binary_simi_matrix(inp,simi_scale='scaled',scale=0,batch_size=500000) 
    print(f'-----Time: {round(all_time,2)} s-----')
    nsmbs,mirror_binary_simi,max_g_simi=gbs_plot(simi_matrix,save=1)    
    print(f'-----max group simi: {max_g_simi}-----')
    np.save('density',nsmbs)
    plt.figure()
    am=plt.scatter(range(nsmbs.shape[0]),nsmbs,marker='.',c=mirror_binary_simi[int(max_g_simi)],cmap='tab20c')
    plt.scatter(max_g_simi,nsmbs[max_g_simi],marker='*',s=50,color='r',label=f'sample {max_g_simi} with highest density')
    plt.ylabel('normalized density')
    plt.legend()
    cbar=plt.colorbar(am)
    cbar.set_label('contact coincidence w.r.t. star sample')
    plt.savefig('density_rank.png') 
    print('-----save fig-----')
         
    traj[int(max_g_simi)].save_pdb('top_density.pdb')
    if extract_traj == 0:
        sys.exit(0)
    elif len(extract_traj)==2:
        traj[int(traj_start):int(traj_end):int(sieve_traj)][(nsmbs>extract_traj[0])&(nsmbs<extract_traj[1])].save_dcd('top.dcd')
        sys.exit(0) 
                       
if __name__ == '__main__':
    main()                


