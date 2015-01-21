#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from meld.remd import ladder, adaptor, master_runner
from meld import system
from meld import comm, vault
from meld import parse
from meld.system.restraints import LinearRamp,ConstantRamp


N_REPLICAS = 30
N_STEPS = 10000
BLOCK_SIZE = 100


hydrophobes = 'AILMFPWV'
hydrophobes_res = ['ALA','ILE','LEU','MET','PHE','PRO','TRP','VAL']

def make_ss_groups(subset=None):
           active = 0
           extended = 0
           sse = []
           ss = open('ss.dat','r').readlines()[0]
           for i,l in enumerate(ss.rstrip()):
               print i,l
               if l not in "HE.":
                   continue
               if l not in 'E' and extended:
                   end = i
                   sse.append((start+1,end))
                   extended = 0
               if l in 'E':
                   if i+1 in subset:
                       active = active + 1
                   if extended:
                       continue
                   else:
                       start = i
                       extended = 1
           print active
           return sse,active

def generate_strand_pairs(s,sse,active,subset=np.array([])):
    n_res = s.residue_numbers[-1]
    subset = subset if subset.size else np.array(range(n_res))+1 
    strand_pair = []
    for i in range(len(sse)):
        start_i,end_i = sse[i]
        for j in range(i+1,len(sse)):
            start_j,end_j = sse[j]
            
            for res_i in range(start_i,end_i+1):
                for res_j in range(start_j,end_j+1):
                    if res_i in subset or res_j in subset:
                        print res_i,res_j
                        g = []
                        make_pairNO(g,s,res_i,res_j)
                        strand_pair.append(s.restraints.create_restraint_group(g,1))
                        g = []
                        make_pairON(g,s,res_i,res_j)
                        strand_pair.append(s.restraints.create_restraint_group(g,1))
    all_rest = len(strand_pair)
    active = int(active * 0.65)
    print "strand_pairs:", all_rest,active
    s.restraints.add_selectively_active_collection(strand_pair, active)

def make_pairNO(g,s,i,j):
    scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    g.append(s.restraints.create_restraint('distance', scaler, r1=0.0, r2=0.0, r3=0.3, r4=0.4, k=250.0,
            atom_1_res_index=i, atom_1_name='N', atom_2_res_index=j, atom_2_name='O'))

def make_pairON(g,s,i,j):
    scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    g.append(s.restraints.create_restraint('distance', scaler, r1=0.0, r2=0.0, r3=0.3, r4=0.4, k=250.0,
            atom_1_res_index=i, atom_1_name='O', atom_2_res_index=j, atom_2_name='N'))

def create_hydrophobes(s,ContactsPerHydroph=1.3,scaler=None,group_1=np.array([]),group_2=np.array([])):
    #Groups should be 1 centered
    n_res = s.residue_numbers[-1]
    group_1 = group_1 if group_1.size else np.array(range(n_res))+1
    group_2 = group_2 if group_2.size else np.array(range(n_res))+1
    scaler = scaler if scaler else s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)

    #Get a list of names and residue numbers, if just use names might skip some residues that are two
    #times in a row
    #make list 1 centered
    sequence = [(i,j) for i,j in zip(s.residue_numbers,s.residue_names)]
    sequence = sorted(set(sequence))
    sequence = dict(sequence)

    print sequence
    print hydrophobes_res
    #Get list of groups with only residues that are hydrophobs
    group_1 = [ res for res in group_1 if (sequence[res] in hydrophobes_res) ]
    group_2 = [ res for res in group_2 if (sequence[res] in hydrophobes_res) ]


    pairs = []
    hydroph_restraints = []
    for i in group_1:
        for j in group_2:

            # don't put the same pair in more than once
            if ( (i,j) in pairs ) or ( (j,i) in pairs ):
                continue

            if ( i ==j ):
                continue

            if (abs(i-j)< 7):
                continue
            pairs.append( (i,j) )

            hydroph_restraints.append(s.restraints.create_restraint('distance', scaler, r1=0.0, r2=0.0, r3=0.9, r4=1.1, k=250.0,
            atom_1_res_index=i, atom_1_name='CB', atom_2_res_index=j, atom_2_name='CB'))
            print 'hydroph:',i,j
    all_rest = len(hydroph_restraints)
    active = int( ContactsPerHydroph * len(group_1) )
    print active,len(group_1),all_rest
    s.restraints.add_selectively_active_collection(hydroph_restraints, active)

def gen_state(s, index):
    pos = s._coordinates
    pos = pos - np.mean(pos, axis=0)
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0
    return system.SystemState(pos, vel, alpha, energy)


def get_dist_restraints(filename, s, scaler):
    dists = []
    rest_group = []
    lines = open(filename).read().splitlines()
    lines = [line.strip() for line in lines]
    for line in lines:
        if not line:
            dists.append(s.restraints.create_restraint_group(rest_group, 1))
            rest_group = []
        else:
            cols = line.split()
            i = int(cols[0])
            name_i = cols[1]
            j = int(cols[2])
            name_j = cols[3]
            dist = float(cols[4]) / 10.

            rest = s.restraints.create_restraint('distance', scaler,
                                                 r1=0.0, r2=0.0, r3=dist, r4=dist+0.2, k=250,
                                                 atom_1_res_index=i, atom_2_res_index=j,
                                                 atom_1_name=name_i, atom_2_name=name_j)
            rest_group.append(rest)
    return dists


def setup_system():
    # load the sequence
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    n_res = len(sequence.split())

    # build the system
    p = system.ProteinMoleculeFromSequence(sequence)
    b = system.SystemBuilder()
    s = b.build_system_from_molecules([p])
    s.temperature_scaler = system.GeometricTemperatureScaler(0, 0.4, 300., 550.)

    #
    # Secondary Structure
    #
    ss_scaler = s.restraints.create_scaler('constant')
    ss_rests = parse.get_secondary_structure_restraints(filename='ss.dat', system=s, scaler=ss_scaler,
            torsion_force_constant=2.5, distance_force_constant=2.5)
    n_ss_keep = int(len(ss_rests) * 0.70) #We enforce 70% of restrains 
    s.restraints.add_selectively_active_collection(ss_rests, n_ss_keep)

    #
    # Confinement Restraints
    #
    conf_scaler = s.restraints.create_scaler('constant')
    confinement_rests = []
    confinement_dist = (16.9*np.log(s.residue_numbers[-1])-15.8)/28.
    for index in range(n_res):
        rest = s.restraints.create_restraint('confine', conf_scaler, LinearRamp(0,100,0,1),res_index=index+1, atom_name='CA', radius=confinement_dist, force_const=250.0)
        confinement_rests.append(rest)
    s.restraints.add_as_always_active_list(confinement_rests)

    #
    # Distance Restraints
    #
    # High reliability
    #
    dist_scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    #contact80_dist = get_dist_restraints('target_contacts_over_80.dat', s, dist_scaler)
    #n_high_keep = int(0.80 * len(contact80_dist))
    #s.restraints.add_selectively_active_collection(contact80_dist, n_high_keep)

    #
    # Long
    #
    #contact60_dist = get_dist_restraints('target_contacts_over_60.dat', s, dist_scaler)
    #n_high_keep = int(0.60 * len(contact60_dist))
    #s.restraints.add_selectively_active_collection(contact60_dist, n_high_keep)

    #
    # Heuristic Restraints
    #
    subset= np.array(range(n_res)) + 1

        #
        # Hydrophobic
        #
    create_hydrophobes(s,scaler=dist_scaler,group_1=subset)

        #
        # Strand Pairing
        #
    sse,active = make_ss_groups(subset=subset)
    generate_strand_pairs(s,sse,active,subset=subset)


    # create the options
    options = system.RunOptions()
    options.implicit_solvent_model = 'gbNeck2'
    options.use_big_timestep = True
    options.cutoff = 1.8

    options.use_amap = True
    options.amap_beta_bias = 1.0
    options.timesteps = 14286
    options.minimize_steps = 20000

    # create a store
    store = vault.DataStore(s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=48 * 48)
    policy = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)

    remd_runner = master_runner.MasterReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS)
    store.save_communicator(c)

    # create and save the initial states
    states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

    return s.n_atoms


setup_system()
