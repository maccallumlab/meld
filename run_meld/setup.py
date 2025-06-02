#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import meld
from meld.remd import ladder, adaptor, leader
import meld.system.montecarlo as mc
from meld import system
from meld.system import patchers
from meld import comm, vault
from meld import parse
from meld import remd
from meld.system import param_sampling
from openmm import unit
from openmm.app import PDBFile
from meld.system.builders.grappa import GrappaOptions, GrappaSystemBuilder

N_REPLICAS = 2
N_STEPS = 2000
BLOCK_SIZE = 50


hydrophobes = 'AILMFPWV'
hydrophobes_res = ['ALA','ILE','LEU','MET','PHE','PRO','TRP','VAL']


def gen_state(s, index):
    state = s.get_state_template()
    state.alpha = index / (N_REPLICAS - 1.0)
    return state

def make_ss_groups(subset=None):
           active = 0
           extended = 0
           sse = []
           ss = open('ss.dat','r').readlines()[0]
           for i,l in enumerate(ss.rstrip()):
               #print i,l
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
           print(active,':number of E residues')
           print(sse,':E residue ranges')
           return sse,active

def create_hydrophobes(s,group_1=np.array([]),group_2=np.array([]),CO=True):
    hy_rest=open('hydrophobe.dat','w')
    atoms = {"ALA":['CA','CB'],
             "VAL":['CA','CB','CG1','CG2'],
             "LEU":['CA','CB','CG','CD1','CD2'],
             "ILE":['CA','CB','CG1','CG2','CD1'],
             "PHE":['CA','CB','CG','CD1','CE1','CZ','CE2','CD2'],
             "TRP":['CA','CB','CG','CD1','NE1','CE2','CZ2','CH2','CZ3','CE3','CD2'],
             "MET":['CA','CB','CG','SD','CE'],
             "PRO":['CD','CG','CB','CA']}
    #Groups should be 1 centered
    n_res = s.residue_numbers[-1]
    print(n_res)
    group_1 = group_1 if group_1.size else np.array(list(range(n_res)))+1
    group_2 = group_2 if group_2.size else np.array(list(range(n_res)))+1
#    scaler = scaler if scaler else s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)

    #Get a list of names and residue numbers, if just use names might skip some residues that are two
 #times in a row
    #make list 1 centered
    sequence = [(i,j) for i,j in zip(s.residue_numbers,s.residue_names)]
    sequence = sorted(set(sequence))
    print(sequence)
    sequence = dict(sequence)

    #print sequence
    #print hydrophobes_res
    #Get list of groups with only residues that are hydrophobs
    print(group_1)
    print(group_2)
    group_1 = [ res for res in group_1 if (sequence[res-1] in hydrophobes_res) ]
    group_2 = [ res for res in group_2 if (sequence[res-1] in hydrophobes_res) ]

    print(group_1)
    print(group_2)
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

            atoms_i = atoms[sequence[i-1]]  #atoms_i = atoms[sequence[i]]
            atoms_j = atoms[sequence[j-1]]  #atoms_j = atoms[sequence[j]]

            local_contact = []
            for a_i in atoms_i:
                for a_j in atoms_j:
                    #if CO:
                        #print i,j,a_i,a_j
                     #   tmp_scaler = scaler(abs(i-j), 'hydrophobic')
                    #else:
                     #   tmp_scaler = scaler
                    hy_rest.write('{} {} {} {}\n'.format(i,a_i, j, a_j))
            hy_rest.write('\n')

def generate_strand_pairs(s,sse,subset=np.array([]),CO=True):
    f=open('strand_pair.dat','w')
    n_res = s.residue_numbers[-1]
    subset = subset if subset.size else np.array(list(range(n_res)))+1
    strand_pair = []
    for i in range(len(sse)):
        start_i,end_i = sse[i]
        for j in range(i+1,len(sse)):
            start_j,end_j = sse[j]

            for res_i in range(start_i,end_i+1):
                for res_j in range(start_j,end_j+1):
                    if res_i in subset or res_j in subset:
                        #print(res_i,res_j)
                        f.write('{} {} {} {}\n'.format(res_i, 'N', res_j, 'O'))
                        #f.write('\n')
                        f.write('{} {} {} {}\n'.format(res_i, 'O', res_j, 'N'))
                        f.write('\n')
                        #g = []
                        #make_pairNO(g,s,res_i,res_j,scaler,CO)
                        #strand_pair.append(s.restraints.create_restraint_group(g,1))
                        #g = []
                        #make_pairON(g,s,res_i,res_j,scaler,CO)
                        #strand_pair.append(s.restraints.create_restraint_group(g,1))
    #all_rest = len(strand_pair)
    #active = int(active * active_per_cent)
    #print(("strand_pairs:", all_rest,active))

def get_dist_restraints_hydrophobe(filename, s, scaler, ramp, seq):
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
            i = int(cols[0])-1
            name_i = cols[1]
            j = int(cols[2])-1
            name_j = cols[3]

            rest = s.restraints.create_restraint('distance', scaler, ramp,
                                                 r1=0.0*u.nanometer, r2=0.0*u.nanometer, r3=0.5*u.nanometer, r4=0.7*u.nanometer,
                                                 k=250*u.kilojoule_per_mole/u.nanometer **2,
                                                 atom1=s.index.atom(i,name_i, expected_resname=seq[i][-3:]),
                                                 atom2=s.index.atom(j,name_j, expected_resname=seq[j][-3:]))
            rest_group.append(rest)
    return dists

def get_dist_restraints_strand_pair(filename, s, scaler, ramp, seq):
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
            i = int(cols[0])-1
            name_i = cols[1]
            j = int(cols[2])-1
            name_j = cols[3]
            #

            rest = s.restraints.create_restraint('distance', scaler, ramp,
                                                 r1=0.0*u.nanometer, r2=0.0*u.nanometer, r3=0.35*u.nanometer, r4=0.55*u.nanometer,
                                                 k=250*u.kilojoule_per_mole/u.nanometer **2,
                                                 atom1=s.index.atom(i,name_i, expected_resname=seq[i][-3:]),
                                                 atom2=s.index.atom(j,name_j, expected_resname=seq[j][-3:]))
            rest_group.append(rest)
    return dists


#######################################

def setup_system():
    
    # load the sequence
    sequence = parse.get_sequence_from_AA1(filename='sequence.dat')
    n_res = len(sequence.split())

    # build the system
    pdb_file = PDBFile('protein_min.pdb')
    topology = pdb_file.topology
    positions = pdb_file.positions

    grappa_options = GrappaOptions(
        grappa_model_tag='latest',
        base_forcefield_files=['amber/ff14SB.xml'],
        default_temperature=300.0 * unit.kelvin,
        cutoff=1.8 * unit.nanometers,
        use_big_timestep=True
    )

    grappa_builder = GrappaSystemBuilder(grappa_options)
    system_spec = grappa_builder.build_system(topology, positions)

    # Initialize a basic System object first
    s = system.System(
        system_spec=system_spec,
        communicator=None, # Will be set later by store.save_communicator
        restraints=None, # Will be populated by add_selectively_active_collection etc.
        director=None, # Not explicitly used in the original script for system creation
        options=None # RunOptions will be set later by store.save_run_options
    )
    s.temperature_scaler = system.temperature.GeometricTemperatureScaler(0, 0.3, 300.*unit.kelvin, 550.*unit.kelvin)

##########################

    ramp = s.restraints.create_scaler('nonlinear_ramp', start_time=1, end_time=200,
                                      start_weight=1e-3, end_weight=1, factor=4.0)
    seq = sequence.split()
    for i in range(len(seq)):
        if seq[i][-3:] =='HIE': seq[i]='HIS'
    print(seq)
    hydrophobic_res_in_protein=[]
    for i in seq:
        for j in hydrophobes_res:
            if i ==j:
                hydrophobic_res_in_protein.append(i)

    no_hy_res=len(hydrophobic_res_in_protein)
    print(no_hy_res,':number of hydrophobic residue')
    #
    # Secondary Structure
    #
    ss_scaler = s.restraints.create_scaler('constant')
    ss_rests = parse.get_secondary_structure_restraints(filename='ss.dat', system=s, scaler=ss_scaler,
            ramp=ramp, torsion_force_constant=0.01*u.kilojoule_per_mole/u.degree **2, distance_force_constant=2.5*u.kilojoule_per_mole/u.nanometer **2, quadratic_cut=2.0*u.nanometer)
    n_ss_keep = int(len(ss_rests) * 0.85)
    s.restraints.add_selectively_active_collection(ss_rests, n_ss_keep)

    conf_scaler = s.restraints.create_scaler('constant')
    confinement_rests = []
    for index in range(n_res):
        rest = s.restraints.create_restraint('confine', conf_scaler, ramp=ramp, atom_index=s.index.atom(index, 'CA', expected_resname=seq[index][-3:]),
                                             radius=4.5*u.nanometer, force_const=250.0*u.kilojoule_per_mole/u.nanometer **2)
        confinement_rests.append(rest)
    s.restraints.add_as_always_active_list(confinement_rests)

    #
    # Setup Scaler
    #
    # Initialize restraints object for the system if it's not already initialized
    if s.restraints is None:
        s.restraints = system.RestraintManager(s)

    scaler = s.restraints.create_scaler('nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    subset1= np.array(list(range(n_res))) + 1
    #


    create_hydrophobes(s,group_1=subset1,group_2=subset1,CO=False)
    
    
    #creates parameter for sampling for hydrophobic contacts
    dists = get_dist_restraints_hydrophobe('hydrophobe.dat', s, scaler, ramp, seq)
    #prior_c13 = param_sampling.ScaledExponentialDiscretePrior(u0=2.0, temperature_scaler=s.temperature_scaler, scaler=scaler)
    #sampler_c13 = param_sampling.DiscreteSampler(int(1), int(1.00 * len(dists)), 1)
    #param_c13 = s.param_sampler.add_discrete_parameter("param_HP", int(1.2 * no_hy_res), prior_c13, sampler_c13)
    #s.restraints.add_selectively_active_collection(dists, param_c13)
    s.restraints.add_selectively_active_collection(dists, int(1.2 * no_hy_res))   

    ##strand pairing
    sse,active = make_ss_groups(subset=subset1)
    generate_strand_pairs(s,sse,subset=subset1,CO=False)
    #
    ##creates parameter sampling for strand pairing
    dists = get_dist_restraints_strand_pair('strand_pair.dat', s, scaler, ramp, seq)
    #prior_n15 = param_sampling.ScaledExponentialDiscretePrior(u0=2.0, temperature_scaler=s.temperature_scaler, scaler=scaler)
    #sampler_n15 = param_sampling.DiscreteSampler(int(1), int(1.00 * len(dists)), 1)

    # Initialize param_sampler object for the system if it's not already initialized
    if not hasattr(s, 'param_sampler') or s.param_sampler is None:
        s.param_sampler = param_sampling.ParameterSampler()

    #param_n15 = s.param_sampler.add_discrete_parameter("param_SP", int(0.45*active), prior_n15, sampler_n15)
    #s.restraints.add_selectively_active_collection(dists, param_n15)
    s.restraints.add_selectively_active_collection(dists, int(0.45*active))    

    # setup mcmc at startup
    movers = []
    n_atoms = s.n_atoms
    for i in range(0, n_res):
        n = s.index.atom(i, 'N', expected_resname=seq[i][-3:])
        ca = s.index.atom(i, 'CA', expected_resname=seq[i][-3:])
        c = s.index.atom(i, 'C', expected_resname=seq[i][-3:])
 
        atom_indxs = list(system.indexing.AtomIndex(j) for j in range(ca,n_atoms))
        mover = mc.DoubleTorsionMover(index1a=n, index1b=ca, atom_indices1=list(system.indexing.AtomIndex(i) for i in range(ca, n_atoms)),
                                      index2a=ca, index2b=c, atom_indices2=list(system.indexing.AtomIndex(j) for j in range(c, n_atoms)))

        movers.append((mover, 1))

    sched = mc.MonteCarloScheduler(movers, n_res * 60)

############################################################
    # create the options
    options = meld.RunOptions(
        timesteps = 14286,
        minimize_steps = 20000,
        min_mc = sched,
        param_mcmc_steps=200
    )


    # create a store
    store = vault.DataStore(gen_state(s,0), N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)  # why i need gen_state(s,0)? doubtful
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=48 * 48)
    policy_1 = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy_1, min_acc_prob=0.02)

    remd_runner = remd.leader.LeaderReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS, timeout=60000)
    store.save_communicator(c)

    # create and save the initial states
    states = [gen_state(s, i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

#################################################
    return s.n_atoms


setup_system()
