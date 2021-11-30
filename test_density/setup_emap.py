#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import meld
from meld.comm import MPICommunicator
from meld import comm, vault
import mdtraj as md
from meld import system
from meld.remd import ladder, adaptor, leader
from meld.system.scalers import LinearRamp,ConstantRamp
import scipy.ndimage
from simtk import unit as u
import copy

N_REPLICAS = 8
N_STEPS = 10000
BLOCK_SIZE = 10
def gen_state_templates(index, templates):                                           

    n_templates = len(templates)
    print((index,n_templates,index%n_templates))
    a = system.subsystem.SubSystemFromPdbFile(templates[index%n_templates])
    #Note that it does not matter which forcefield we use here to build
    #as that information is not passed on, it is used for all the same as
    #in the setup part of the script
    b = system.builder.SystemBuilder(forcefield="ff14sbside")
    c = b.build_system([a])
    pos = c._coordinates
    c._box_vectors=np.array([0.,0.,0.])
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0
    return system.state.SystemState(pos, vel, alpha, energy,c._box_vectors)

def gen_state(s, index):
    state = s.get_state_template()
    state.alpha = index / (N_REPLICAS - 1.0)
    return state
def map_potential(emap, threshold, scale_factor):
    emap_cp = copy.deepcopy(emap)
    emap = scale_factor * ((emap - threshold) / (emap.max() - threshold))
    emap_where = np.where(emap <= 0)
    emap_cp = scale_factor * (1 - (emap_cp - threshold) / (emap_cp.max() - threshold))
    emap_cp[emap_where[0], emap_where[1], emap_where[2]] = scale_factor
    return emap_cp
def setup_system():
    template = "1ake_centered.pdb"
    p = meld.SubSystemFromPdbFile(template)
    b = meld.SystemBuilder(forcefield="ff14sbside")
    s = b.build_system([p])
    s.temperature_scaler = meld.ConstantTemperatureScaler(300.0 * u.kelvin)  #
    n_res = s.residue_numbers[-1]
    blur_scaler = s.restraints.create_scaler(
        "linear_blur", alpha_min=0, alpha_max=1, min_blur=0.0, max_blur=0.0
    )
    map_id = s.density.add_density("4ake_t_4A.mrc", blur_scaler, 0.3,0.25)

    map_restraints = []
    for i in range(n_res):
        for atom_name in ["N", "CA", "C", "O"]:
            r = s.restraints.create_restraint(
                "density",
                atom=s.index.atom(resid=i, atom_name=atom_name),
                density_id=map_id,
                strength=1.0 * u.kilojoule_per_mole,
            )
            map_restraints.append(r)
    s.restraints.add_as_always_active_list(map_restraints)
    #s.restraints.add_selectively_active_collection(map_restraints,10)
    dssp=md.compute_dssp(md.load_pdb('1ake_centered.pdb'))
    E=np.where(dssp=='E')
    H=np.where(dssp=='H')                                                                                          
    beta=[]
    alpha=[]
    tmp=[]
    for i in range(dssp.shape[1]):
        if i in E[1] and i+1 in E[1]:
            tmp.append(i)
        else:
            if len(tmp) >=1:
                beta.append(tmp)
            tmp=[]
    tmp=[] 
    for i in range(dssp.shape[1]):
        if i in H[1] and i+1 in H[1]:
            tmp.append(i)
        else:
            if len(tmp) >=1:
                alpha.append(tmp)
            tmp=[]
    HB=np.concatenate((np.concatenate(beta),np.concatenate(alpha))) 
    torsion_rests = []
    dist_scaler = s.restraints.create_scaler('constant')#, alpha_min=0.4, alpha_max=1.0, factor=4.0)
    psi=np.round(md.compute_psi(md.load_pdb('1ake_centered.pdb'))[1][0]*180/np.pi,2)
    for i in range(1,n_res):
        if i in HB:
            psi_avg = float(psi[i-1])
            psi_sd = 15
            res = i
            atoms = [s.index.atom(res, 'N',one_based=True),s.index.atom(res, 'CA',one_based=True),s.index.atom(res, 'C',one_based=True),s.index.atom(res+1, 'N',one_based=True)]
            psi_rest = s.restraints.create_restraint('torsion', dist_scaler,
                                                      phi=psi_avg * u.degree, delta_phi=psi_sd * u.degree, k=0.1*u.kilojoule_per_mole / u.degree ** 2,
                                                      atom1=atoms[0],
                                                      atom2=atoms[1],
                                                      atom3=atoms[2],
                                                      atom4=atoms[3])
            torsion_rests.append(psi_rest)  
    phi=np.round(md.compute_phi(md.load_pdb('1ake_centered.pdb'))[1][0]*180/np.pi,2)
    for i in range(2,n_res+1):            
        if i in HB:
            phi_avg = float(phi[i-2])                
            phi_sd = 15    
            res = i                 
            atoms = [s.index.atom(res-1, 'C',one_based=True),s.index.atom(res, 'N',one_based=True),s.index.atom(res, 'CA',one_based=True),s.index.atom(res, 'C',one_based=True)]
            phi_rest = s.restraints.create_restraint('torsion', dist_scaler,
                                             phi=phi_avg * u.degree, delta_phi=phi_sd * u.degree, k=0.1*u.kilojoule_per_mole / u.degree ** 2,
                                             atom1=atoms[0],
                                             atom2=atoms[1],
                                             atom3=atoms[2],          
                                             atom4=atoms[3])
            torsion_rests.append(phi_rest)  
    s.restraints.add_as_always_active_list(torsion_rests)   
    ##s.restraints.add_selectively_active_collection(torsion_rests,int(len(torsion_rests)*0.8))
    rest_group = []     
    pdbs =md.load_pdb('1ake_centered.pdb')
    cas=pdbs.top.select("name CA")
    for i in range(n_res-3):
      if i in HB and i+1 in HB and i+2 in HB: 
        ca_0=pdbs.top.select(f"resid {i} and name CA")
        ca_1=pdbs.top.select(f"resid {i+2} and name CA")
        dist=float(md.compute_distances(pdbs,np.array([ca_0,ca_1]).T)[0][0])
                       
        r1 = dist -0.1 
        if r1 < 0:     
            r1 = 0.0   
        rest = s.restraints.create_restraint('distance', dist_scaler,LinearRamp(0,100,0,1),
                                              r1=r1*u.nanometer, r2=dist*u.nanometer, r3=(dist+0.1)*u.nanometer, r4=(dist+0.2)*u.nanometer, k=700*u.kilojoule_per_mole / u.nanometer ** 2,
                                              atom1=s.index.atom(i+1, 'CA',one_based=True), atom2=s.index.atom(i+3, 'CA',one_based=True))
        rest_group.append(rest)
    s.restraints.add_as_always_active_list(rest_group)




    # create the options
    options = system.options.RunOptions()
    options.implicit_solvent_model = 'gbNeck2'
    options.use_big_timestep = False
    options.use_bigger_timestep = True
    options.cutoff = 1.8 * u.nanometer
 
    options.use_amap = False
    options.amap_alpha_bias = 1.0
    options.amap_beta_bias = 1.0
    options.timesteps = 222
    options.minimize_steps = 1
    options.min_mc = None
    options.run_mc = None
 
    # create a store
    store = vault.DataStore(gen_state(s,0), N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)
 
    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=100)
    policy = adaptor.AdaptationPolicy(2.0, 50, 50)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)
 
    remd_runner = leader.LeaderReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)
 
    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS)
    store.save_communicator(c)
 
    # create and save the initial states
    states = [gen_state(s, i) for i in range(N_REPLICAS)]                                  
    store.save_states(states, 0)
 
    # save data_store
    store.save_data_store()
 
 
setup_system()
