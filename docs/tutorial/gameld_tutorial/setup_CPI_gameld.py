#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import meld
from meld import unit as u
from meld.system.scalers import LinearRamp
from meld import parse


# Hamiltonian and temperature Replica Exchange Molecular Dynamics (H,T-REMD).
N_REPLICAS = 9
N_STEPS = 100000
GAMD: bool = True
TIMESTEPS = 2500
CONVENTIONAL_MD_PREP = 100
CONVENTIONAL_MD = 1000
GAMD_EQUILIBRATION_PREP = 900
GAMD_EQUILIBRATION = 9000

hydrophobes = 'AILMFPWV'
hydrophobes_res = ['ALA', 'ILE', 'LEU', 'MET', 'PHE', 'PRO', 'TRP', 'VAL']


def make_ss_groups(subset=None):
    active = 0
    extended = 0
    sse = []
    ss = open('ss.dat', 'r').readlines()[0]  # change file
    for i, l in enumerate(ss.rstrip()):
        if l not in "HE.":
            continue
        if l not in 'E' and extended:
            end = i
            sse.append((start+1, end))
            extended = 0
        if l in 'E':
            if i+1 in subset:
                active = active + 1
            if extended:
                continue
            else:
                start = i
                extended = 1
    return sse, active


def generate_strand_pairs(s, sse, active, subset=np.array([])):
    n_res = s.residue_numbers[-1]
    subset = subset if subset.size else np.array(range(n_res))+1
    strand_pair = []
    for i in range(len(sse)):
        start_i, end_i = sse[i]
        for j in range(i+1, len(sse)):
            start_j, end_j = sse[j]

            for res_i in range(start_i, end_i+1):
                for res_j in range(start_j, end_j+1):
                    if res_i in subset or res_j in subset:
                        g = []
                        make_pairNO(g, s, res_i, res_j)
                        strand_pair.append(
                            s.restraints.create_restraint_group(g, 1))
                        g = []
                        make_pairON(g, s, res_i, res_j)
                        strand_pair.append(
                            s.restraints.create_restraint_group(g, 1))
    all_rest = len(strand_pair)
    active = int(active * 0.65)
    s.restraints.add_selectively_active_collection(strand_pair, active)


def make_pairNO(g, s, i, j):
    scaler = s.restraints.create_scaler(
        'nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    atom1 = s.index.atom(
        i, atom_name="N", expected_resname=None, chainid=None, one_based=False)
    atom2 = s.index.atom(
        j, atom_name="O", expected_resname=None, chainid=None, one_based=False)
    g.append(s.restraints.create_restraint('distance', scaler, r1=0.0 * u.nanometer, r2=0.0 * u.nanometer, r3=0.3 * u.nanometer, r4=0.4 * u.nanometer, k=250.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                                           atom1=atom1, atom2=atom2))


def make_pairON(g, s, i, j):
    scaler = s.restraints.create_scaler(
        'nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)
    atom1 = s.index.atom(
        i, atom_name="O", expected_resname=None, chainid=None, one_based=False)
    atom2 = s.index.atom(
        j, atom_name="N", expected_resname=None, chainid=None, one_based=False)
    g.append(s.restraints.create_restraint('distance', scaler, r1=0.0 * u.nanometer, r2=0.0 * u.nanometer, r3=0.3 * u.nanometer, r4=0.4 * u.nanometer, k=250.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                                           atom1=atom1, atom2=atom2))


def create_hydrophobes(s, ContactsPerHydroph=1.3, scaler=None, group_1=np.array([]), group_2=np.array([])):
    n_res = s.residue_numbers[-1]
    group_1 = group_1 if group_1.size else np.array(range(n_res))+1
    group_2 = group_2 if group_2.size else np.array(range(n_res))+1
    scaler = scaler if scaler else s.restraints.create_scaler(
        'nonlinear', alpha_min=0.4, alpha_max=1.0, factor=4.0)

    sequence = [(i, j) for i, j in zip(s.residue_numbers, s.residue_names)]
    sequence = sorted(set(sequence))
    sequence = dict(sequence)

    group_1 = [res for res in group_1 if (sequence[res] in hydrophobes_res)]
    group_2 = [res for res in group_2 if (sequence[res] in hydrophobes_res)]

    pairs = []
    hydroph_restraints = []
    for i in group_1:
        for j in group_2:

            if ((i, j) in pairs) or ((j, i) in pairs):
                continue

            if (i == j):
                continue

            if (abs(i-j) < 7):
                continue
            pairs.append((i, j))
            atom1 = s.index.atom(
                i, atom_name="CB", expected_resname=None, chainid=None, one_based=False)
            atom2 = s.index.atom(
                j, atom_name="CB", expected_resname=None, chainid=None, one_based=False)
            hydroph_restraints.append(s.restraints.create_restraint('distance', scaler, r1=0.0 * u.nanometer, r2=0.0 * u.nanometer, r3=0.9 * u.nanometer, r4=1.1 * u.nanometer, k=250.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                                                                    atom1=atom1, atom2=atom2))
    all_rest = len(hydroph_restraints)
    active = int(ContactsPerHydroph * len(group_1))
    s.restraints.add_selectively_active_collection(hydroph_restraints, active)


def gen_state(s, index):
    pos = s._coordinates
    pos = pos - np.mean(pos, axis=0)
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0
    return s.SystemState(pos, vel, alpha, energy)


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
            atom1 = s.index.atom(
                i, atom_name=name_i, expected_resname=None, chainid=None, one_based=False)
            atom2 = s.index.atom(
                j, atom_name=name_j, expected_resname=None, chainid=None, one_based=False)
            rest = s.restraints.create_restraint('distance', scaler,
                                                 r1=0.0 * u.nanometer, r2=0.0 * u.nanometer, r3=dist * u.nanometer, r4=(dist+0.2) * u.nanometer, k=250 * u.kilojoule_per_mole / u.nanometer ** 2,
                                                 atom1=atom1, atom2=atom2)
            rest_group.append(rest)
    return dists


def setup_system():
    # load the sequence
    sequence = parse.get_sequence_from_AA1(
        filename='sequence.dat')  # change file
    n_res = len(sequence.split())

    # build the system
    # Initial conformations are fully extended as generated by the tleap (4) sequence command
    p = meld.AmberSubSystemFromSequence(sequence)

    build_options = meld.AmberOptions(
        forcefield="ff14sbside",
        implicit_solvent_model="gbNeck2",
        # use_big_timestep=True, # We use the OpenMM suite of programs (7) with a 2 femtosecond (fs) time step and Langevin dynamics.
        cutoff=1.8*u.nanometer,
        enable_gamd=GAMD,
        boost_type_str="upper-total",
        conventional_md_prep=CONVENTIONAL_MD_PREP * TIMESTEPS,
        conventional_md=CONVENTIONAL_MD * TIMESTEPS,
        gamd_equilibration_prep=GAMD_EQUILIBRATION_PREP * TIMESTEPS,
        gamd_equilibration=GAMD_EQUILIBRATION * TIMESTEPS,
        total_simulation_length=N_STEPS * TIMESTEPS,
        averaging_window_interval=TIMESTEPS,
    )
    b = meld.AmberSystemBuilder(build_options)
    s = b.build_system([p]).finalize()
    s.temperature_scaler = meld.GeometricTemperatureScaler(
        0, 1, 300. * u.kelvin, 450. * u.kelvin)  # The temperature ranges from 300K in the lowest replica to 450K in the highest, increasing geometrically.

    #
    # Secondary Structure
    #
    ss_scaler = s.restraints.create_scaler('constant')
    # torsion_force_constant=2.5 * u.kilojoule_per_mole, distance_force_constant=2.5 * u.kilojoule_per_mole # We obtain secondary structure predictions from PsiPred (2) or Porter (3). We turn these secondary structure predictions into a set of geometric restraints (see (1)).
    ss_rests = parse.get_secondary_structure_restraints(
        filename='ss.dat', system=s, scaler=ss_scaler, ramp=LinearRamp(0, 100, 0, 1))
    n_ss_keep = int(len(ss_rests) * 0.80)  # Since we know from prior study that secondary structure predictions are typically about 80 percent accurate, we set our active-fraction criterion for the secondary structure restraints to 0.8âmeaning that once 80 percent of the secondary structure restraints are satisfied, the rest are ignored.We enforce 70% of restrains
    s.restraints.add_selectively_active_collection(ss_rests, n_ss_keep)

    #
    # Confinement Restraints
    #
    conf_scaler = s.restraints.create_scaler('constant')
    confinement_rests = []
    confinement_dist = (16.9*np.log(s.residue_numbers[-1])-15.8)/28.
    for index in range(n_res):
        atom_index = s.index.atom(
            index, atom_name="CA", expected_resname=None, chainid=None, one_based=False)
        rest = s.restraints.create_restraint('confine', conf_scaler, LinearRamp(
            0, 100, 0, 1), atom_index=atom_index, radius=confinement_dist * u.nanometer, force_const=250.0 * u.kilojoule_per_mole / u.nanometer ** 2)  # index+1 atom_name='CA',
        confinement_rests.append(rest)
    s.restraints.add_as_always_active_list(confinement_rests)

    #
    # Distance Restraints
    #
    # High reliability
    #
    dist_scaler = s.restraints.create_scaler(
        'nonlinear', alpha_min=0.5, alpha_max=1.0, factor=4.0)
    subset = np.array(range(n_res))  # + 1

    #
    # Hydrophobic
    #
    create_hydrophobes(s, scaler=dist_scaler, group_1=subset)

    #
    # Strand Pairing
    #
    sse, active = make_ss_groups(subset=subset)
    generate_strand_pairs(s, sse, active, subset=subset)

    # create the options
    options = meld.RunOptions(timesteps=TIMESTEPS, minimize_steps=20000,
                              enable_gamd=GAMD)

    remd = meld.setup_replica_exchange(s, n_replicas=N_REPLICAS, n_steps=N_STEPS, n_trials=48*48,
                                       adaptation_growth_factor=2.0, adaptation_burn_in=50, adaptation_adapt_every=50)

    meld.setup_data_store(s, options, remd)


setup_system()
