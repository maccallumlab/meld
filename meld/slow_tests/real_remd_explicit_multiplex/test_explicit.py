#!/usr/bin/env python
# encoding: utf-8

import numpy as np  # type: ignore
import unittest
from meld.remd import ladder, adaptor, leader
from meld import system
from meld import comm, vault
from meld import parse
from meld.test import helper
import os
import shutil
import subprocess


N_REPLICAS = 2
N_STEPS = 4
BLOCK_SIZE = 2


def gen_state(s, index):
    pos = s._coordinates
    box_vectors = s._box_vectors
    vel = np.zeros_like(pos)
    alpha = index / (N_REPLICAS - 1.0)
    energy = 0
    return system.SystemState(pos, vel, alpha, energy, box_vectors)


def make_group_indices(filename):
    group = []
    with open(filename) as f:
        for line in f:
            index = int(line.strip())
            group.append((index, "CA"))
    return group


def setup_system():
    s = system.builder.load_amber_system("system.top", "system.mdcrd")
    s.temperature_scaler = system.ConstantTemperatureScaler(300.)

    # create the options
    options = system.RunOptions(solvation="explicit")
    options.enable_pme = True
    options.pme_tolerance = 0.0005
    options.enable_pressure_coupling = True
    options.pressure = 1.0
    options.pressure_coupling_update_steps = 25
    options.timesteps = 10
    options.minimize_steps = 50

    # create a store
    store = vault.DataStore(
        s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BLOCK_SIZE
    )
    store.initialize(mode="w")
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=48 * 48)
    a = adaptor.NullAdaptor(N_REPLICAS)

    remd_runner = leader.LeaderReplicaExchangeRunner(
        N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a
    )
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


class FakeRemdTestCase(unittest.TestCase, helper.TempDirHelper):
    def setUp(self):
        self.setUpTempDir()

        # copy over files
        cwd = os.getcwd()
        path = os.path.dirname(os.path.realpath(__file__))
        shutil.copy(os.path.join(path, "system.top"), cwd)
        shutil.copy(os.path.join(path, "system.mdcrd"), cwd)

        self.n_atoms = setup_system()

        # now run it
        subprocess.check_call("launch_remd_multiplex", shell=True)

    def tearDown(self):
        self.tearDownTempDir()

    def test_should_have_correct_results(self):
        # make sure the data directory is there
        self.assertTrue(os.path.exists("Data"))

        # make sure we're backing things up
        self.assertTrue(os.path.exists("Data/Backup/system.dat"))

        # make sure we have the right number of steps
        s = vault.DataStore.load_data_store()
        s.initialize(mode="a")
        pos = s.load_positions(N_STEPS)

        # make sure things have the right shape
        self.assertEqual(pos.shape[0], N_REPLICAS)
        self.assertEqual(pos.shape[1], self.n_atoms)
        self.assertEqual(pos.shape[2], 3)
