#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import comm
from meld import vault
from meld.remd import ladder
from meld.remd import adaptor
from meld.remd import leader
from meld.system import state
from meld.system import options
from meld.system import pdb_writer
from meld.system import temperature
from meld.system import param_sampling
from meld.test import helper

import unittest
import os
import subprocess
import numpy as np  # type: ignore


N_ATOMS = 500
N_DISCRETE = 10
N_CONTINUOUS = 5
N_REPLICAS = 4
N_STEPS = 100
BACKUP_FREQ = 100


def gen_state(index):
    pos = index * np.ones((N_ATOMS, 3))
    vel = index * np.ones((N_ATOMS, 3))
    alpha = 0.0
    energy = 0.0
    box_vectors = np.zeros(3)
    discrete = np.zeros(N_DISCRETE, dtype=np.int32)
    continuous = np.zeros(N_CONTINUOUS, dtype=np.float64)
    params = param_sampling.ParameterState(discrete, continuous)
    return state.SystemState(pos, vel, alpha, energy, box_vectors, params)


def setup_system():
    # create a store
    writer = pdb_writer.PDBWriter(
        range(N_ATOMS), ["CA"] * N_ATOMS, [1] * N_ATOMS, ["ALA"] * N_ATOMS
    )
    store = vault.DataStore(gen_state(0), N_REPLICAS, writer, block_size=BACKUP_FREQ)
    store.initialize(mode="w")

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=100)
    policy = adaptor.AdaptationPolicy(1.0, 50, 100)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)
    remd_runner = leader.LeaderReplicaExchangeRunner(
        N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a
    )
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS)
    store.save_communicator(c)

    # create and store the fake system
    s = helper.FakeSystem()
    s.temperature_scaler = temperature.ConstantTemperatureScaler(300.0)
    store.save_system(s)

    # create and store the options
    o = options.RunOptions()
    o.runner = "fake_runner"
    store.save_run_options(o)

    # create and save the initial states
    states = [gen_state(i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()


class FakeRemdTestCase(unittest.TestCase, helper.TempDirHelper):
    def setUp(self):
        self.setUpTempDir()
        setup_system()
        subprocess.check_call("mpirun -np 4 launch_remd", shell=True)

    def tearDown(self):
        self.tearDownTempDir()

    @unittest.skip("Problems with mpirun in tests.")
    def test_data_dir_should_be_present(self):
        self.assertTrue(os.path.exists("Data"))

    @unittest.skip("Problems with mpirun in tests.")
    def test_files_should_have_been_backed_up(self):
        self.assertTrue(os.path.exists("Data/Backup/system.dat"))

    @unittest.skip("Problems with mpirun in tests.")
    def test_should_have_correct_number_of_steps(self):
        s = vault.DataStore.load_data_store()
        s.initialize(mode="a")
        pos = s.load_positions(N_STEPS)

        self.assertEqual(pos.shape[0], N_REPLICAS)
        self.assertEqual(pos.shape[1], N_ATOMS)
        self.assertEqual(pos.shape[2], 3)
