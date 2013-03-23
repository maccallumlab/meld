import unittest
import mock
import os
import shutil
import tempfile
import subprocess
import numpy as np
from meld.remd import ladder, adaptor, launch, master_runner
from meld.system import state
from meld import comm, vault
from meld.test import helper


N_ATOMS = 500
N_SPRINGS = 100
N_REPLICAS = 4
N_STEPS = 100
BACKUP_FREQ = 100


def gen_state(index):
    pos = index * np.ones((N_ATOMS, 3))
    vel = index * np.ones((N_ATOMS, 3))
    ss = np.ones(N_SPRINGS)
    se = np.zeros(N_SPRINGS)
    lam = 0
    energy = 0
    return state.SystemState(pos, vel, ss, lam, energy, se)


def setup_system():
    # create a store
    store = vault.DataStore(N_ATOMS, N_SPRINGS, N_REPLICAS, backup_freq=BACKUP_FREQ)
    store.initialize(mode='new')

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=100)
    policy = adaptor.AdaptationPolicy(1.0, 50, 100)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)
    remd_runner = master_runner.MasterReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(N_ATOMS, N_REPLICAS, N_SPRINGS)
    store.save_communicator(c)

    # create and store the fake system
    s = helper.FakeSystem()
    store.save_system(s)

    # create and save the initial states
    states = [gen_state(i) for i in range(N_REPLICAS)]
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()


class FakeRemdTestCase(unittest.TestCase):
    def setUp(self):
        # create and change to temp dir
        self.cwd = os.getcwd()
        self.tmpdir = tempfile.mkdtemp()
        os.chdir(self.tmpdir)

        setup_system()

        # now run it
        subprocess.check_call('mpirun -np 4 launch_remd', shell=True)

    def tearDown(self):
        # switch to original dir and clean up
        os.chdir(self.cwd)
        shutil.rmtree(self.tmpdir)

    def test_data_dir_should_be_present(self):
        self.assertTrue(os.path.exists('Data'))

    def test_files_should_have_been_backed_up(self):
        self.assertTrue(os.path.exists('Data/Backup/results.nc'))

    def test_should_have_correct_number_of_steps(self):
        s = vault.DataStore.load_data_store()
        s.initialize(mode='existing')
        pos = s.load_positions(N_STEPS)

        self.assertEqual(pos.shape[0], N_REPLICAS)
        self.assertEqual(pos.shape[1], N_ATOMS)
        self.assertEqual(pos.shape[2], 3)
