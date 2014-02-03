import unittest
import os
import subprocess
import numpy as np
from meld.remd import ladder, adaptor, master_runner
from meld import system
from meld import comm, vault
from meld.test import helper
from meld import parse


N_REPLICAS = 2
N_STEPS = 5
BACKUP_FREQ = 2


def gen_state(s):
    pos = s._coordinates
    vel = np.zeros_like(pos)
    alpha = 0
    energy = 0
    return system.SystemState(pos, vel, alpha, energy)


def setup_system():
    # get the sequence
    sequence = 'AAAAAAAAAAAAAAAA'
    sequence = parse.get_sequence_from_AA1(contents=sequence)
    # create the system
    p = system.ProteinMoleculeFromSequence(sequence)
    b = system.SystemBuilder()
    s = b.build_system_from_molecules([p])
    s.temperature_scaler = system.LinearTemperatureScaler(0, 1, 300, 310)

    rest_scaler = s.restraints.create_scaler('nonlinear', alpha_min=0, alpha_max=1, factor=4.0)
    secondary = 'H'*16
    secondary_restraints = parse.get_secondary_structure_restraints(system=s, scaler=rest_scaler,
                                                                          contents=secondary)
    s.restraints.add_selectively_active_collection(secondary_restraints, len(secondary_restraints))

    # create the options
    options = system.RunOptions()
    options.use_big_timestep = True

    # create a store
    store = vault.DataStore(s.n_atoms, N_REPLICAS, s.get_pdb_writer(), block_size=BACKUP_FREQ)
    store.initialize(mode='w')
    store.save_system(s)
    store.save_run_options(options)

    # create and store the remd_runner
    l = ladder.NearestNeighborLadder(n_trials=1)
    policy = adaptor.AdaptationPolicy(1.0, 50, 100)
    a = adaptor.EqualAcceptanceAdaptor(n_replicas=N_REPLICAS, adaptation_policy=policy)
    remd_runner = master_runner.MasterReplicaExchangeRunner(N_REPLICAS, max_steps=N_STEPS, ladder=l, adaptor=a)
    store.save_remd_runner(remd_runner)

    # create and store the communicator
    c = comm.MPICommunicator(s.n_atoms, N_REPLICAS)
    store.save_communicator(c)

    # create and save the initial states
    states = [gen_state(s) for i in range(N_REPLICAS)]
    states[1].alpha = 1.0
    store.save_states(states, 0)

    # save data_store
    store.save_data_store()

    return s.n_atoms


class FakeRemdTestCase(unittest.TestCase, helper.TempDirHelper):
    def setUp(self):
        self.setUpTempDir()

        self.n_atoms = setup_system()

        # now run it
        subprocess.check_call('launch_remd_multiplex', shell=True)

    def tearDown(self):
        self.tearDownTempDir()

    def test_data_dir_should_be_present(self):
        self.assertTrue(os.path.exists('Data'))

    def test_files_should_have_been_backed_up(self):
        self.assertTrue(os.path.exists('Data/Backup/system.dat'))

    def test_should_have_correct_number_of_steps(self):
        s = vault.DataStore.load_data_store()
        s.initialize(mode='a')
        pos = s.load_positions(N_STEPS)

        self.assertEqual(pos.shape[0], N_REPLICAS)
        self.assertEqual(pos.shape[1], self.n_atoms)
        self.assertEqual(pos.shape[2], 3)
