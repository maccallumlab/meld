#!/usr/bin/env python

import numpy as np
import unittest
import os
from meld import vault, comm
from meld.remd import master_runner, ladder, adaptor
from meld.system import state
from meld.test.helper import TempDirHelper
from meld.util import in_temp_dir


class DataStorePickleTestCase(unittest.TestCase):
    '''
    Test that we can read and write the items that are pickled into the Data directory.
    '''
    def setUp(self):
        self.N_ATOMS = 500
        self.N_REPLICAS = 4

    def test_init_mode_w_creates_directories(self):
        "calling initialize should create the Data and Data/Backup directories"
        with in_temp_dir():
            store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
            store.initialize(mode='new')

            self.assertTrue(os.path.exists('Data'), 'Data directory does not created')
            self.assertTrue(os.path.exists('Data/Backup'), 'Backup directory not created')

    def test_init_mode_w_creates_results(self):
        "calling initialize should create the results.h5 file"
        with in_temp_dir():
            store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
            store.initialize(mode='new')

            self.assertTrue(os.path.exists('Data/results.nc'), 'results.nc not created')

    def test_init_mode_w_raises_when_dirs_exist(self):
        "calling initialize should raise RuntimeError when Data and Data/Backup directories exist"
        with in_temp_dir():
            os.mkdir('Data')
            os.mkdir('Data/Backup')
            store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)

            with self.assertRaises(RuntimeError):
                store.initialize(mode='new')

    def test_save_and_load_data_store(self):
        "should be able to save and then reload the DataStore"
        with in_temp_dir():
            store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
            store.initialize(mode='new')

            store.save_data_store()
            store2 = vault.DataStore.load_data_store()

            self.assertEqual(store.n_atoms, store2.n_atoms)
            self.assertEqual(store.n_replicas, store2.n_replicas)
            self.assertIsNone(store2._cdf_data_set)
            self.assertTrue(os.path.exists('Data/data_store.dat'))

    def test_save_and_load_communicator(self):
        "should be able to save and reload the communicator"
        with in_temp_dir():
            store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
            store.initialize(mode='new')
            c = comm.MPICommunicator(self.N_ATOMS, self.N_REPLICAS)
            # set _mpi_comm to something
            # this should not be saved
            c._mpi_comm = 'foo'

            store.save_communicator(c)
            c2 = store.load_communicator()

            self.assertEqual(c.n_atoms, c2.n_atoms)
            self.assertEqual(c.n_replicas, c2.n_replicas)
            self.assertIsNone(c2._mpi_comm, '_mpi_comm should not be saved')
            self.assertTrue(os.path.exists('Data/communicator.dat'))

    def test_save_and_load_remd_runner(self):
        "should be able to save and reload an remd runner"
        with in_temp_dir():
            store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
            store.initialize(mode='new')
            l = ladder.NearestNeighborLadder(n_trials=100)
            policy = adaptor.AdaptationPolicy(1.0, 50, 100)
            a = adaptor.EqualAcceptanceAdaptor(n_replicas=self.N_REPLICAS, adaptation_policy=policy)
            runner = master_runner.MasterReplicaExchangeRunner(self.N_REPLICAS, max_steps=100, ladder=l, adaptor=a)

            store.save_remd_runner(runner)
            runner2 = store.load_remd_runner()

            self.assertEqual(runner.n_replicas, runner2.n_replicas)
            self.assertTrue(os.path.exists('Data/remd_runner.dat'))

    def test_save_and_load_system(self):
        "should be able to save and load a System"
        with in_temp_dir():
            store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
            store.initialize(mode='new')
            fake_system = object()

            store.save_system(fake_system)
            store.load_system()

            self.assertTrue(os.path.exists('Data/system.dat'))

    def test_save_and_load_run_options(self):
        "should be able to save and load run options"
        with in_temp_dir():
            store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
            store.initialize(mode='new')
            fake_run_options = object()

            store.save_run_options(fake_run_options)
            store.load_run_options()

            self.assertTrue(os.path.exists('Data/run_options.dat'))


class DataStoreHD5TestCase(unittest.TestCase, TempDirHelper):
    '''
    Test that we can read and write the data that goes in the hd5 file.
    '''
    def setUp(self):
        self.setUpTempDir()

        # setup data store
        self.N_ATOMS = 500
        self.N_REPLICAS = 16
        self.store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
        self.store.initialize(mode='new')

    def tearDown(self):
        self.tearDownTempDir()

    def test_can_save_and_load_positions(self):
        "should be able to save and load positions"
        test_pos = np.zeros((self.N_REPLICAS, self.N_ATOMS, 3))
        for i in range(self.N_REPLICAS):
            test_pos[i, :, :] = i

        STAGE = 0
        self.store.save_positions(test_pos, STAGE)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode='existing')
        test_pos2 = store2.load_positions(STAGE)

        np.testing.assert_equal(test_pos, test_pos2)

    def test_can_save_and_load_velocities(self):
        "should be able to save and load velocities"
        test_vel = np.zeros((self.N_REPLICAS, self.N_ATOMS, 3))
        for i in range(self.N_REPLICAS):
            test_vel[i, :, :] = i

        STAGE = 0
        self.store.save_velocities(test_vel, STAGE)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode='existing')
        test_vel2 = store2.load_velocities(STAGE)

        np.testing.assert_equal(test_vel, test_vel2)

    def test_can_save_and_load_alphas(self):
        "should be able to save and load lambdas"
        test_lambdas = np.zeros(self.N_REPLICAS)
        for i in range(self.N_REPLICAS):
            test_lambdas[i] = i / (self.N_REPLICAS - 1)

        STAGE = 0
        self.store.save_alphas(test_lambdas, STAGE)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode='existing')
        test_lambdas2 = store2.load_alphas(STAGE)

        np.testing.assert_equal(test_lambdas, test_lambdas2)

    def test_can_save_and_load_energies(self):
        "should be able to save and load energies"
        test_energies = np.zeros(self.N_REPLICAS)
        for i in range(self.N_REPLICAS):
            test_energies[i] = i

        STAGE = 0
        self.store.save_energies(test_energies, STAGE)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode='existing')
        test_energies2 = store2.load_energies(STAGE)

        np.testing.assert_equal(test_energies, test_energies2)

    def test_can_save_and_load_states(self):
        "should be able to save and load states"
        def gen_state(index, n_atoms):
            pos = index * np.ones((n_atoms, 3))
            vel = index * np.ones((n_atoms, 3))
            energy = index
            lam = index / 100.
            return state.SystemState(pos, vel, lam, energy)

        states = [gen_state(i, self.N_ATOMS) for i in range(self.N_REPLICAS)]
        STAGE = 0

        self.store.save_states(states, STAGE)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode='existing')
        states2 = store2.load_states(STAGE)

        np.testing.assert_equal(states[-1].positions, states2[-1].positions)

    def test_can_save_and_load_two_states(self):
        "should be able to save and load states"
        def gen_state(index, n_atoms):
            pos = index * np.ones((n_atoms, 3))
            vel = index * np.ones((n_atoms, 3))
            energy = index
            lam = index / 100.
            return state.SystemState(pos, vel, lam, energy)

        states = [gen_state(i, self.N_ATOMS) for i in range(self.N_REPLICAS)]
        STAGE = 0

        self.store.save_states(states, STAGE)
        self.store.save_states(states, STAGE + 1)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode='existing')
        states2 = store2.load_states(STAGE)

        np.testing.assert_equal(states[-1].positions, states2[-1].positions)

    def test_can_save_and_load_permutation_vector(self):
        "should be able to save and load permutation vector"
        test_vec = np.array(range(self.N_REPLICAS))
        STAGE = 0

        self.store.save_permutation_vector(test_vec, STAGE)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode='existing')
        test_vec2 = store2.load_permutation_vector(STAGE)

        np.testing.assert_equal(test_vec, test_vec2)


class DataStoreBackupTestCase(unittest.TestCase, TempDirHelper):
    '''
    Test that backup files are created/copied correctly.
    '''
    def setUp(self):
        self.setUpTempDir()

        self.N_ATOMS = 500
        self.N_REPLICAS = 16

        # setup objects to save to disk
        c = comm.MPICommunicator(self.N_ATOMS, self.N_REPLICAS)

        l = ladder.NearestNeighborLadder(n_trials=100)
        policy = adaptor.AdaptationPolicy(1.0, 50, 100)
        a = adaptor.EqualAcceptanceAdaptor(n_replicas=self.N_REPLICAS, adaptation_policy=policy)

        # make some states
        def gen_state(index, n_atoms):
            pos = index * np.ones((n_atoms, 3))
            vel = index * np.ones((n_atoms, 3))
            energy = index
            lam = index / 100.
            return state.SystemState(pos, vel, lam, energy)

        states = [gen_state(i, self.N_ATOMS) for i in range(self.N_REPLICAS)]
        runner = master_runner.MasterReplicaExchangeRunner(self.N_REPLICAS, max_steps=100, ladder=l, adaptor=a)

        self.store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS)
        self.store.initialize(mode='new')

        # save some stuff
        self.store.save_data_store()
        self.store.save_communicator(c)
        self.store.save_remd_runner(runner)
        self.store.save_states(states, stage=0)

    def tearDown(self):
        self.tearDownTempDir()

    def test_backup_copies_comm(self):
        "communicator.dat should be backed up"
        self.store.backup(stage=0)

        self.assertTrue(os.path.exists('Data/Backup/communicator.dat'))

    def test_backup_copies_store(self):
        "data_store.dat should be backed up"
        self.store.backup(stage=0)

        self.assertTrue(os.path.exists('Data/Backup/data_store.dat'))

    def test_backup_copies_remd_runner(self):
        "remd_runner.dat should be backed up"
        self.store.backup(stage=0)

        self.assertTrue(os.path.exists('Data/Backup/remd_runner.dat'))

    def test_backup_copies_h5(self):
        "results.h5 should be backed up"
        self.store.backup(stage=0)

        self.assertTrue(os.path.exists('Data/Backup/results.nc'))
        # make sure we can still access the hd5 file after backup
        self.store.load_states(stage=0)


class TestSafeMode(unittest.TestCase, TempDirHelper):
    def setUp(self):
        self.setUpTempDir()

        self.N_ATOMS = 500
        self.N_REPLICAS = 16

        # setup objects to save to disk
        c = comm.MPICommunicator(self.N_ATOMS, self.N_REPLICAS)

        l = ladder.NearestNeighborLadder(n_trials=100)
        policy = adaptor.AdaptationPolicy(1.0, 50, 100)
        a = adaptor.EqualAcceptanceAdaptor(n_replicas=self.N_REPLICAS, adaptation_policy=policy)

        # make some states
        def gen_state(index, n_atoms):
            pos = index * np.ones((n_atoms, 3))
            vel = index * np.ones((n_atoms, 3))
            energy = index
            lam = index / 100.
            return state.SystemState(pos, vel, lam, energy)

        states_0 = [gen_state(0, self.N_ATOMS) for i in range(self.N_REPLICAS)]
        states_1 = [gen_state(0, self.N_ATOMS) for i in range(self.N_REPLICAS)]
        runner = master_runner.MasterReplicaExchangeRunner(self.N_REPLICAS, max_steps=100, ladder=l, adaptor=a)

        store = vault.DataStore(self.N_ATOMS, self.N_REPLICAS, backup_freq=1)
        store.initialize(mode='new')

        # save some stuff
        store.save_data_store()
        store.save_communicator(c)
        store.save_remd_runner(runner)
        store.save_system(object())
        store.save_states(states_0, stage=0)
        store.backup(0)
        store.save_states(states_1, stage=0)
        store.close()

        self.store = vault.DataStore.load_data_store()
        self.store.initialize(mode='safe')

    def tearDown(self):
        self.tearDownTempDir()

    def test_should_fail_to_load_comm_with_no_backup(self):
        # remove the communicator
        os.remove(self.store.communicator_backup_path)
        with self.assertRaises(IOError):
            self.store.load_communicator()

    def test_saving_comm_should_raise(self):
        with self.assertRaises(RuntimeError):
            self.store.save_communicator(object())

    def test_should_fail_to_load_remd_runner_with_no_backup(self):
        # remove the remd_runner
        os.remove(self.store.remd_runner_backup_path)
        with self.assertRaises(IOError):
            self.store.load_remd_runner()

    def test_saving_remd_runner_should_raise(self):
        with self.assertRaises(RuntimeError):
            self.store.save_remd_runner(object())

    def test_should_fail_to_load_system_with_no_backup(self):
        # remove the system
        os.remove(self.store.system_backup_path)
        with self.assertRaises(IOError):
            self.store.load_system()

    def test_saving_system_should_raise(self):
        with self.assertRaises(RuntimeError):
            self.store.save_system(object())

    def test_should_fail_to_initialize_when_no_backup(self):
        self.store.close()
        os.remove(self.store.net_cdf_backup_path)
        s = vault.DataStore.load_data_store()
        with self.assertRaises(RuntimeError):
            s.initialize(mode='safe')

    def test_saving_states_should_raise(self):
        states = self.store.load_states(stage=0)
        with self.assertRaises(RuntimeError):
            self.store.save_states(states, stage=2)

    def test_should_load_correct_states(self):
        # the backup was done after states_0, but before states_1
        # so, we should be able to load the states for stage=0
        states = self.store.load_states(stage=0)
        # and the positions should be all zeros
        self.assertEqual(states[0].positions[0, 0], 0)
        # but we shouldn't be able to load stage=1
        with self.assertRaises(IndexError):
            self.store.load_states(stage=1)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
