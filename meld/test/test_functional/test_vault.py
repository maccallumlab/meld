#!/usr/bin/env python

#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import numpy as np  # type: ignore
import unittest
import os
from meld import vault
from meld import comm
from meld.system import options
from meld.system import state
from meld.system import pdb_writer
from meld.system import param_sampling
from meld.system import mapping
from meld.remd import leader
from meld.remd import ladder
from meld.remd import adaptor
from meld.test.helper import TempDirHelper
from meld.util import in_temp_dir


class DataStorePickleTestCase(unittest.TestCase):
    """
    Test that we can read and write the items in the Data directory.
    """

    def setUp(self):
        self.N_ATOMS = 500
        self.N_REPLICAS = 4
        self.N_DISCRETE = 10
        self.N_CONTINUOUS = 5
        self.N_MAPPING = 10
        self.state_template = state.SystemState(
            np.zeros((self.N_ATOMS, 3)),
            np.zeros((self.N_ATOMS, 3)),
            0.0,
            0.0,
            np.zeros(vault.ENERGY_GROUPS),
            np.zeros(3),
            param_sampling.ParameterState(
                np.zeros(self.N_DISCRETE, dtype=np.int32),
                np.zeros(self.N_CONTINUOUS, dtype=np.float64),
            ),
            np.arange(self.N_MAPPING, dtype=int),
        )

    def test_init_mode_w_creates_directories(self):
        "calling initialize should create the Data and Data/Backup directories"
        with in_temp_dir():
            # dummy pdb writer; can't use a mock because they can't be pickled
            pdb_writer = object()
            store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)
            store.initialize(mode="w")

            self.assertTrue(os.path.exists("Data"), "Data directory does not created")
            self.assertTrue(
                os.path.exists("Data/Backup"), "Backup directory not created"
            )

    def test_init_mode_w_creates_results(self):
        "calling initialize should create the results.h5 file"
        with in_temp_dir():
            # dummy pdb writer; can't use a mock because they can't be pickled
            pdb_writer = object()
            store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)
            store.initialize(mode="w")

            self.assertTrue(
                os.path.exists("Data/Blocks/block_000000.nc"),
                "results_000000.nc not created",
            )

    def test_init_mode_w_raises_when_dirs_exist(self):
        "calling initialize should raise RuntimeError when Data directories exist"
        with in_temp_dir():
            os.mkdir("Data")
            os.mkdir("Data/Backup")
            # dummy pdb writer; can't use a mock because they can't be pickled
            pdb_writer = object()
            store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)

            with self.assertRaises(RuntimeError):
                store.initialize(mode="w")

    def test_save_and_load_data_store(self):
        "should be able to save and then reload the DataStore"
        with in_temp_dir():
            # dummy pdb writer; can't use a mock because they can't be pickled
            pdb_writer = object()
            store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)
            store.initialize(mode="w")

            store.save_data_store()
            store2 = vault.DataStore.load_data_store()

            self.assertEqual(store.n_atoms, store2.n_atoms)
            self.assertEqual(store.n_replicas, store2.n_replicas)
            self.assertIsNone(store2._cdf_data_set)
            self.assertTrue(os.path.exists("Data/data_store.dat"))

    def test_save_and_load_communicator(self):
        "should be able to save and reload the communicator"
        with in_temp_dir():
            # dummy pdb writer; can't use a mock because they can't be pickled
            pdb_writer = object()
            store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)
            store.initialize(mode="w")
            c = comm.MPICommunicator(self.N_ATOMS, self.N_REPLICAS)
            # set _mpi_comm to something
            # this should not be saved
            c._mpi_comm = "foo"

            store.save_communicator(c)
            c2 = store.load_communicator()

            self.assertEqual(c.n_atoms, c2.n_atoms)
            self.assertEqual(c.n_replicas, c2.n_replicas)
            self.assertTrue(os.path.exists("Data/communicator.dat"))

    def test_save_and_load_remd_runner(self):
        "should be able to save and reload an remd runner"
        with in_temp_dir():
            # dummy pdb writer; can't use a mock because they can't be pickled
            pdb_writer = object()
            store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)
            store.initialize(mode="w")
            l = ladder.NearestNeighborLadder(n_trials=100)
            policy = adaptor.AdaptationPolicy(1.0, 50, 100)
            a = adaptor.EqualAcceptanceAdaptor(
                n_replicas=self.N_REPLICAS, adaptation_policy=policy
            )
            runner = leader.LeaderReplicaExchangeRunner(
                self.N_REPLICAS, max_steps=100, ladder=l, adaptor=a
            )

            store.save_remd_runner(runner)
            runner2 = store.load_remd_runner()

            self.assertEqual(runner.n_replicas, runner2.n_replicas)
            self.assertTrue(os.path.exists("Data/remd_runner.dat"))

    def test_save_and_load_system(self):
        "should be able to save and load a System"
        with in_temp_dir():
            # dummy pdb writer; can't use a mock because they can't be pickled
            pdb_writer = object()
            store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)
            store.initialize(mode="w")
            fake_system = object()

            store.save_system(fake_system)
            store.load_system()

            self.assertTrue(os.path.exists("Data/system.dat"))

    def test_save_and_load_run_options(self):
        "should be able to save and load run options"
        with in_temp_dir():
            # dummy pdb writer; can't use a mock because they can't be pickled
            pdb_writer = object()
            store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)
            store.initialize(mode="w")
            fake_run_options = options.RunOptions()

            store.save_run_options(fake_run_options)
            store.load_run_options()

            self.assertTrue(os.path.exists("Data/run_options.dat"))


class DataStoreHD5TestCase(unittest.TestCase, TempDirHelper):
    """
    Test that we can read and write the data that goes in the hd5 file.
    """

    def setUp(self):
        self.setUpTempDir()

        # setup data store
        self.N_ATOMS = 500
        self.N_REPLICAS = 16
        self.N_DISCRETE = 10
        self.N_CONTINUOUS = 5
        self.N_MAPPINGS = 10
        self.state_template = state.SystemState(
            np.zeros((self.N_ATOMS, 3)),
            np.zeros((self.N_ATOMS, 3)),
            0.0,
            0.0,
            np.zeros(vault.ENERGY_GROUPS),
            np.zeros(3),
            param_sampling.ParameterState(
                np.zeros(self.N_DISCRETE, dtype=np.int32),
                np.zeros(self.N_CONTINUOUS, dtype=np.float64),
            ),
            np.arange(self.N_MAPPINGS),
        )

        # dummy pdb writer; can't use a mock because they can't be pickled
        pdb_writer = object()
        self.store = vault.DataStore(
            self.state_template, self.N_REPLICAS, pdb_writer, block_size=10
        )
        self.store.initialize(mode="w")

    def tearDown(self):
        self.tearDownTempDir()

    def test_should_raise_stage_is_reduces(self):
        "should raise if we try to write to a previous stage"
        test_pos = np.zeros((self.N_REPLICAS, self.N_ATOMS, 3))
        self.store.save_positions(test_pos, 0)
        self.store.save_positions(test_pos, 1)

        with self.assertRaises(RuntimeError):
            self.store.save_positions(test_pos, 0)

    def test_should_create_second_block(self):
        "should create a second block once the first one fills up"
        test_pos = np.zeros((self.N_REPLICAS, self.N_ATOMS, 3))
        for i in range(11):
            self.store.save_positions(test_pos, i)

        self.assertTrue(os.path.exists("Data/Blocks/block_000000.nc"))
        self.assertTrue(os.path.exists("Data/Blocks/block_000001.nc"))

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
        store2.initialize(mode="a")
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
        store2.initialize(mode="a")
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
        store2.initialize(mode="a")
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
        store2.initialize(mode="a")
        test_energies2 = store2.load_energies(STAGE)

        np.testing.assert_equal(test_energies, test_energies2)

    def test_can_save_and_load_states(self):
        "should be able to save and load states"

        def gen_state(index, n_atoms):
            pos = index * np.ones((n_atoms, 3))
            vel = index * np.ones((n_atoms, 3))
            energy = index
            group_energies = np.zeros(vault.ENERGY_GROUPS)
            lam = index / 100.0
            discrete = np.zeros(self.N_DISCRETE, dtype=np.int32)
            continuous = np.zeros(self.N_CONTINUOUS, dtype=np.float64)
            params = param_sampling.ParameterState(discrete, continuous)
            mappings = np.arange(self.N_MAPPINGS)
            return state.SystemState(
                pos, vel, lam, energy, group_energies, np.zeros(3), params, mappings
            )

        states = [gen_state(i, self.N_ATOMS) for i in range(self.N_REPLICAS)]
        STAGE = 0

        self.store.save_states(states, STAGE)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode="a")
        states2 = store2.load_states(STAGE)

        np.testing.assert_equal(states[-1].positions, states2[-1].positions)

    def test_can_save_and_load_two_states(self):
        "should be able to save and load states"

        def gen_state(index, n_atoms):
            pos = index * np.ones((n_atoms, 3))
            vel = index * np.ones((n_atoms, 3))
            energy = index
            group_energies = np.zeros(vault.ENERGY_GROUPS)
            lam = index / 100.0
            discrete = np.zeros(self.N_DISCRETE, dtype=np.int32)
            continuous = np.zeros(self.N_CONTINUOUS, dtype=np.float64)
            params = param_sampling.ParameterState(discrete, continuous)
            mappings = np.arange(self.N_MAPPINGS)
            return state.SystemState(
                pos, vel, lam, energy, group_energies, np.zeros(3), params, mappings
            )

        states = [gen_state(i, self.N_ATOMS) for i in range(self.N_REPLICAS)]
        STAGE = 0

        self.store.save_states(states, STAGE)
        self.store.save_states(states, STAGE + 1)
        self.store.save_data_store()
        self.store.close()
        store2 = vault.DataStore.load_data_store()
        store2.initialize(mode="a")
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
        store2.initialize(mode="a")
        test_vec2 = store2.load_permutation_vector(STAGE)

        np.testing.assert_equal(test_vec, test_vec2)


class DataStoreBackupTestCase(unittest.TestCase, TempDirHelper):
    """
    Test that backup files are created/copied correctly.
    """

    def setUp(self):
        self.setUpTempDir()

        self.N_ATOMS = 500
        self.N_REPLICAS = 16
        self.N_DISCRETE = 10
        self.N_CONTINUOUS = 5
        self.N_MAPPINGS = 10
        self.state_template = state.SystemState(
            np.zeros((self.N_ATOMS, 3)),
            np.zeros((self.N_ATOMS, 3)),
            0.0,
            0.0,
            np.zeros(vault.ENERGY_GROUPS),
            np.zeros(3),
            param_sampling.ParameterState(
                np.zeros(self.N_DISCRETE, dtype=np.int32),
                np.zeros(self.N_CONTINUOUS, dtype=np.float64),
            ),
            np.arange(self.N_MAPPINGS),
        )

        # setup objects to save to disk
        c = comm.MPICommunicator(self.N_ATOMS, self.N_REPLICAS)

        l = ladder.NearestNeighborLadder(n_trials=100)
        policy = adaptor.AdaptationPolicy(1.0, 50, 100)
        a = adaptor.EqualAcceptanceAdaptor(
            n_replicas=self.N_REPLICAS, adaptation_policy=policy
        )

        # make some states
        def gen_state(index, n_atoms):
            pos = index * np.ones((n_atoms, 3))
            vel = index * np.ones((n_atoms, 3))
            energy = index
            group_energies = np.zeros(vault.ENERGY_GROUPS)
            lam = index / 100.0
            discrete = np.zeros(self.N_DISCRETE, dtype=np.int32)
            continuous = np.zeros(self.N_CONTINUOUS, dtype=np.float64)
            params = param_sampling.ParameterState(discrete, continuous)
            mappings = np.arange(self.N_MAPPINGS)
            return state.SystemState(
                pos, vel, lam, energy, group_energies, np.zeros(3), params, mappings
            )

        states = [gen_state(i, self.N_ATOMS) for i in range(self.N_REPLICAS)]
        runner = leader.LeaderReplicaExchangeRunner(
            self.N_REPLICAS, max_steps=100, ladder=l, adaptor=a
        )

        # dummy pdb writer; can't use a mock because they can't be pickled
        pdb_writer = object()
        self.store = vault.DataStore(self.state_template, self.N_REPLICAS, pdb_writer)
        self.store.initialize(mode="w")

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

        self.assertTrue(os.path.exists("Data/Backup/communicator.dat"))

    def test_backup_copies_store(self):
        "data_store.dat should be backed up"
        self.store.backup(stage=0)

        self.assertTrue(os.path.exists("Data/Backup/data_store.dat"))

    def test_backup_copies_remd_runner(self):
        "remd_runner.dat should be backed up"
        self.store.backup(stage=0)

        self.assertTrue(os.path.exists("Data/Backup/remd_runner.dat"))


class TestReadOnlyMode(unittest.TestCase, TempDirHelper):
    def setUp(self):
        self.setUpTempDir()

        self.N_ATOMS = 500
        self.N_REPLICAS = 16
        self.N_DISCRETE = 10
        self.N_CONTINUOUS = 5
        self.N_MAPPINGS = 10
        self.state_template = state.SystemState(
            np.zeros((self.N_ATOMS, 3)),
            np.zeros((self.N_ATOMS, 3)),
            0.0,
            0.0,
            np.zeros(vault.ENERGY_GROUPS),
            np.zeros(3),
            param_sampling.ParameterState(
                np.zeros(self.N_DISCRETE, dtype=np.int32),
                np.zeros(self.N_CONTINUOUS, dtype=np.float64),
            ),
            np.arange(self.N_MAPPINGS),
        )

        # setup objects to save to disk
        c = comm.MPICommunicator(self.N_ATOMS, self.N_REPLICAS)

        l = ladder.NearestNeighborLadder(n_trials=100)
        policy = adaptor.AdaptationPolicy(1.0, 50, 100)
        a = adaptor.EqualAcceptanceAdaptor(
            n_replicas=self.N_REPLICAS, adaptation_policy=policy
        )

        # make some states
        def gen_state(index, n_atoms):
            pos = index * np.ones((n_atoms, 3))
            vel = index * np.ones((n_atoms, 3))
            energy = index
            group_energies = np.zeros(vault.ENERGY_GROUPS)
            lam = index / 100.0
            discrete = np.zeros(self.N_DISCRETE, dtype=np.int32)
            continuous = np.zeros(self.N_CONTINUOUS, dtype=np.float64)
            params = param_sampling.ParameterState(discrete, continuous)
            mappings = np.arange(self.N_MAPPINGS)
            return state.SystemState(
                pos,
                vel,
                lam,
                energy,
                group_energies,
                np.zeros(3),
                params,
                mappings,
            )

        runner = leader.LeaderReplicaExchangeRunner(
            self.N_REPLICAS, max_steps=100, ladder=l, adaptor=a
        )

        self.pdb_writer = object()
        store = vault.DataStore(
            self.state_template, self.N_REPLICAS, self.pdb_writer, block_size=10
        )
        store.initialize(mode="w")

        # save some stuff
        store.save_communicator(c)
        store.save_remd_runner(runner)
        store.save_system(object())

        for index in range(100):
            states = [gen_state(index, self.N_ATOMS) for i in range(self.N_REPLICAS)]
            store.save_states(states, stage=index)
        store.close()
        store.save_data_store()

        self.store = vault.DataStore.load_data_store()
        self.store.initialize(mode="r")

    def tearDown(self):
        self.tearDownTempDir()

    def test_saving_comm_should_raise(self):
        with self.assertRaises(RuntimeError):
            self.store.save_communicator(object())

    def test_saving_remd_runner_should_raise(self):
        with self.assertRaises(RuntimeError):
            self.store.save_remd_runner(object())

    def test_saving_system_should_raise(self):
        with self.assertRaises(RuntimeError):
            self.store.save_system(object())

    def test_should_load_correct_states(self):
        for i in range(90):
            states = self.store.load_states(stage=i)
            self.assertAlmostEqual(states[0].positions[0, 0], i)

    def test_load_all_positions_should_give_the_correct_positions(self):
        positions = self.store.load_all_positions()
        self.assertEqual(positions.shape[0], self.N_REPLICAS)
        self.assertEqual(positions.shape[1], self.N_ATOMS)
        self.assertEqual(positions.shape[2], 3)
        self.assertEqual(positions.shape[3], 90)
        for i in range(90):
            self.assertAlmostEqual(positions[0, 0, 0, i], i)


class TestPDBWriter(unittest.TestCase):
    def setUp(self):
        self.atom_numbers = [1000, 1001]
        self.atom_names = ["ABCD", "A2"]
        self.residue_numbers = [999, 1000]
        self.residue_names = ["XYZ", "R2"]
        self.coords = np.zeros((2, 3))
        self.coords[0, :] = 0.1
        self.coords[1, :] = 0.2
        self.writer = pdb_writer.PDBWriter(
            self.atom_numbers, self.atom_names, self.residue_numbers, self.residue_names
        )

    def test_should_raise_with_wrong_number_of_atom_names(self):
        with self.assertRaises(AssertionError):
            pdb_writer.PDBWriter(
                self.atom_numbers, ["CA"], self.residue_numbers, self.residue_names
            )

    def test_should_raise_with_wrong_number_of_residue_numbers(self):
        with self.assertRaises(AssertionError):
            pdb_writer.PDBWriter(
                self.atom_numbers, self.atom_names, [1], self.residue_names
            )

    def test_should_raise_with_wrong_number_of_residue_names(self):
        with self.assertRaises(AssertionError):
            pdb_writer.PDBWriter(
                self.atom_numbers, self.atom_names, self.residue_numbers, ["R1"]
            )

    def test_should_raise_with_bad_coordinate_size(self):
        with self.assertRaises(AssertionError):
            self.writer.get_pdb_string(np.zeros((3, 3)), 1)

    def test_output_should_have_six_lines(self):
        result = self.writer.get_pdb_string(self.coords, 1)
        lines = result.splitlines()

        self.assertEqual(len(lines), 6)

    def test_header_should_have_correct_stage(self):
        result = self.writer.get_pdb_string(self.coords, 1)
        lines = result.splitlines()

        self.assertIn("REMARK", lines[0])
        self.assertIn("stage 1", lines[0])

    def test_atom_line_should_have_correct_format(self):
        result = self.writer.get_pdb_string(self.coords, 1)
        lines = result.splitlines()

        result = lines[1]
        expected_result = "ATOM   1000 ABCD XYZ   999       1.000   1.000   1.000"
        self.assertEqual(result, expected_result)

    def test_other_atom_line_should_have_correct_format(self):
        result = self.writer.get_pdb_string(self.coords, 1)
        lines = result.splitlines()

        result = lines[2]
        expected_result = "ATOM   1001  A2   R2  1000       2.000   2.000   2.000"
        self.assertEqual(result, expected_result)


def main():
    unittest.main()


if __name__ == "__main__":
    main()
