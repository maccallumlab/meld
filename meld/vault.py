#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module for MELD input/output

The main class is :class:`DataStore` which handles all IO for a MELD run.
"""

import contextlib
import os
import pickle
import shutil
import time
from typing import Iterator, Optional, Sequence

import netCDF4 as cdf  # type: ignore
import numpy as np  # type: ignore

from meld import interfaces
from meld.system import options, param_sampling, pdb_writer, state

try:
    from gamd.stage_integrator import GamdStageIntegrator  #type: ignore
    has_gamd = True
except ImportError:
    has_gamd = False

ENERGY_GROUPS = 7


def _load_pickle(data):
    """Read in pickle file.

    Work around incompatibility between python 2 and 3 when
    reading in pickle files saved with python 2.
    """
    try:
        return pickle.load(data)
    except UnicodeDecodeError:
        return pickle.load(data, encoding="latin1")


class DataStore:
    """
    Load / store data for MELD runs.

    Data will be stored in the 'Data' subdirectory. Backups will be stored
    in 'Data/Backup'.

    Some information is stored as python pickled files:

    - Data/data_store.dat -- the :class:`DataStore` object
    - Data/communicator.dat -- the :class:`MPICommunicator` object
    - Data/remd_runner.dat -- the :class:`LeaderReplicaExchangeRunner` object

    Other data (positions, velocities, etc) is stored in files in the Data/Blocks
    diretory.
    """

    #
    # data paths
    #
    _data_dir: str = "Data"
    _backup_dir: str = os.path.join(_data_dir, "Backup")
    _blocks_dir: str = os.path.join(_data_dir, "Blocks")
    log_dir: str = "Logs"
    "Sub-directory for log files"

    _data_store_filename: str = "data_store.dat"
    _data_store_path: str = os.path.join(_data_dir, _data_store_filename)
    _data_store_backup_path: str = os.path.join(_backup_dir, _data_store_filename)

    _communicator_filename: str = "communicator.dat"
    _communicator_path: str = os.path.join(_data_dir, _communicator_filename)
    _communicator_backup_path: str = os.path.join(_backup_dir, _communicator_filename)

    _remd_runner_filename: str = "remd_runner.dat"
    _remd_runner_path: str = os.path.join(_data_dir, _remd_runner_filename)
    _remd_runner_backup_path: str = os.path.join(_backup_dir, _remd_runner_filename)

    _system_filename: str = "system.dat"
    _system_path: str = os.path.join(_data_dir, _system_filename)
    _system_backup_path: str = os.path.join(_backup_dir, _system_filename)

    _run_options_filename: str = "run_options.dat"
    _run_options_path: str = os.path.join(_data_dir, _run_options_filename)
    _run_options_backup_path: str = os.path.join(_backup_dir, _run_options_filename)

    _net_cdf_filename_template: str = "block_{:06d}.nc"
    _net_cdf_path_template: str = os.path.join(_blocks_dir, _net_cdf_filename_template)

    _traj_filename: str = "trajectory.pdb"
    _traj_path: str = os.path.join(_data_dir, _traj_filename)

    _cdf_data_set: Optional[cdf.Dataset] = None

    _integrator_filename: str = "integrator.dat"
    _integrator_path: str = os.path.join(_data_dir, _integrator_filename)
    _integrator_backup_path: str = os.path.join(_backup_dir, _integrator_filename)

    def __init__(
        self,
        state_template: interfaces.IState,
        n_replicas: int,
        pdb_writer: pdb_writer.PDBWriter,
        block_size: int = 100,
    ):
        """
        Initialize the DataStore

        Args:
            n_atoms: number of atoms
            n_replicas: number of replicas
            pdb_writer: the object to handle writing pdb files
            block_size: size of netcdf blocks and frequency to do backups
        """
        self._n_atoms = state_template.positions.shape[0]
        self._n_discrete_parameters = state_template.parameters.discrete.shape[0]
        self._n_continuous_parameters = state_template.parameters.continuous.shape[0]
        self._n_mappings = state_template.mappings.shape[0]
        self._n_alignments = state_template.rdc_alignments.shape[0]
        self._n_replicas = n_replicas
        self._block_size = block_size
        self._cdf_data_set = None
        self._readonly_mode = False
        self._pdb_writer = pdb_writer
        self._current_stage = 0
        self._current_block = 0
        self._max_safe_block = -1
        self._readonly_mode = False

    def __getstate__(self):
        # don't save some fields to disk
        excluded = ["_cdf_data_set"]
        return dict((k, v) for (k, v) in self.__dict__.items() if k not in excluded)

    def __setstate__(self, state):
        # set _cdf_data_set to None
        self.__dict__ = state
        self._cdf_data_set = None

    def __del__(self):
        # close the _cdf_data_set when we go out of scope
        if hasattr(self, "_cdf_data_set"):
            if self._cdf_data_set:
                self._cdf_data_set.close()

    #
    # properties
    #
    @property
    def n_replicas(self) -> int:
        "The number of replicas"
        return self._n_replicas

    @property
    def n_atoms(self) -> int:
        "The number of atoms"
        return self._n_atoms

    #
    # public methods
    #
    def initialize(self, mode: str):
        """
        Prepare to use the DataStore object.

        Args:
            mode: mode to open with

        Available modes are:

        - 'w' -- create a new directory structure and initialize the hd5 file
        - 'a' -- append to the existing files
        - 'r' -- open the file in read-only mode
        """
        if mode == "w":
            if os.path.exists(self._data_dir):
                raise RuntimeError("Data directory already exists")
            else:
                os.mkdir(self._data_dir)
                os.mkdir(self._blocks_dir)
                os.mkdir(self._backup_dir)
            if os.path.exists(self.log_dir):
                raise RuntimeError("Logs directory already exists")
            else:
                os.mkdir(self.log_dir)
            self._current_block = 0
            self._current_stage = 0
            self._create_cdf_file()
        elif mode == "a":
            block_path = self._net_cdf_path_template.format(self._current_block)
            if os.path.exists(block_path):
                self._cdf_data_set = cdf.Dataset(block_path, "a")
            else:
                self._create_cdf_file()
        elif mode == "r":
            self._current_block = 0
            self._readonly_mode = True
            self._load_cdf_file_readonly()
        else:
            raise RuntimeError(f"Unknown value for mode={mode}")

    def close(self):
        """Close the DataStore"""
        if self._cdf_data_set:
            self._cdf_data_set.close()
            self._cdf_data_set = None

    def save_data_store(self):
        """Save this object to disk."""
        with open(self._data_store_path, "wb") as store_file:
            pickle.dump(self, store_file)

    @classmethod
    def load_data_store(cls, load_backup: bool = False):
        """
        Load the DataStore object from disk.

        Args:
            load_backup: whether to load the backup
        """
        path = cls._data_store_backup_path if load_backup else cls._data_store_path
        with open(path, "rb") as store_file:
            return _load_pickle(store_file)

    def save_communicator(self, comm: interfaces.ICommunicator):
        """Save the communicator to disk"""
        self._can_save()
        with open(self._communicator_path, "wb") as comm_file:
            pickle.dump(comm, comm_file)

    def load_communicator(self) -> interfaces.ICommunicator:
        """Load the communicator from disk"""
        if self._readonly_mode:
            path = self._communicator_backup_path
        else:
            path = self._communicator_path
        with open(path, "rb") as comm_file:
            return _load_pickle(comm_file)

    def save_positions(self, positions: np.ndarray, stage: int):
        """
        Save the positions to disk.

        Args:
            positions: n_replicas x n_atoms x 3 array
            stage: stage to store
        """
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        self._cdf_data_set.variables["positions"][..., stage] = positions

    def load_positions(self, stage: int) -> np.ndarray:
        """
        Load positions from disk.

        Args:
            stage: stage to load

        Returns:
            n_replicas x n_atoms x 3 array

        .. warning::
           :meth:`load_positions` can only access moving forward in time.
           Attempts to move backwards in time will raise an error.
        """
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        return self._cdf_data_set.variables["positions"][..., stage]

    def load_positions_random_access(self, stage: int) -> np.ndarray:
        """
        Load positions from disk.

        Args:
            stage: stage to load

        Returns:
            n_replicas x n_atoms x 3 array

        Note:
           This differs from :meth:`load_positions` in that you can positions
           from any stage, while :meth:`load_positions` can only move forward
           in time. However, this comes at a performance penalty.
        """
        # get the block for this stage
        block = self._block_for_stage(stage)

        # if it's the current block, then just return the positions
        if block == self._current_block:
            assert self._cdf_data_set is not None
            return self._cdf_data_set.variables["positions"][..., stage]

        # otherwise open the file, grab the positions, and then close it
        else:
            path = self._net_cdf_path_template.format(block)
            with contextlib.closing(cdf.Dataset(path, "r")) as dataset:
                return dataset.variables["positions"][..., stage]

    def load_all_positions(self) -> np.ndarray:
        """
        Load all positions from disk.

        Returns:
            n_steps x n_replicas x n_atoms x 3 array

        .. warning::
            This could use a lot of memory.
        """
        return np.concatenate(
            [
                np.array(self.load_positions(i))[..., np.newaxis]
                for i in range(self.max_safe_frame)
            ],
            axis=-1,
        )

    def iterate_positions(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Iterator[np.ndarray]:
        """
        Iterate the positions over time.

        Args:
            start: starting step
            end: ending step

        Returns:
            An iterator over steps of n_replicas x n_atoms x 3 array
        """
        if start is None:
            start = 0
        if end is None:
            end = self.max_safe_frame

        for i in range(start, end):
            yield self.load_positions(i)

    def save_velocities(self, velocities: np.ndarray, stage: int):
        """
        Save velocities to disk.

        Args:
            velocities: n_replicas x n_atoms x 3 array
            stage: stage to store

        """
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        self._cdf_data_set.variables["velocities"][..., stage] = velocities

    def load_velocities(self, stage: int) -> np.ndarray:
        """
        Load velocities from disk.

        Args:
            stage: stage to load

        Returns:
            n_replicas x n_atoms x 3 array

        """
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        return self._cdf_data_set.variables["velocities"][..., stage]

    def load_all_velocities(self) -> np.ndarray:
        """
        Load all velocities from disk.

        Returns:
            n_steps x n_replicas x n_atoms x 3 array

        .. warning::
           This could use a lot of memory.
        """
        return np.concatenate(
            [
                np.array(self.load_velocities(i))[..., np.newaxis]
                for i in range(self.max_safe_frame)
            ],
            axis=-1,
        )

    def save_box_vectors(self, box_vectors: np.ndarray, stage: int):
        """
        Save the box_vectors to disk.

        Args:
            positions: n_replicas x 3 x 3 array
            stage: stage to store
        """
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        self._cdf_data_set.variables["box_vectors"][..., stage] = box_vectors

    def load_box_vectors(self, stage: int) -> np.ndarray:
        """
        Load box_vectors from disk.

        Args:
            stage: stage to load

        Returns:
            n_replicas x 3 x 3 array
        """
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        return self._cdf_data_set.variables["box_vectors"][..., stage]

    def load_all_box_vectors(self) -> np.ndarray:
        """
        Load all box_vectors from disk.

        Returns:
            n_steps x n_replicas x 3 x 3 array

        .. warning::
           This could use a lot of memory.

        """
        return np.concatenate(
            [
                np.array(self.load_box_vectors(i))[..., np.newaxis]
                for i in range(self.max_safe_frame)
            ],
            axis=-1,
        )

    def iterate_box_vectors(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Iterator[np.ndarray]:
        """
        Iterate over the box_vectors from disk.

        Args:
            start: starting frame
            end: ending frame

        Returns:
            iterator over n_replicas x 3 x 3 array
        """
        if start is None:
            start = 0
        if end is None:
            end = self.max_safe_frame

        for i in range(start, end):
            yield self.load_box_vectors(i)

    def save_states(self, states: Sequence[interfaces.IState], stage: int):
        """
        Save states to disk.

        Args:
            states: states to store
            stage: stage to store
        """
        self._can_save()
        self._handle_save_stage(stage)
        positions = np.array([s.positions for s in states])
        velocities = np.array([s.velocities for s in states])
        alphas = np.array([s.alpha for s in states])
        energies = np.array([s.energy for s in states])
        group_energies = np.array([s.group_energies for s in states])
        box_vectors = np.array([s.box_vector for s in states])
        discrete_parameters = np.array(
            [s.parameters.discrete for s in states], dtype=np.int32
        )
        continuous_parameters = np.array(
            [s.parameters.continuous for s in states], dtype=np.float64
        )
        mappings = np.array([s.mappings for s in states], dtype=int)
        alignments = np.array([s.rdc_alignments for s in states], dtype=np.float64)

        self.save_positions(positions, stage)
        self.save_velocities(velocities, stage)
        self.save_box_vectors(box_vectors, stage)
        self.save_alphas(alphas, stage)
        self.save_energies(energies, stage)
        self.save_group_energies(group_energies, stage)
        self.save_discrete_parameters(discrete_parameters, stage)
        self.save_continuous_parameters(continuous_parameters, stage)
        self.save_mappings(mappings, stage)
        self.save_alignments(alignments, stage)

    def load_states(self, stage: int) -> Sequence[interfaces.IState]:
        """
        Load states from disk

        Args:
            stage: stage to load

        Returns:
            list of states
        """
        self._handle_load_stage(stage)
        positions = self.load_positions(stage)
        velocities = self.load_velocities(stage)
        box_vectors = self.load_box_vectors(stage)
        alphas = self.load_alphas(stage)
        energies = self.load_energies(stage)
        group_energies = self.load_group_energies(stage)
        discrete_parameters = self.load_discrete_parameters(stage)
        continuous_parameters = self.load_continuous_parameters(stage)
        mappings = self.load_mappings(stage)
        alignments = self.load_alignments(stage)

        states = []
        for i in range(self._n_replicas):
            s = state.SystemState(
                positions[i],
                velocities[i],
                alphas[i],
                energies[i],
                group_energies[i],
                box_vectors[i],
                param_sampling.ParameterState(
                    discrete_parameters[i], continuous_parameters[i]
                ),
                mappings[i],
                alignments[i],
            )
            states.append(s)
        return states

    def append_traj(self, state: interfaces.IState, stage: int):
        """
        Append structure from state to end of trajectory

        Args:
            state: state to append
            stage: stage number
        """
        pdb_string = self._pdb_writer.get_pdb_string(state.positions, stage)
        with open(self._traj_path, "a") as traj_file:
            traj_file.write(pdb_string)

    def save_alphas(self, alphas: np.ndarray, stage: int):
        """
        Save alphas to disk.

        Args:
            alphas: n_replicas array
            stage: stage to store
        """
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        self._cdf_data_set.variables["alphas"][..., stage] = alphas

    def load_alphas(self, stage: int) -> np.ndarray:
        """
        Load alphas from disk.

        Args:
            stage: stage to load from disk
        Returns:
            n_replicas array
        """
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        return self._cdf_data_set.variables["alphas"][..., stage]

    def load_all_alphas(self) -> np.ndarray:
        """
        Load all alphas from disk.

        Returns:
            n_stage x n_replicas array

        .. warning::
           This could use a lot of memory.

        """
        return np.concatenate(
            [
                np.array(self.load_alphas(i))[..., np.newaxis]
                for i in range(self.max_safe_frame)
            ],
            axis=-1,
        )

    def save_energies(self, energies: np.ndarray, stage: int):
        """
        Save energies to disk.

        Args:
            energies: n_replicas array of energy
            stage: stage to save
        """
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        self._cdf_data_set.variables["energies"][..., stage] = energies

    def load_energies(self, stage) -> np.ndarray:
        """
        Load energies from disk.

        Args:
            stage: stage to load

        Returns:
            n_replicas array of energies

        """
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        return self._cdf_data_set.variables["energies"][..., stage]

    def save_group_energies(self, group_energies: np.ndarray, stage):
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        self._cdf_data_set.variables["group_energies"][..., stage] = group_energies

    def load_group_energies(self, stage) -> np.ndarray:
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        return self._cdf_data_set.variables["group_energies"][..., stage]

    def load_all_energies(self) -> np.ndarray:
        """
        Load all energies from disk.

        Returns:
            n_stage x n_replicas array of energies

        .. warning::
           This could use a lot of memory
        """
        return np.concatenate(
            [
                np.array(self.load_energies(i))[..., np.newaxis]
                for i in range(self.max_safe_frame)
            ],
            axis=-1,
        )

    def save_energy_matrix(self, energy_matrix: np.ndarray, stage: int):
        """
        Save energy matrix to disk

        Args:
            energy_matrix: n_replicas x n_replicas matrix of energies
            stage: stage to store
        """
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        self._cdf_data_set.variables["energy_matrix"][..., stage] = energy_matrix

    def load_energy_matrix(self, stage: int) -> np.ndarray:
        """
        Load energy matrix from disk

        Args:
            stage: stage to laod

        Returns:
            n_replicas x n_replicas array of energies
        """
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        return self._cdf_data_set.variables["energy_matrix"][..., stage]

    def load_all_energy_matrices(self) -> np.ndarray:
        """
        Load all energy matrix from disk

        Returns:
            n_stages x n_replicas x n_replicas array of energies
        """
        return np.concatenate(
            [
                np.array(self.load_energy_matrix(i))[..., np.newaxis]
                for i in range(self.max_safe_frame)
            ],
            axis=-1,
        )

    def save_permutation_vector(self, perm_vec: np.ndarray, stage: int):
        """
        Save permutation vector to disk.

        Args:
            perm_vec: n_replicas array of int
            stage: stage to store
        """
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        self._cdf_data_set.variables["permutation_vectors"][..., stage] = perm_vec

    def load_permutation_vector(self, stage: int) -> np.ndarray:
        """
        Load permutation vector from disk.

        Args:
            stage: stage to load

        Returns:
            n_replicas array of int
        """
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        return self._cdf_data_set.variables["permutation_vectors"][..., stage]

    def load_all_permutation_vectors(self) -> np.ndarray:
        """
        Load all permutation vector from disk.

        Returns:
            n_stages x n_replicas array of int

        .. warning::
           This might take a lot of memory
        """
        return np.concatenate(
            [
                np.array(self.load_permutation_vector(i))[..., np.newaxis]
                for i in range(self.max_safe_frame)
            ],
            axis=-1,
        )

    def iterate_permutation_vectors(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Iterator[np.ndarray]:
        """
        Iterate over the permutation vectors from disk.

        Args:
            start: starting stage
            end: ending stage

        Returns:
            an iterator over n_replicas array of int
        """
        if start is None:
            start = 0
        if end is None:
            end = self.max_safe_frame

        for i in range(start, end):
            yield self.load_permutation_vector(i)

    def save_acceptance_probabilities(self, accept_probs: np.ndarray, stage: int):
        """
        Save acceptance probabilities vector to disk.

        Args:
            accept_probs: n_replica_pairs array
            stage: stage to store
        """
        self._can_save()
        self._handle_save_stage(stage)
        assert self._cdf_data_set is not None
        ds = self._cdf_data_set
        ds.variables["acceptance_probabilities"][..., stage] = accept_probs

    def load_acceptance_probabilities(self, stage: int) -> np.ndarray:
        """
        Load acceptance probability vector from disk.

        Args:
            stage: stage to load

        Returns:
            n_replica_pairs array
        """
        self._handle_load_stage(stage)
        assert self._cdf_data_set is not None
        ds = self._cdf_data_set
        return ds.variables["acceptance_probabilities"][..., stage]

    def load_all_acceptance_probabilities(self) -> np.ndarray:
        """
        Load all acceptance probabilities from disk

        Returns:
            n_stages x n_replica_pairs array

        .. warning::
           This might take a lot of memory
        """
        return np.concatenate(
            [
                np.array(self.load_acceptance_probabilities(i))[..., np.newaxis]
                for i in range(self.max_safe_frame)
            ],
            axis=-1,
        )

    def save_discrete_parameters(self, data, stage):
        self._can_save()
        self._handle_save_stage(stage)
        ds = self._cdf_data_set
        ds.variables["discrete_parameters"][..., stage] = data

    def load_discrete_parameters(self, stage):
        self._handle_load_stage(stage)
        ds = self._cdf_data_set
        return ds.variables["discrete_parameters"][..., stage]

    def save_continuous_parameters(self, data, stage):
        self._can_save()
        self._handle_save_stage(stage)
        ds = self._cdf_data_set
        ds.variables["continuous_parameters"][..., stage] = data

    def load_continuous_parameters(self, stage):
        self._handle_load_stage(stage)
        ds = self._cdf_data_set
        return ds.variables["continuous_parameters"][..., stage]

    def save_mappings(self, data, stage):
        self._can_save()
        self._handle_save_stage(stage)
        ds = self._cdf_data_set
        ds.variables["mappings"][..., stage] = data

    def load_mappings(self, stage):
        self._handle_load_stage(stage)
        ds = self._cdf_data_set
        return ds.variables["mappings"][..., stage]

    def save_alignments(self, data, stage):
        self._can_save()
        self._handle_save_stage(stage)
        ds = self._cdf_data_set
        ds.variables["rdc_alignments"][..., stage] = data

    def load_alignments(self, stage):
        self._handle_load_stage(stage)
        ds = self._cdf_data_set
        return ds.variables["rdc_alignments"][..., stage]

    def save_remd_runner(self, runner):
        """
        Save replica runner to disk

        Args:
            runner (LeaderReplicaExchangeRunner): replica exchange runner to save
        """
        self._can_save()
        with open(self._remd_runner_path, "wb") as runner_file:
            pickle.dump(runner, runner_file)

    def load_remd_runner(self):
        """
        Load replica runner from disk

        Returns:
            LeaderReplicaExchangeRunner
        """
        path = (
            self._remd_runner_backup_path
            if self._readonly_mode
            else self._remd_runner_path
        )
        with open(path, "rb") as runner_file:
            return _load_pickle(runner_file)

    def save_system(self, system: interfaces.ISystem):
        """
        Save MELD system to disk

        Args:
            system: system to save
        """
        self._can_save()
        with open(self._system_path, "wb") as system_file:
            pickle.dump(system, system_file)

    def load_system(self) -> interfaces.ISystem:
        """Load MELD system from disk"""
        path = self._system_backup_path if self._readonly_mode else self._system_path
        with open(path, "rb") as system_file:
            return _load_pickle(system_file)

    def save_run_options(self, run_options: options.RunOptions):
        """
        Save RunOptions to disk
        Args:
            run_options: options to save to disk
        """
        self._can_save()
        with open(self._run_options_path, "wb") as options_file:
            pickle.dump(run_options, options_file)

    def load_run_options(self) -> options.RunOptions:
        """Load RunOptions from disk"""
        path = (
            self._run_options_backup_path
            if self._readonly_mode
            else self._run_options_path
        )
        with open(path, "rb") as options_file:
            options = _load_pickle(options_file)
        return options

    if has_gamd == True:

        def save_integrator(self, integrator: GamdStageIntegrator):
            """
            Save integrator.
            """
            self._can_save()
            with open(self._integrator_path, "wb") as integrator_file:
                pickle.dump(integrator, integrator_file)

        def load_integrator(self) -> GamdStageIntegrator:
            """Load integrator"""
            path = (
                self._integrator_backup_path
                if self._readonly_mode
                else self._integrator_path
            )
            with open(path, "rb") as integrator_file:
                return _load_pickle(integrator_file)

    def backup(self, stage: int):
        """
        Backup all files to Data/Backup.

        Backup will occur if `stage % backup_freq == 0`

        Args:
            stage: stage
        """
        self._can_save()
        if not stage % self._block_size:
            self._backup(self._communicator_path, self._communicator_backup_path)
            self._backup(self._data_store_path, self._data_store_backup_path)
            self._backup(self._remd_runner_path, self._remd_runner_backup_path)
            self._backup(self._system_path, self._system_backup_path)
            self._backup(self._run_options_path, self._run_options_backup_path)
            self._backup(self._integrator_path, self._integrator_backup_path)
            
    #
    # private methods
    #

    def _create_cdf_file(self):
        # create the file
        path = self._net_cdf_path_template.format(self._current_block)
        ds = cdf.Dataset(path, "w", format="NETCDF4")

        # setup dimensions
        ds.createDimension("n_replicas", self._n_replicas)
        ds.createDimension("n_replica_pairs", self._n_replicas - 1)
        ds.createDimension("n_atoms", self._n_atoms)
        ds.createDimension("cartesian", 3)
        ds.createDimension("timesteps", None)
        ds.createDimension("n_discrete_parameters", self._n_discrete_parameters)
        ds.createDimension("n_continuous_parameters", self._n_continuous_parameters)
        ds.createDimension("n_mappings", self._n_mappings)
        ds.createDimension("n_alignments", self._n_alignments)
        ds.createDimension("n_energy_groups", ENERGY_GROUPS)

        # setup variables
        ds.createVariable(
            "positions",
            float,
            ["n_replicas", "n_atoms", "cartesian", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "velocities",
            float,
            ["n_replicas", "n_atoms", "cartesian", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "box_vectors",
            float,
            ["n_replicas", "cartesian", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "alphas",
            float,
            ["n_replicas", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "energies",
            float,
            ["n_replicas", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "group_energies",
            float,
            ["n_replicas", "n_energy_groups", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "permutation_vectors",
            int,
            ["n_replicas", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "energy_matrix",
            float,
            ["n_replicas", "n_replicas", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "acceptance_probabilities",
            float,
            ["n_replica_pairs", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "discrete_parameters",
            int,
            ["n_replicas", "n_discrete_parameters", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )
        ds.createVariable(
            "continuous_parameters",
            float,
            ["n_replicas", "n_continuous_parameters", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )

        ds.createVariable(
            "mappings",
            int,
            ["n_replicas", "n_mappings", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )

        ds.createVariable(
            "rdc_alignments",
            float,
            ["n_replicas", "n_alignments", "timesteps"],
            zlib=True,
            fletcher32=True,
            shuffle=True,
            complevel=9,
        )

        self._cdf_data_set = ds

    def _backup(self, src, dest):
        if os.path.exists(src):
            try:
                shutil.copy(src, dest)
            except IOError:
                # if we encounter an error, wait five seconds and try again
                time.sleep(5)
                shutil.copy(src, dest)

    def _can_save(self):
        if self._readonly_mode:
            raise RuntimeError("Cannot save in safe mode.")

    def _handle_save_stage(self, stage):
        if stage < self._current_stage:
            raise RuntimeError("Cannot go back in time")
        self._current_stage = stage
        block_index = self._block_for_stage(stage)

        if block_index > self._current_block:
            self.close()
            self._max_safe_block = self._current_block
            self._current_block = block_index
            self._create_cdf_file()

    def _handle_load_stage(self, stage):
        block_index = self._block_for_stage(stage)
        if self._readonly_mode:
            if block_index > self._max_safe_block:
                raise RuntimeError("Tried to read an unsafe block")
        else:
            if block_index < self._current_block:
                raise RuntimeError(
                    "Tried to load from an index before the current block,"
                    "which is not allowed."
                )

        if block_index != self._current_block:
            self.close()
            self._current_block = block_index
            self._load_cdf_file_readonly()

    def _block_for_stage(self, stage):
        return stage // self._block_size

    def _load_cdf_file_readonly(self):
        path = self._net_cdf_path_template.format(self._current_block)
        self._cdf_data_set = cdf.Dataset(path, "r")

    @property
    def max_safe_frame(self):
        """Maximum safe fram that can be read"""
        return (self._max_safe_block + 1) * self._block_size

    @property
    def max_safe_block(self):
        """Maximum safe block that can be read"""
        return self._max_safe_block
