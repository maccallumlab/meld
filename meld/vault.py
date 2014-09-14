import contextlib
import os
import time
import cPickle as pickle
import netCDF4 as cdf
import numpy as np
import shutil
from meld.system import state


class DataStore(object):
    """
    Class to handle storing data from MELD runs.

    :param n_atoms: number of atoms
    :param n_replicas: number of replicas
    :param block_size: size of netcdf blocks and frequency to do backups

    Data will be stored in the 'Data' subdirectory. Backups will be stored in 'Data/Backup'.

    Some information is stored as python pickled files:
    
    - data_store.dat -- the DataStore object
    - communicator.dat -- the MPICommunicator object
    - remd_runner.dat -- the MasterReplicaExchangeRunner object

    Other data (positions, velocities, etc) is stored in the results.nc file.

    """
    #
    # data paths
    #
    data_dir = 'Data'
    backup_dir = os.path.join(data_dir, 'Backup')
    blocks_dir = os.path.join(data_dir, 'Blocks')

    data_store_filename = 'data_store.dat'
    data_store_path = os.path.join(data_dir, data_store_filename)
    data_store_backup_path = os.path.join(backup_dir, data_store_filename)

    communicator_filename = 'communicator.dat'
    communicator_path = os.path.join(data_dir, communicator_filename)
    communicator_backup_path = os.path.join(backup_dir, communicator_filename)

    remd_runner_filename = 'remd_runner.dat'
    remd_runner_path = os.path.join(data_dir, remd_runner_filename)
    remd_runner_backup_path = os.path.join(backup_dir, remd_runner_filename)

    system_filename = 'system.dat'
    system_path = os.path.join(data_dir, system_filename)
    system_backup_path = os.path.join(backup_dir, system_filename)

    run_options_filename = 'run_options.dat'
    run_options_path = os.path.join(data_dir, run_options_filename)
    run_options_backup_path = os.path.join(backup_dir, run_options_filename)

    net_cdf_filename_template = 'block_{:06d}.nc'
    net_cdf_path_template = os.path.join(blocks_dir, net_cdf_filename_template)

    traj_filename = 'trajectory.pdb'
    traj_path = os.path.join(data_dir, traj_filename)
    traj_backup_path = os.path.join(backup_dir, traj_filename)

    def __init__(self, n_atoms, n_replicas, pdb_writer, block_size=100):
        self._n_atoms = n_atoms
        self._n_replicas = n_replicas
        self._block_size = block_size
        self._cdf_data_set = None
        self._readonly_mode = False
        self._pdb_writer = pdb_writer
        self._current_stage = None
        self._current_block = None
        self._max_safe_block = -1
        self._readonly_mode = False

    def __getstate__(self):
        # don't save some fields to disk
        excluded = ['_cdf_data_set']
        return dict((k, v) for (k, v) in self.__dict__.iteritems() if not k in excluded)

    def __setstate__(self, state):
        # set _cdf_data_set to None
        self.__dict__ = state
        self._cdf_data_set = None

    def __del__(self):
        # close the _cdf_data_set when we go out of scope
        if self._cdf_data_set:
            self._cdf_data_set.close()
    #
    # properties
    #

    @property
    def n_replicas(self):
        return self._n_replicas

    @property
    def n_atoms(self):
        return self._n_atoms

    #
    # public methods
    #
    def initialize(self, mode):
        """
        Prepare to use the DataStore object.

        :param mode: mode to open in.

        Available modes are:
        
        - 'w' -- create a new directory structure and initialize the hd5 file
        - 'a' -- append to the existing files
        - 'r' -- open the file in read-only mode

        """
        if mode == 'w':
            if os.path.exists(self.data_dir):
                raise RuntimeError('Data directory already exists')
            os.mkdir(self.data_dir)
            os.mkdir(self.blocks_dir)
            os.mkdir(self.backup_dir)
            self._current_block = 0
            self._current_stage = 0
            self._create_cdf_file()
        elif mode == 'a':
            block_path = self.net_cdf_path_template.format(self._current_block)
            if os.path.exists(block_path):
                self._cdf_data_set = cdf.Dataset(block_path, 'a')
            else:
                self._create_cdf_file()
        elif mode == 'r':
            self._current_block = 0
            self._readonly_mode = True
            self._load_cdf_file_readonly()
        else:
            raise RuntimeError('Unknown value for mode={}'.format(mode))

    def close(self):
        """Close the DataStore"""
        if self._cdf_data_set:
            self._cdf_data_set.close()
            self._cdf_data_set = None

    def save_data_store(self):
        """Save this object to disk."""
        with open(self.data_store_path, 'w') as store_file:
            pickle.dump(self, store_file)

    @classmethod
    def load_data_store(cls, load_backup=False):
        """Load the DataStore object from disk."""
        path = cls.data_store_backup_path if load_backup else cls.data_store_path
        with open(path) as store_file:
            return pickle.load(store_file)

    def save_communicator(self, comm):
        """Save the communicator to disk"""
        self._can_save()
        with open(self.communicator_path, 'w') as comm_file:
            pickle.dump(comm, comm_file)

    def load_communicator(self):
        """Load the communicator from disk"""
        if self._readonly_mode:
            path = self.communicator_backup_path
        else:
            path = self.communicator_path
        with open(path) as comm_file:
            return pickle.load(comm_file)

    def save_positions(self, positions, stage):
        """
        Save the positions to disk.

        :param positions: n_replicas x n_atoms x 3 array
        :param stage: int stage to store

        """
        self._can_save()
        self._handle_save_stage(stage)
        self._cdf_data_set.variables['positions'][..., stage] = positions

    def load_positions(self, stage):
        """
        Load positions from disk.

        :param stage: int stage to load

        """
        self._handle_load_stage(stage)
        return self._cdf_data_set.variables['positions'][..., stage]

    def load_positions_random_access(self, stage):
        """
        Load positions from disk.

        :param stage: int stage to load

        This differs from :meth:`load_positions` in that you can positions from any stage,
        while :meth:`load_positions` can only move forward in time. However, this comes at
        a performance penalty.
        """
        # get the block for this stage
        block = self._block_for_stage(stage)

        # if it's the current block, then just return the positions
        if block == self._current_block:
            return self._cdf_data_set.variables['positions'][..., stage]

        # otherwise open the file, grab the positions, and then close it
        else:
            path = self.net_cdf_path_template.format(block)
            with contextlib.closing(cdf.Dataset(path, 'r')) as dataset:
                return dataset.variables['positions'][..., stage]

    def load_all_positions(self):
        """
        Load all positions from disk.

        Warning, this could use a lot of memory.

        """
        return np.concatenate([np.array(self.load_positions(i))[..., np.newaxis]
                               for i in range(self.max_safe_frame())], axis=-1)

    def iterate_positions(self, start=None, end=None):
        """
        Iterate over the positions from disk.

        """
        if start is None:
            start = 0
        if end is None:
            end = self.max_safe_frame()

        for i in range(start, end):
            yield self.load_positions(i)

    def save_velocities(self, velocities, stage):
        """
        Save velocities to disk.

        :param velocities: n_replicas x n_atoms x 3 array
        :param stage: int stage to store

        """
        self._can_save()
        self._handle_save_stage(stage)
        self._cdf_data_set.variables['velocities'][..., stage] = velocities

    def load_velocities(self, stage):
        """
        Load velocities from disk.

        :param stage: int stage to load

        """
        self._handle_load_stage(stage)
        return self._cdf_data_set.variables['velocities'][..., stage]

    def load_all_velocities(self):
        """
        Load all velocities from disk.

        Warning, this could use a lot of memory.

        """
        return np.concatenate([np.array(self.load_velocities(i))[..., np.newaxis]
                               for i in range(self.max_safe_frame())], axis=-1)

    def save_states(self, states, stage):
        """
        Save states to disk.

        :param states: list of SystemStage objects to store
        :param stage: int stage to store

        """
        self._can_save()
        self._handle_save_stage(stage)
        positions = np.array([s.positions for s in states])
        velocities = np.array([s.velocities for s in states])
        alphas = np.array([s.alpha for s in states])
        energies = np.array([s.energy for s in states])
        self.save_positions(positions, stage)
        self.save_velocities(velocities, stage)
        self.save_alphas(alphas, stage)
        self.save_energies(energies, stage)

    def load_states(self, stage):
        """
        Load states from disk

        :param stage: integer stage to load

        :return: list of SystemState objects

        """
        self._handle_load_stage(stage)
        positions = self.load_positions(stage)
        velocities = self.load_velocities(stage)
        alphas = self.load_alphas(stage)
        energies = self.load_energies(stage)
        states = []
        for i in range(self._n_replicas):
            s = state.SystemState(positions[i], velocities[i], alphas[i], energies[i])
            states.append(s)
        return states

    def append_traj(self, state, stage):
        pdb_string = self._pdb_writer.get_pdb_string(state.positions, stage)
        with open(self.traj_path, 'a') as traj_file:
            traj_file.write(pdb_string)

    def save_alphas(self, alphas, stage):
        """
        Save alphas to disk.

        :param alphas: n_replicas array
        :param stage: int stage to store

        """
        self._can_save()
        self._handle_save_stage(stage)
        self._cdf_data_set.variables['alphas'][..., stage] = alphas

    def load_alphas(self, stage):
        """
        Load alphas from disk.

        :param stage: int stage to load from disk
        :return: n_replicas array

        """
        self._handle_load_stage(stage)
        return self._cdf_data_set.variables['alphas'][..., stage]

    def load_all_alphas(self):
        """
        Load all alphas from disk.

        Warning, this could use a lot of memory.

        """
        return np.concatenate([np.array(self.load_alphas(i))[..., np.newaxis]
                               for i in range(self.max_safe_frame())], axis=-1)

    def save_energies(self, energies, stage):
        """
        Save energies to disk.

        :param energies: n_replicas array
        :param stage: int stage to save

        """
        self._can_save()
        self._handle_save_stage(stage)
        self._cdf_data_set.variables['energies'][..., stage] = energies

    def load_energies(self, stage):
        """
        Load energies from disk.

        :param stage: int stage to load
        :return: n_replicas array

        """
        self._handle_load_stage(stage)
        return self._cdf_data_set.variables['energies'][..., stage]

    def load_all_energies(self):
        """
        Load all energies from disk.

        Warning, this could use a lot of memory

        """
        return np.concatenate([np.array(self.load_energies(i))[..., np.newaxis]
                               for i in range(self.max_safe_frame())], axis=-1)

    def save_energy_matrix(self, energy_matrix, stage):
        self._can_save()
        self._handle_save_stage(stage)
        self._cdf_data_set.variables['energy_matrix'][..., stage] = energy_matrix

    def load_energy_matrix(self, stage):
        self._handle_load_stage(stage)
        return self._cdf_data_set.variables['energy_matrix'][..., stage]

    def load_all_energy_matrices(self):
        return np.concatenate([np.array(self.load_energy_matrix(i))[..., np.newaxis]
                               for i in range(self.max_safe_frame())], axis=-1)

    def save_permutation_vector(self, perm_vec, stage):
        """
        Save permutation vector to disk.

        :param perm_vec: n_replicas array of int
        :param stage: int stage to store

        """
        self._can_save()
        self._handle_save_stage(stage)
        self._cdf_data_set.variables['permutation_vectors'][..., stage] = perm_vec

    def load_permutation_vector(self, stage):
        """
        Load permutation vector from disk.

        :param stage: int stage to load

        :return: n_replicas array of int

        """
        self._handle_load_stage(stage)
        return self._cdf_data_set.variables['permutation_vectors'][..., stage]

    def load_all_permutation_vectors(self):
        """
        Load all permutation vector from disk.

        Warning, this might take a lot of memory

        """
        return np.concatenate([np.array(self.load_permutation_vector(i))[..., np.newaxis]
                               for i in range(self.max_safe_frame())], axis=-1)

    def iterate_permutation_vectors(self, start=None, end=None):
        """
        Iterate over the permutation vectors from disk.

        """
        if start is None:
            start = 0
        if end is None:
            end = self.max_safe_frame()

        for i in range(start, end):
            yield self.load_permutation_vector(i)

    def save_acceptance_probabilities(self, accept_probs, stage):
        """
        Save acceptance probabilities vector to disk.

        :param accept_probs: n_replicas array of int
        :param stage: int stage to store

        """
        self._can_save()
        self._handle_save_stage(stage)
        self._cdf_data_set.variables['acceptance_probabilities'][..., stage] = accept_probs

    def load_acceptance_probabilities(self, stage):
        """
        Load acceptance probability vector from disk.

        :param stage: int stage to load
        :return: n_replica_pairs array of int

        """
        self._handle_load_stage(stage)
        return self._cdf_data_set.variables['acceptance_probabilities'][..., stage]

    def load_all_acceptance_probabilities(self):
        """
        Load all acceptance probabilities from disk

        Warning, this might take a lot of memory

        """
        return np.concatenate([np.array(self.load_acceptance_probabilities(i))[..., np.newaxis]
                               for i in range(self.max_safe_frame())], axis=-1)

    def save_remd_runner(self, runner):
        """Save replica runner to disk"""
        self._can_save()
        with open(self.remd_runner_path, 'w') as runner_file:
            pickle.dump(runner, runner_file)

    def load_remd_runner(self):
        """Load replica runner from disk"""
        path = self.remd_runner_backup_path if self._readonly_mode else self.remd_runner_path
        with open(path) as runner_file:
            return pickle.load(runner_file)

    def save_system(self, system):
        self._can_save()
        with open(self.system_path, 'w') as system_file:
            pickle.dump(system, system_file)

    def load_system(self):
        path = self.system_backup_path if self._readonly_mode else self.system_path
        with open(path) as system_file:
            return pickle.load(system_file)

    def save_run_options(self, run_options):
        self._can_save()
        with open(self.run_options_path, 'w') as options_file:
            pickle.dump(run_options, options_file)

    def load_run_options(self):
        path = self.run_options_backup_path if self._readonly_mode else self.run_options_path
        with open(path) as options_file:
            return pickle.load(options_file)

    def backup(self, stage):
        """
        Backup all files to Data/Backup.

        :param stage: int stage

        Backup will occur if `stage % backup_freq == 0`

        """
        self._can_save()
        if not stage % self._block_size:
            self._backup(self.communicator_path, self.communicator_backup_path)
            self._backup(self.data_store_path, self.data_store_backup_path)
            self._backup(self.remd_runner_path, self.remd_runner_backup_path)
            self._backup(self.system_path, self.system_backup_path)
            self._backup(self.run_options_path, self.run_options_backup_path)

    #
    # private methods
    #

    def _create_cdf_file(self):
        # create the file
        path = self.net_cdf_path_template.format(self._current_block)
        self._cdf_data_set = cdf.Dataset(path, 'w', format='NETCDF4')

        # setup dimensions
        self._cdf_data_set.createDimension('n_replicas', self._n_replicas)
        self._cdf_data_set.createDimension('n_replica_pairs', self._n_replicas - 1)
        self._cdf_data_set.createDimension('n_atoms', self._n_atoms)
        self._cdf_data_set.createDimension('cartesian', 3)
        self._cdf_data_set.createDimension('timesteps', None)

        # setup variables
        self._cdf_data_set.createVariable('positions', float, ['n_replicas', 'n_atoms', 'cartesian', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True, complevel=9)
        self._cdf_data_set.createVariable('velocities', float, ['n_replicas', 'n_atoms', 'cartesian', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True, complevel=9)
        self._cdf_data_set.createVariable('alphas', float, ['n_replicas', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True, complevel=9)
        self._cdf_data_set.createVariable('energies', float, ['n_replicas', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True, complevel=9)
        self._cdf_data_set.createVariable('permutation_vectors', int, ['n_replicas', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True, complevel=9)
        self._cdf_data_set.createVariable('energy_matrix', float, ['n_replicas', 'n_replicas',
                                          'timesteps'], zlib=True, fletcher32=True, shuffle=True, complevel=9)
        self._cdf_data_set.createVariable('acceptance_probabilities', float, ['n_replica_pairs', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True, complevel=9)

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
            raise RuntimeError('Cannot save in safe mode.')

    def _handle_save_stage(self, stage):
        if stage < self._current_stage:
            raise RuntimeError('Cannot go back in time')
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
                raise RuntimeError('Tried to read an unsafe block')
        else:
            if block_index < self._current_block:
                raise RuntimeError('Cannot go back in time')

        if block_index != self._current_block:
            self.close()
            self._current_block = block_index
            self._load_cdf_file_readonly()

    def _block_for_stage(self, stage):
        return stage / self._block_size

    def _load_cdf_file_readonly(self):
        path = self.net_cdf_path_template.format(self._current_block)
        self._cdf_data_set = cdf.Dataset(path, 'r')

    def max_safe_frame(self):
        return (self._max_safe_block + 1) * self._block_size
