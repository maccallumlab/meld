import os
import cPickle as pickle
import netCDF4 as cdf
import numpy as np
import shutil
from meld.system import state


class DataStore(object):
    '''
    Class to handle storing data from MeLD runs.

    Data will be stored in the 'Data' subdirectory. Backups will be stored in 'Data/Backup'.

    Some information is stored as python pickled files:
        data_store.dat -- the DataStore object
        communicator.dat -- the MPICommunicator object
        remd_runner.dat -- the MasterReplicaExchangeRunner object

    Other data (positions, velocities, etc) is stored in the results.nc file.

    '''
    #
    # data paths
    #
    data_dir = 'Data'
    backup_dir = os.path.join(data_dir, 'Backup')

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

    net_cdf_filename = 'results.nc'
    net_cdf_path = os.path.join(data_dir, net_cdf_filename)
    net_cdf_backup_path = os.path.join(backup_dir, net_cdf_filename)

    def __init__(self, n_atoms, n_springs, n_replicas, backup_freq=100):
        '''
        Create a DataStore object.

        Parameters
            n_atoms -- number of atoms
            n_springs -- number of springs
            n_replicas -- number of replicas
            backup_freq -- frequency to perform backups

        '''
        self._n_atoms = n_atoms
        self._n_springs = n_springs
        self._n_replicas = n_replicas
        self._backup_freq = backup_freq
        self._cdf_data_set = None
        self._safe_mode = False

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

    @property
    def n_springs(self):
        return self._n_springs

    #
    # public methods
    #

    def initialize(self, mode):
        '''
        Prepare to use the DataStore object.

        Parameters
            mode -- mode to open in.

        Available modes are:
            new -- create a new directory structure and initialize the hd5 file
            existing -- open the existing files
            safe -- open the backup copies in read-only mode to prevent corruption

        '''
        if mode == 'new':
            if os.path.exists(self.data_dir):
                raise RuntimeError('Data directory already exists')
            os.mkdir(self.data_dir)
            os.mkdir(self.backup_dir)
            self._cdf_data_set = cdf.Dataset(self.net_cdf_path, 'w', format='NETCDF4')
            self._setup_cdf()
        elif mode == 'existing':
            self._cdf_data_set = cdf.Dataset(self.net_cdf_path, 'a')
        elif mode == 'safe':
            self._cdf_data_set = cdf.Dataset(self.net_cdf_backup_path, 'r')
            self._safe_mode = True
        else:
            raise RuntimeError('Unknown value for mode={}'.format(mode))

    def close(self):
        '''Close the DataStore'''
        if self._cdf_data_set:
            self._cdf_data_set.close()
            self._cdf_data_set = None

    def save_data_store(self):
        '''Save this object to disk.'''
        with open(self.data_store_path, 'w') as store_file:
            pickle.dump(self, store_file)

    @classmethod
    def load_data_store(cls):
        '''Load the DataStore object from disk.'''
        with open(cls.data_store_path) as store_file:
            return pickle.load(store_file)

    def save_communicator(self, comm):
        '''Save the communicator to disk'''
        self._check_save()
        with open(self.communicator_path, 'w') as comm_file:
            pickle.dump(comm, comm_file)

    def load_communicator(self):
        '''Load the communicator from disk'''
        if self._safe_mode:
            path = self.communicator_backup_path
        else:
            path = self.communicator_path
        with open(path) as comm_file:
            return pickle.load(comm_file)

    def save_positions(self, positions, stage):
        '''
        Save the positions to disk.

        Parameters
            positions -- n_replicas x n_atoms x 3 array
            stage -- int stage to store

        '''
        self._cdf_data_set.variables['positions'][..., stage] = positions

    def load_positions(self, stage):
        '''
        Load positions from disk.

        Parameters
            stage -- int stage to load

        '''
        return self._cdf_data_set.variables['positions'][..., stage]

    def save_velocities(self, velocities, stage):
        '''
        Save velocities to disk.

        Parameters
            velocities -- n_replicas x n_atoms x 3 array
            stage -- int stage to store

        '''
        self._check_save()
        self._cdf_data_set.variables['velocities'][..., stage] = velocities

    def load_velocities(self, stage):
        '''
        Load velocities from disk.

        Parameters
            stage -- int stage to load

        '''
        return self._cdf_data_set.variables['velocities'][..., stage]

    def save_spring_states(self, spring_states, stage):
        '''
        Save spring states to disk.

        Parameters
            spring_states -- n_replicas x n_springs array
            stage -- int stage to store

        '''
        self._check_save()
        self._cdf_data_set.variables['spring_states'][..., stage] = spring_states

    def load_spring_states(self, stage):
        '''
        Load spring states from disk.

        Parameters
            stage -- int stage to load

        '''
        return self._cdf_data_set.variables['spring_states'][:, :, stage]

    def save_spring_energies(self, spring_energies, stage):
        '''
        Save spring energies to disk.

        Parameters
            spring_energies --- n_replicas x n_springs array
            stage -- int stage to store

        '''
        self._check_save()
        self._cdf_data_set.variables['spring_energies'][..., stage] = spring_energies

    def load_spring_energies(self, stage):
        '''
        Load spring energies from disk.

        Parameters
            stage -- int stage to load

        Returns
            n_replicas x n_springs array

        '''
        return self._cdf_data_set.variables['spring_energies'][..., stage]

    def save_states(self, states, stage):
        '''
        Save states to disk.

        Parameters
            states -- list of SystemStage objects to store
            stage -- int stage to store

        '''
        self._check_save()
        positions = np.array([s.positions for s in states])
        velocities = np.array([s.velocities for s in states])
        spring_states = np.array([s.spring_states for s in states])
        spring_energies = np.array([s.spring_energies for s in states])
        lambdas = np.array([s.lam for s in states])
        energies = np.array([s.energy for s in states])
        self.save_positions(positions, stage)
        self.save_velocities(velocities, stage)
        self.save_spring_states(spring_states, stage)
        self.save_spring_energies(spring_energies, stage)
        self.save_lambdas(lambdas, stage)
        self.save_energies(energies, stage)

    def load_states(self, stage):
        '''
        Load states from disk

        Parameters
            stage -- integer stage to load

        Returns
            list of SystemState objects

        '''
        positions = self.load_positions(stage)
        velocities = self.load_velocities(stage)
        spring_states = self.load_spring_states(stage)
        spring_energies = self.load_spring_energies(stage)
        lambdas = self.load_lambdas(stage)
        energies = self.load_energies(stage)
        states = []
        for i in range(self._n_replicas):
            s = state.SystemState(positions[i], velocities[i], spring_states[i], lambdas[i], energies[i],
                                  spring_energies[i])
            states.append(s)
        return states

    def append_traj(self, state):
        pass

    def save_lambdas(self, lambdas, stage):
        '''
        Save lambdas to disk.

        Parameters
            lambdas -- n_replicas array
            stage -- int stage to store

        '''
        self._check_save()
        self._cdf_data_set.variables['lambdas'][..., stage] = lambdas

    def load_lambdas(self, stage):
        '''
        Load lambdas from disk.

        Parameters
            stage -- int stage to load from disk

        Returns
            n_replicas array

        '''
        return self._cdf_data_set.variables['lambdas'][..., stage]

    def save_energies(self, energies, stage):
        '''
        Save energies to disk.

        Parameters
            energies -- n_replicas array
            stage -- int stage to save

        '''
        self._check_save()
        self._cdf_data_set.variables['energies'][..., stage] = energies

    def load_energies(self, stage):
        '''
        Load energies from disk.

        Parameters
            stage -- int stage to load

        Returns
            n_replicas array

        '''
        return self._cdf_data_set.variables['energies'][..., stage]

    def save_permutation_vector(self, perm_vec, stage):
        '''
        Save permutation vector to disk.

        Parameters
            perm_vec -- n_replicas array of int
            stage -- int stage to store

        '''
        self._check_save()
        self._cdf_data_set.variables['permutation_vectors'][..., stage] = perm_vec

    def load_permutation_vector(self, stage):
        '''
        Load permutation vector from disk.

        Parameters
            stage -- int stage to load

        Returns
            n_replicas array of int

        '''
        return self._cdf_data_set.variables['permutation_vectors'][..., stage]

    def save_remd_runner(self, runner):
        '''Save replica runner to disk'''
        self._check_save()
        with open(self.remd_runner_path, 'w') as runner_file:
            pickle.dump(runner, runner_file)

    def load_remd_runner(self):
        '''Load replica runner from disk'''
        path = self.remd_runner_backup_path if self._safe_mode else self.remd_runner_path
        with open(path) as runner_file:
            return pickle.load(runner_file)

    def save_system(self, system):
        self._check_save()
        with open(self.system_path, 'w') as system_file:
            pickle.dump(system, system_file)

    def load_system(self):
        path = self.system_backup_path if self._safe_mode else self.system_path
        with open(path) as system_file:
            return pickle.load(system_file)

    def backup(self, stage):
        '''
        Backup all files to Data/Backup.

        Parameters
            stage -- int stage

        Backup will occur if stage mod backup_freq == 0

        '''
        self._check_save()
        if not stage % self._backup_freq:
            self._backup(self.communicator_path, self.communicator_backup_path)
            self._backup(self.data_store_path, self.data_store_backup_path)
            self._backup(self.remd_runner_path, self.remd_runner_backup_path)
            self._backup(self.system_path, self.system_backup_path)

            self._cdf_data_set.close()
            self._backup(self.net_cdf_path, self.net_cdf_backup_path)
            self._cdf_data_set = cdf.Dataset(self.net_cdf_path, 'a')

    #
    # private methods
    #

    def _setup_cdf(self):
        # setup dimensions
        self._cdf_data_set.createDimension('n_replicas', self._n_replicas)
        self._cdf_data_set.createDimension('n_atoms', self._n_atoms)
        self._cdf_data_set.createDimension('cartesian', 3)
        self._cdf_data_set.createDimension('n_springs', self._n_springs)
        self._cdf_data_set.createDimension('timesteps', None)

        # setup variables
        self._cdf_data_set.createVariable('positions', float, ['n_replicas', 'n_atoms', 'cartesian', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True)
        self._cdf_data_set.createVariable('velocities', float, ['n_replicas', 'n_atoms', 'cartesian', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True)
        self._cdf_data_set.createVariable('spring_states', float, ['n_replicas', 'n_springs', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True)
        self._cdf_data_set.createVariable('spring_energies', float, ['n_replicas', 'n_springs', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True)
        self._cdf_data_set.createVariable('lambdas', float, ['n_replicas', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True)
        self._cdf_data_set.createVariable('energies', float, ['n_replicas', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True)
        self._cdf_data_set.createVariable('permutation_vectors', int, ['n_replicas', 'timesteps'],
                                          zlib=True, fletcher32=True, shuffle=True)

    def _backup(self, src, dest):
        if os.path.exists(src):
            shutil.copy(src, dest)

    def _check_save(self):
        if self._safe_mode:
            raise RuntimeError('Cannot save in safe mode.')
