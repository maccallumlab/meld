import numpy as np
import platform
from collections import defaultdict
import logging
from util import log_timing

logger = logging.getLogger(__name__)


class MPICommunicator(object):
    """
    Class to handle communications between master and slaves using MPI

    """
    def __init__(self, n_atoms, n_replicas):
        """
        Create an MPICommunicator

        Parameters
            n_atoms -- number of atoms
            n_replicas -- number of replicas

        Note: creating an MPI communicator will not actually initialize MPI. To do that,
        call initialize().

        """
        # We're not using n_atoms and n_replicas, but if we switch
        # to more efficient buffer-based MPI routines, we'll need them.
        self._n_atoms = n_atoms
        self._n_replicas = n_replicas
        self._mpi_comm = None

    def __getstate__(self):
        # don't pickle _mpi_comm
        return dict((k, v) for (k, v) in self.__dict__.iteritems() if not k == '_mpi_comm')

    def __setstate__(self, state):
        # set _mpi_comm to None
        self.__dict__ = state
        self._mpi_comm = None

    def initialize(self):
        """
        Initialize and start MPI

        """
        self._mpi_comm = get_mpi_comm_world()
        self._my_rank = self._mpi_comm.Get_rank()

    def is_master(self):
        """
        Is this the master node?

        Returns
            True if we are the master, otherwise False

        """
        if self._my_rank == 0:
            return True
        else:
            return False

    @log_timing(logger)
    def broadcast_alphas_to_slaves(self, alphas):
        """
        Send the alpha values to the slaves

        Parameters
            alphas -- a list of alpha values, one for each replica
        Returns
            None

        The master node's alpha value should be included in this list.
        The master node will always be at alpha=0.0

        """
        self._mpi_comm.scatter(alphas, root=0)

    @log_timing(logger)
    def receive_alpha_from_master(self):
        """
        Receive alpha value from master node

        Returns
            a floating point value for alpha in [0,1]

        """
        return self._mpi_comm.scatter(None, root=0)

    @log_timing(logger)
    def broadcast_states_to_slaves(self, states):
        """
        Send a state to each slave

        Parameters
            states -- a list of states
        Returns
            the state to run on the master node

        The list of states should include the state for the master node. These are the
        states that will be simulated on each replica for each step.

        """
        return self._mpi_comm.scatter(states, root=0)

    @log_timing(logger)
    def receive_state_from_master(self):
        """
        Get state to run for this step

        Returns
            the state to run for this step

        """
        return self._mpi_comm.scatter(None, root=0)

    @log_timing(logger)
    def gather_states_from_slaves(self, state_on_master):
        """
        Receive states from all slaves

        Parameters
            state_on_master -- the state on the master after simulating
        Returns
            a list of states, one from each replica

        The returned states are the states after simulating.

        """
        return self._mpi_comm.gather(state_on_master, root=0)

    @log_timing(logger)
    def send_state_to_master(self, state):
        """
        Send state to master

        Parameters
            state -- state to send to master
        Returns
            None

        This is the state after simulating this step.

        """
        self._mpi_comm.gather(state, root=0)

    @log_timing(logger)
    def broadcast_states_for_energy_calc_to_slaves(self, states):
        """
        Broadcast states to all slaves

        Parameters
            states -- a list of states
        Returns
            None

        Send all results from this step to every slave so that we can calculate
        the energies and do replica exchange.

        """
        self._mpi_comm.bcast(states, root=0)

    @log_timing(logger)
    def receive_states_for_energy_calc_from_master(self):
        """
        Receive all states from master

        Returns
            a list of states to calculate the energy of

        """
        return self._mpi_comm.bcast(None, root=0)

    @log_timing(logger)
    def gather_energies_from_slaves(self, energies_on_master):
        """
        Receive a list of energies from each slave

        Parameters
            energies_on_master -- a list of energies from the master
        Returns
            a square matrix of every state on every replica to be used for replica exchange

        """
        energies = self._mpi_comm.gather(energies_on_master, root=0)
        return np.array(energies)

    @log_timing(logger)
    def send_energies_to_master(self, energies):
        """
        Send a list of energies to the master

        Parameters
            energies -- a list of energies to send to the master
        Returns
            None

        """
        return self._mpi_comm.gather(energies, root=0)

    @log_timing(logger)
    def negotiate_device_id(self):
        hostname = platform.node()
        hostnames = self._mpi_comm.gather(hostname, root=0)
        if self._my_rank == 0:
            host_counts = defaultdict(int)
            device_ids = []
            for hostname in hostnames:
                device_ids.append(host_counts[hostname])
                host_counts[hostname] += 1
        else:
            device_ids = None
        device_id = self._mpi_comm.scatter(device_ids, root=0)
        return device_id

    @property
    def n_replicas(self):
        return self._n_replicas

    @property
    def n_atoms(self):
        return self._n_atoms

    @property
    def rank(self):
        return self._my_rank


def get_mpi_comm_world():
    """
    Helper function to import mpi4py and return the comm_world.

    """
    from mpi4py import MPI
    return MPI.COMM_WORLD
