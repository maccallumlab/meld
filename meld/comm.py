import os
import numpy as np
import platform
from collections import defaultdict, namedtuple
import logging
from meld.util import log_timing
import sys

logger = logging.getLogger(__name__)


# setup exception handling to abort when there is unhandled exception
sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    get_mpi_comm_world().Abort(1)


sys.excepthook = mpi_excepthook


class MPICommunicator(object):
    """
    Class to handle communications between master and slaves using MPI.
    
    :param n_atoms: number of atoms
    :param n_replicas: number of replicas

    .. note::
        creating an MPI communicator will not actually initialize MPI. To do that,
        call :meth:`initialize`.
    """

    def __init__(self, n_atoms, n_replicas):
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

        :returns: :const:`True` if we are the master, otherwise :const:`False`

        """
        if self._my_rank == 0:
            return True
        else:
            return False

    @log_timing(logger)
    def barrier(self):
        self._mpi_comm.barrier()

    @log_timing(logger)
    def broadcast_alphas_to_slaves(self, alphas):
        """
        broadcast_alphas_to_slaves(alphas)
        Send the alpha values to the slaves.

        :param alphas: a list of alpha values, one for each replica.
            The master node's alpha value should be included in this list.
            The master node will always be at alpha=0.0
        :returns: :const:`None`

        """
        self._mpi_comm.scatter(alphas, root=0)

    @log_timing(logger)
    def receive_alpha_from_master(self):
        """
        receive_alpha_from_master()
        Receive alpha value from master node.

        :returns: a floating point value for alpha in ``[0,1]``

        """
        return self._mpi_comm.scatter(None, root=0)

    @log_timing(logger)
    def broadcast_states_to_slaves(self, states):
        """
        broadcast_states_to_slaves(states)
        Send a state to each slave.

        :param states: a list of states. The list of states should include
            the state for the master node. These are the states that will
            be simulated on each replica for each step.
        :returns: the state to run on the master node

        """
        return self._mpi_comm.scatter(states, root=0)

    @log_timing(logger)
    def receive_state_from_master(self):
        """
        receive_state_from_master()
        Get state to run for this step

        :returns: the state to run for this step

        """
        return self._mpi_comm.scatter(None, root=0)

    @log_timing(logger)
    def gather_states_from_slaves(self, state_on_master):
        """
        gather_states_from_slaves(state_on_master)
        Receive states from all slaves

        :param state_on_master: the state on the master after simulating
        :returns: A list of states, one from each replica.
                  The returned states are the states after simulating.

        """
        return self._mpi_comm.gather(state_on_master, root=0)

    @log_timing(logger)
    def send_state_to_master(self, state):
        """
        send_state_to_master(state)
        Send state to master

        :param state: State to send to master. This is the state after
                      simulating this step.
        :returns: :const:`None`

        """
        self._mpi_comm.gather(state, root=0)

    @log_timing(logger)
    def broadcast_states_for_energy_calc_to_slaves(self, states):
        """
        broadcast_states_for_energy_calc_to_slaves(states)
        Broadcast states to all slaves. Send all results from this step to every
        slave so that we can calculate the energies and do replica exchange.

        :param states: a list of states
        :returns: :const:`None`

        """
        self._mpi_comm.bcast(states, root=0)

    @log_timing(logger)
    def exchange_states_for_energy_calc(self, state):
        """
        exchange_states_for_energy_calc(state)
        Exchange states between all processes.

        :param state: the state for this node
        :returns: a list of states from all nodes

        """
        return self._mpi_comm.allgather(state)

    @log_timing(logger)
    def receive_states_for_energy_calc_from_master(self):
        """
        receive_states_for_energy_calc_from_master()
        Receive all states from master.

        :returns: a list of states to calculate the energy of

        """
        return self._mpi_comm.bcast(None, root=0)

    @log_timing(logger)
    def gather_energies_from_slaves(self, energies_on_master):
        """
        gather_energies_from_slaves(energies_on_master)
        Receive a list of energies from each slave.

        :param energies_on_master: a list of energies from the master
        :returns: a square matrix of every state on every replica to be used
                  for replica exchange

        """
        energies = self._mpi_comm.gather(energies_on_master, root=0)
        return np.array(energies)

    @log_timing(logger)
    def send_energies_to_master(self, energies):
        """
        send_energies_to_master(energies)
        Send a list of energies to the master.

        :param energies: a list of energies to send to the master
        :returns: :const:`None`

        """
        return self._mpi_comm.gather(energies, root=0)

    @log_timing(logger)
    def negotiate_device_id(self):
        hostname = platform.node()
        try:
            visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
            logger.debug('%s found cuda devices: %s', hostname, visible_devices)
            visible_devices = visible_devices.split(',')
            if visible_devices:
                visible_devices = [int(dev) for dev in visible_devices]
            else:
                raise RuntimeError('No cuda devices available')
        except KeyError:
            logger.debug('%s CUDA_VISIBLE_DEVICES is not set.', hostname)
            visible_devices = None

        hosts = self._mpi_comm.gather(HostInfo(hostname, visible_devices), root=0)

        # the master computes the device ids
        if self._my_rank == 0:
            if hosts[0].devices is None:
                # if CUDA_VISIBLE_DEVICES isn't set on the master, we assume it
                # isn't set for any node

                # create an empty default dict to count hosts
                host_counts = defaultdict(int)

                # list of device ids
                # this assumes that available devices for each node
                # are numbered starting from 0
                device_ids = []
                for host in hosts:
                    assert host.devices is None
                    device_ids.append(host_counts[host.host_name])
                    host_counts[host.host_name] += 1
            else:
                # CUDA_VISIBLE_DEVICES is set on the master, so we
                # assume it is set for all nodes

                # create a dict to hold the device ids available on each host
                available_devices = {}
                # store the available devices on each node
                for host in hosts:
                    if host.host_name in available_devices:
                        assert host.devices == available_devices[host.host_name]
                    else:
                        available_devices[host.host_name] = host.devices

                # CUDA numbers the devices from 0 always,
                # e.g. if CUDA_VISIBLE_DEVICES=2,3 we still need to ask for
                # devices 0 and 1 to get physical devices 2 and 3.
                # So, we subtract the minimum value from each each to make it zero
                for host in hosts:
                    min_device_id = min(available_devices[host.host_name])
                    available_devices[host.host_name] = [device_id - min_device_id for device_id in available_devices[host.host_name]]

                # device ids for each node
                device_ids = []
                for host in hosts:
                    # pop off the first device_id for this host name
                    device_ids.append(available_devices[host.host_name].pop(0))

        # receive device id from master
        else:
            device_ids = None
        # do the communication
        device_id = self._mpi_comm.scatter(device_ids, root=0)
        logger.debug('hostname: %s, device_id: %d', hostname, device_id)
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


# namedtuple to hold results for negotiate id
HostInfo = namedtuple('HostInfo', 'host_name devices')
