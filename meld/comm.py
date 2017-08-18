#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import signal
import threading
import time
import os
import six
import numpy as np
import platform
from collections import defaultdict, namedtuple
import contextlib
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
        creating an MPI communicator will not actually initialize MPI.
        To do that, call :meth:`initialize`.
    """

    def __init__(self, n_atoms, n_replicas, timeout=600):
        # We're not using n_atoms and n_replicas, but if we switch
        # to more efficient buffer-based MPI routines, we'll need them.
        self._n_atoms = n_atoms
        self._n_replicas = n_replicas
        self._mpi_comm = None
        self._timeout = timeout
        self._timeout_message = 'Call to {{:s}} did not complete in {:d} seconds'.format(timeout)

    def __getstate__(self):
        # don't pickle _mpi_comm
        return dict((k, v) for (k, v) in six.iteritems(self.__dict__)
                    if not k == '_mpi_comm')

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
        with timeout(self._timeout,
                     RuntimeError(self._timeout_message.format('barrier'))):
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
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('broadcast_alphas_to_slaves'))):
            self._mpi_comm.scatter(alphas, root=0)

    @log_timing(logger)
    def broadcast_logger_address_to_slaves(self, address):
        """
        Broadcast the hostname and port of the logger to slaves.

        :param address: a tuple (hostname, port)
        :return: :const: `None`
        """
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('broadcast_logger_address_to_slaves'))):
            self._mpi_comm.bcast(address, root=0)

    @log_timing(logger)
    def receive_logger_address_from_master(self):
        """
        Receive the hostname and port of the logger from the master

        :return: a (hostname, port) tuple
        """
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('receive_logger_address_from_master'))):
            return self._mpi_comm.bcast(None, root=0)

    @log_timing(logger)
    def receive_alpha_from_master(self):
        """
        receive_alpha_from_master()
        Receive alpha value from master node.

        :returns: a floating point value for alpha in ``[0,1]``

        """
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('receive_alpha_from_master'))):
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
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('broadcast_states_to_slaves'))):
            return self._mpi_comm.scatter(states, root=0)

    @log_timing(logger)
    def receive_state_from_master(self):
        """
        receive_state_from_master()
        Get state to run for this step

        :returns: the state to run for this step

        """
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('receive_state_from_master'))):
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
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('gather_states_from_slaves'))):
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
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('send_state_to_master'))):
            self._mpi_comm.gather(state, root=0)

    @log_timing(logger)
    def broadcast_states_for_energy_calc_to_slaves(self, states):
        """
        broadcast_states_for_energy_calc_to_slaves(states)
        Broadcast states to all slaves. Send all results from this step
        to every slave so that we can calculate the energies and do
        replica exchange.

        :param states: a list of states
        :returns: :const:`None`

        """
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('broadcast_states_for_energy_calc_to_slaves'))):
            self._mpi_comm.bcast(states, root=0)

    @log_timing(logger)
    def exchange_states_for_energy_calc(self, state):
        """
        exchange_states_for_energy_calc(state)
        Exchange states between all processes.

        :param state: the state for this node
        :returns: a list of states from all nodes

        """
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('exchange_states_for_energy_calc'))):
            return self._mpi_comm.allgather(state)

    @log_timing(logger)
    def receive_states_for_energy_calc_from_master(self):
        """
        receive_states_for_energy_calc_from_master()
        Receive all states from master.

        :returns: a list of states to calculate the energy of

        """
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('receive_states_for_energy_calc_from_master'))):
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
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('gather_energies_from_slaves'))):
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
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('send_energies_to_master'))):
            return self._mpi_comm.gather(energies, root=0)

    @log_timing(logger)
    def negotiate_device_id(self):
        with timeout(self._timeout,
                     RuntimeError(
                         self._timeout_message.format('negotiate_device_id'))):
            hostname = platform.node()
            try:
                visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
                logger.info('%s found cuda devices: %s', hostname, visible_devices)
                visible_devices = visible_devices.split(',')
                if visible_devices:
                    visible_devices = [int(dev) for dev in visible_devices]
                else:
                    raise RuntimeError('No cuda devices available')
            except KeyError:
                logger.info('%s CUDA_VISIBLE_DEVICES is not set.', hostname)
                visible_devices = None

            hosts = self._mpi_comm.gather(
                HostInfo(hostname, visible_devices), root=0)

            # the master computes the device ids
            if self._my_rank == 0:
                if hosts[0].devices is None:
                    # if CUDA_VISIBLE_DEVICES isn't set on the master, we assume it
                    # isn't set for any node
                    logger.info('CUDA_VISIBLE_DEVICES is not set.')
                    logger.info('Assuming each mpi process has access')
                    logger.info('to a CUDA device, where the device')
                    logger.info('numbering starts from 0.')

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
                    logger.info('CUDA_VISIBLE_DEVICES is set.')

                    # create a dict to hold the device ids available on each host
                    available_devices = {}
                    # store the available devices on each node
                    for host in hosts:
                        if host.host_name in available_devices:
                            if host.devices != available_devices[host.host_name]:
                                raise RuntimeError(
                                    'GPU devices for host do not match')
                        else:
                            available_devices[host.host_name] = host.devices

                    # CUDA numbers the devices from 0 always,
                    # e.g. if CUDA_VISIBLE_DEVICES=2,3 we still need to ask for
                    # devices 0 and 1 to get physical devices 2 and 3.
                    # So, we subtract the minimum value from each each to make
                    # it zero
                    # but we don't do this if the device ids are set to -1, which
                    # allows openmm to choose the gpu
                    for host in hosts:
                        min_device_id = min(available_devices[host.host_name])
                        if min_device_id != -1:
                            available_devices[host.host_name] = [
                                device_id - min_device_id for device_id in
                                available_devices[host.host_name]]

                    # device ids for each node
                    device_ids = []
                    for host in hosts:
                        try:
                            # pop off the first device_id for this host name
                            device_ids.append(available_devices[host.host_name].pop(0))
                        except IndexError:
                            logger.error('More mpi processes than GPUs')
                            raise RuntimeError('More mpi process than GPUs')

            # receive device id from master
            else:
                device_ids = None

            # do the communication
            device_id = self._mpi_comm.scatter(device_ids, root=0)
            logger.info('hostname: %s, device_id: %d', hostname, device_id)
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


# Adapted from interrupting cow
# https://bitbucket.org/evzijst/interruptingcow
#
# Original license below
#
# The MIT License (MIT)
#
# Copyright (c) 2012 Erik van Zijst <erik.van.zijst@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class StateException(Exception):
    pass


class Quota(object):
    def __init__(self, seconds):
        if seconds <= 0:
            raise ValueError('Invalid timeout: %s' % seconds)
        else:
            self._timeleft = seconds
        self._depth = 0
        self._starttime = None

    def __str__(self):
        return '<Quota remaining=%s>' % self.remaining()

    def _start(self):
        if self._depth is 0:
            self._starttime = time.time()
        self._depth += 1

    def _stop(self):
        if self._depth is 1:
            self._timeleft = self.remaining()
            self._starttime = None
        self._depth -= 1

    def running(self):
        return self._depth > 0

    def remaining(self):
        if self.running():
            return max(self._timeleft - (time.time() - self._starttime), 0)
        else:
            return max(self._timeleft, 0)

def _bootstrap():
    Timer = namedtuple('Timer', 'expiration exception')
    timers = []

    def handler(*args):
        exception = timers.pop().exception
        if timers:
            timeleft = timers[-1].expiration - time.time()
            if timeleft > 0:
                signal.setitimer(signal.ITIMER_REAL, timeleft)
            else:
                handler(*args)

        raise exception

    def set_sighandler():
        current = signal.getsignal(signal.SIGALRM)
        if current == signal.SIG_DFL:
            signal.signal(signal.SIGALRM, handler)
        elif current != handler:
            raise StateException('Your process alarm handler is already in '
                                 'use! Interruptingcow cannot be used in '
                                 'programs that use SIGALRM.')

    def timeout(seconds, exception):
        if threading.currentThread().name != 'MainThread':
            raise StateException('Interruptingcow can only be used from the '
                                 'MainThread.')
        if isinstance(seconds, Quota):
            quota = seconds
        else:
            quota = Quota(float(seconds))
        set_sighandler()
        seconds = quota.remaining()

        depth = len(timers)
        parenttimeleft = signal.getitimer(signal.ITIMER_REAL)[0]
        if not timers or parenttimeleft > seconds:
            try:
                quota._start()
                timers.append(Timer(time.time() + seconds, exception))
                if seconds > 0:
                    signal.setitimer(signal.ITIMER_REAL, seconds)
                    yield
                else:
                    handler()
            finally:
                quota._stop()
                if len(timers) > depth:
                    # cancel our timer
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    timers.pop()
                    if timers:
                        # reinstall the parent timer
                        parenttimeleft = timers[-1].expiration - time.time()
                        if parenttimeleft > 0:
                            signal.setitimer(signal.ITIMER_REAL, parenttimeleft)
                        else:
                            # the parent timer has expired, trigger the handler
                            handler()
        else:
            # not enough time left on the parent timer
            try:
                quota._start()
                yield
            finally:
                quota._stop()

    @contextlib.contextmanager
    def timeout_context_manager(seconds, exception):
        t = timeout(seconds, exception)
        next(t)
        yield
        next(t)

    return timeout_context_manager

timeout = _bootstrap()
