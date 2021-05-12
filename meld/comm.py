#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

try:
    from mpi4py import MPI  # type: ignore
except ImportError:
    print()
    print("****")
    print("Error importing mpi4py.")
    print()
    print("Meld depends on mpi4py, but does not automatically install it")
    print(
        "as a dependency. See https://github.com/maccallumlab/meld/blob/master/README.md"
    )
    print("for details.")
    print("****")
    print()
    raise

import signal
import threading
import time
import os
import numpy as np  # type: ignore
import platform
from collections import defaultdict, namedtuple
import contextlib
import logging
import sys
from meld.util import log_timing
from meld.system.state import SystemState
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


# setup exception handling to abort when there is unhandled exception
sys_excepthook = sys.excepthook


def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    rank = get_mpi_comm_world().rank + 1
    size = get_mpi_comm_world().size
    node_name = f"{rank}/{size}"
    logger.critical(f"MPI node {node_name} raised exception.")
    sys.stdout.flush()
    sys.stderr.flush()
    get_mpi_comm_world().Abort(1)


sys.excepthook = mpi_excepthook


class MPICommunicator:
    """
    Class to handle communications between leader and followers using MPI.

    :param n_atoms: number of atoms
    :param n_replicas: number of replicas

    .. note::
        creating an MPI communicator will not actually initialize MPI.
        To do that, call :meth:`initialize`.
    """

    _mpi_comm: MPI.Comm

    def __init__(self, n_atoms: int, n_replicas: int, timeout: int = 600) -> None:
        # We're not using n_atoms and n_replicas, but if we switch
        # to more efficient buffer-based MPI routines, we'll need them.
        self._n_atoms = n_atoms
        self._n_replicas = n_replicas
        self._timeout = timeout
        self._timeout_message = f"Call to {{:s}} did not complete in {timeout} seconds"

    def __getstate__(self) -> Dict[str, Any]:
        # don't pickle _mpi_comm
        return dict((k, v) for (k, v) in self.__dict__.items() if not k == "_mpi_comm")

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # set _mpi_comm to None
        self.__dict__ = state

    def initialize(self) -> None:
        """
        Initialize and start MPI

        """
        self._mpi_comm = get_mpi_comm_world()
        self._my_rank = self._mpi_comm.Get_rank()

    def is_leader(self) -> bool:
        """
        Is this the leader node?

        :returns: :const:`True` if we are the leader, otherwise :const:`False`

        """
        if self._my_rank == 0:
            return True
        else:
            return False

    @log_timing(logger)
    def barrier(self) -> None:
        with timeout(
            self._timeout, RuntimeError(self._timeout_message.format("barrier"))
        ):
            self._mpi_comm.barrier()

    @log_timing(logger)
    def broadcast_alphas_to_followers(self, alphas: List[float]) -> None:
        """
        broadcast_alphas_to_followers(alphas)
        Send the alpha values to the followers.

        :param alphas: a list of alpha values, one for each replica.
            The leader's alpha value should be included in this list.
            The leader's node will always be at alpha=0.0
        :returns: :const:`None`

        """
        with timeout(
            self._timeout,
            RuntimeError(self._timeout_message.format("broadcast_alphas_to_followers")),
        ):
            self._mpi_comm.scatter(alphas, root=0)

    @log_timing(logger)
    def broadcast_logger_address_to_followers(self, address: Tuple[str, int]) -> None:
        """
        Broadcast the hostname and port of the logger to followers.

        :param address: a tuple (hostname, port)
        :return: :const: `None`
        """
        with timeout(
            self._timeout,
            RuntimeError(
                self._timeout_message.format("broadcast_logger_address_to_followers")
            ),
        ):
            self._mpi_comm.bcast(address, root=0)

    @log_timing(logger)
    def receive_logger_address_from_leader(self) -> Tuple[str, int]:
        """
        Receive the hostname and port of the logger from the leader

        :return: a (hostname, port) tuple
        """
        with timeout(
            self._timeout,
            RuntimeError(
                self._timeout_message.format("receive_logger_address_from_leader")
            ),
        ):
            return self._mpi_comm.bcast(None, root=0)

    @log_timing(logger)
    def receive_alpha_from_leader(self) -> float:
        """
        receive_alpha_from_leader()
        Receive alpha value from leader node.

        :returns: a floating point value for alpha in ``[0,1]``

        """
        with timeout(
            self._timeout,
            RuntimeError(self._timeout_message.format("receive_alpha_from_leader")),
        ):
            return self._mpi_comm.scatter(None, root=0)

    @log_timing(logger)
    def broadcast_states_to_followers(self, states: List[SystemState]) -> SystemState:
        """
        broadcast_states_to_followers(states)
        Send a state to each follower.

        :param states: a list of states. The list of states should include
            the state for the leader node. These are the states that will
            be simulated on each replica for each step.
        :returns: the state to run on the leader node

        """
        with timeout(
            self._timeout,
            RuntimeError(self._timeout_message.format("broadcast_states_to_followers")),
        ):
            return self._mpi_comm.scatter(states, root=0)

    @log_timing(logger)
    def receive_state_from_leader(self) -> SystemState:
        """
        receive_state_from_leader()
        Get state to run for this step

        :returns: the state to run for this step

        """
        with timeout(
            self._timeout,
            RuntimeError(self._timeout_message.format("receive_state_from_leader")),
        ):
            return self._mpi_comm.scatter(None, root=0)

    @log_timing(logger)
    def gather_states_from_followers(
        self, state_on_leader: SystemState
    ) -> List[SystemState]:
        """
        gather_states_from_followers(state_on_leader)
        Receive states from all followers

        :param state_on_leader: the state on the leader after simulating
        :returns: A list of states, one from each replica.
                  The returned states are the states after simulating.

        """
        with timeout(
            self._timeout,
            RuntimeError(self._timeout_message.format("gather_states_from_followers")),
        ):
            return self._mpi_comm.gather(state_on_leader, root=0)

    @log_timing(logger)
    def send_state_to_leader(self, state: SystemState) -> None:
        """
        send_state_to_leader(state)
        Send state to leader

        :param state: State to send to leader. This is the state after
                      simulating this step.
        :returns: :const:`None`

        """
        with timeout(
            self._timeout,
            RuntimeError(self._timeout_message.format("send_state_to_leader")),
        ):
            self._mpi_comm.gather(state, root=0)

    @log_timing(logger)
    def broadcast_states_for_energy_calc_to_followers(
        self, states: List[SystemState]
    ) -> None:
        """
        broadcast_states_for_energy_calc_to_followers(states)
        Broadcast states to all followers. Send all results from this step
        to every follower so that we can calculate the energies and do
        replica exchange.

        :param states: a list of states
        :returns: :const:`None`

        """
        with timeout(
            self._timeout,
            RuntimeError(
                self._timeout_message.format(
                    "broadcast_states_for_energy_calc_to_followers"
                )
            ),
        ):
            self._mpi_comm.bcast(states, root=0)

    @log_timing(logger)
    def exchange_states_for_energy_calc(self, state: SystemState) -> List[SystemState]:
        """
        exchange_states_for_energy_calc(state)
        Exchange states between all processes.

        :param state: the state for this node
        :returns: a list of states from all nodes

        """
        with timeout(
            self._timeout,
            RuntimeError(
                self._timeout_message.format("exchange_states_for_energy_calc")
            ),
        ):
            return self._mpi_comm.allgather(state)

    @log_timing(logger)
    def receive_states_for_energy_calc_from_leader(self) -> List[SystemState]:
        """
        receive_states_for_energy_calc_from_leader()
        Receive all states from leader.

        :returns: a list of states to calculate the energy of

        """
        with timeout(
            self._timeout,
            RuntimeError(
                self._timeout_message.format(
                    "receive_states_for_energy_calc_from_leader"
                )
            ),
        ):
            return self._mpi_comm.bcast(None, root=0)

    @log_timing(logger)
    def gather_energies_from_followers(
        self, energies_on_leader: List[float]
    ) -> np.ndarray:
        """
        gather_energies_from_followers(energies_on_leader)
        Receive a list of energies from each follower.

        :param energies_on_leader: a list of energies from the leader
        :returns: a square matrix of every state on every replica to be used
                  for replica exchange

        """
        with timeout(
            self._timeout,
            RuntimeError(
                self._timeout_message.format("gather_energies_from_followers")
            ),
        ):
            energies = self._mpi_comm.gather(energies_on_leader, root=0)
            return np.array(energies)

    @log_timing(logger)
    def send_energies_to_leader(self, energies: List[float]) -> None:
        """
        send_energies_to_leader(energies)
        Send a list of energies to the leader.

        :param energies: a list of energies to send to the leader
        :returns: :const:`None`

        """
        with timeout(
            self._timeout,
            RuntimeError(self._timeout_message.format("send_energies_to_leader")),
        ):
            return self._mpi_comm.gather(energies, root=0)

    @log_timing(logger)
    def negotiate_device_id(self) -> int:
        with timeout(
            self._timeout,
            RuntimeError(self._timeout_message.format("negotiate_device_id")),
        ):
            hostname = platform.node()
            try:
                env_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
                logger.info("%s found cuda devices: %s", hostname, env_visible_devices)
                visible_devices: Optional[List[int]] = [
                    int(dev) for dev in env_visible_devices.split(",")
                ]
                if not visible_devices:
                    raise RuntimeError("No cuda devices available")
            except KeyError:
                logger.info("%s CUDA_VISIBLE_DEVICES is not set.", hostname)
                visible_devices = None

            hosts = self._mpi_comm.gather(HostInfo(hostname, visible_devices), root=0)

            # the leader computes the device ids
            if self._my_rank == 0:
                if hosts[0].devices is None:
                    # if CUDA_VISIBLE_DEVICES isn't set on the leader, we assume it
                    # isn't set for any node
                    logger.info("CUDA_VISIBLE_DEVICES is not set.")
                    logger.info("Assuming each mpi process has access")
                    logger.info("to a CUDA device, where the device")
                    logger.info("numbering starts from 0.")

                    # create an empty default dict to count hosts
                    host_counts: Dict[str, int] = defaultdict(int)

                    # list of device ids
                    # this assumes that available devices for each node
                    # are numbered starting from 0
                    device_ids = []
                    for host in hosts:
                        assert host.devices is None
                        device_ids.append(host_counts[host.host_name])
                        host_counts[host.host_name] += 1
                else:
                    # CUDA_VISIBLE_DEVICES is set on the leader, so we
                    # assume it is set for all nodes
                    logger.info("CUDA_VISIBLE_DEVICES is set.")

                    # create a dict to hold the device ids available on each host
                    available_devices: Dict[str, List[int]] = {}
                    # store the available devices on each node
                    for host in hosts:
                        if host.host_name in available_devices:
                            if host.devices != available_devices[host.host_name]:
                                raise RuntimeError("GPU devices for host do not match")
                        else:
                            available_devices[host.host_name] = host.devices

                    # CUDA numbers the devices contiguously, starting from zero.
                    # For example, if `CUDA_VISIBLE_DEVICES=2,4,5`, we would
                    # access these as ids 0, 1, 2.
                    available_devices = {
                        host_name: list(range(len(devices)))
                        for host_name, devices in available_devices.items()
                    }

                    # device ids for each node
                    device_ids = []
                    for host in hosts:
                        try:
                            # pop off the first device_id for this host name
                            device_ids.append(available_devices[host.host_name].pop(0))
                        except IndexError:
                            logger.error("More mpi processes than GPUs")
                            raise RuntimeError("More mpi process than GPUs")

            # receive device id from leader
            else:
                device_ids = []

            # do the communication
            device_id = self._mpi_comm.scatter(
                device_ids if device_ids else None, root=0
            )
            logger.info("hostname: %s, device_id: %d", hostname, device_id)
            return device_id

    @property
    def n_replicas(self) -> int:
        return self._n_replicas

    @property
    def n_atoms(self) -> int:
        return self._n_atoms

    @property
    def rank(self) -> int:
        return self._my_rank


def get_mpi_comm_world() -> MPI.Comm:
    """
    Helper function to return the comm_world.

    """
    return MPI.COMM_WORLD


# namedtuple to hold results for negotiate id
HostInfo = namedtuple("HostInfo", "host_name devices")


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


class Quota:
    def __init__(self, seconds):
        if seconds <= 0:
            raise ValueError("Invalid timeout: %s" % seconds)
        else:
            self._timeleft = seconds
        self._depth = 0
        self._starttime = None

    def __str__(self):
        return "<Quota remaining=%s>" % self.remaining()

    def _start(self):
        if self._depth == 0:
            self._starttime = time.time()
        self._depth += 1

    def _stop(self):
        if self._depth == 1:
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
    Timer = namedtuple("Timer", "expiration exception")
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
            raise StateException(
                "Your process alarm handler is already in "
                "use! Interruptingcow cannot be used in "
                "programs that use SIGALRM."
            )

    def timeout(seconds, exception):
        if threading.currentThread().name != "MainThread":
            raise StateException(
                "Interruptingcow can only be used from the " "MainThread."
            )
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

    return timeout_context_manager


timeout = _bootstrap()
