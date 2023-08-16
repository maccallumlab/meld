"""
Helper functions to simplify setting up a MELD calculation.
"""

from inspect import istraceback
from meld.remd.adaptor import AdaptationPolicy, EqualAcceptanceAdaptor
from meld.remd.ladder import NearestNeighborLadder
from meld.remd.leader import LeaderReplicaExchangeRunner
from meld.comm import MPICommunicator
from meld.vault import DataStore
from meld.interfaces import IState, ISystem
from meld.system.options import RunOptions
from typing import Optional, NamedTuple


class REMDInfo(NamedTuple):
    remd_runner: LeaderReplicaExchangeRunner
    communicator: MPICommunicator
    n_replicas: int


def setup_replica_exchange(
    system: ISystem,
    n_replicas: int,
    n_steps: int,
    n_trials: Optional[int] = None,
    adaptation_growth_factor: float = 2.0,
    adaptation_burn_in: int = 50,
    adaptation_adapt_every: int = 50,
    adaptation_stop_after: Optional[int] = None,
    adaptation_min_acc_prob: float = 0.02,
    mpi_timeout: int = 60_000,
) -> REMDInfo:
    """
    Setup replica exchange

    Args:
        system: The system to be simulated
        n_replicas: The number of replicas to be simulated
        n_steps: The number of steps of replica exchange to run
        n_trials: The number of trials to run per exchange
        adaptation_growth_factor: The growth factor for adaptation
        adaptation_burn_in: The number of steps to ignore after adapting
        adaptation_adapt_every: The number of steps to run between adaptation
        adaptation_stop_after: The number of steps to run before stopping adaptation
        adaptation_min_acc_prob: The minimum acceptance probability when adapting
        mpi_timeout: The number of seconds to wait for MPI communication to complete

    Note:
        Replica exchange adaptation works as follows. First `adaptation_burn_in` steps
        are run and no statistics about exchange are collected. Then `adaptation_adapt_every`
        steps are run and the average acceptance rate is calculated. Any acceptance rates
        below `adaptation_min_acc_prob` are clamped at the minimum. Then adaptation
        is performed to equalize the acceptance rates. After each adaptation, the values
        of `adaptation_burn_in` and `adaptation_adapt_every` are multiplied by
        `adaptation_growth_factor`.

    Note:
        If `n_trials` is `None`, then the number of trials is `n_replicas**2`.
    """
    if n_trials is None:
        n_trials = n_replicas * n_replicas

    ladder = NearestNeighborLadder(n_trials=n_trials)
    policy = AdaptationPolicy(
        growth_factor=adaptation_growth_factor,
        burn_in=adaptation_burn_in,
        adapt_every=adaptation_adapt_every,
        stop_after=adaptation_stop_after,
    )
    adaptor = EqualAcceptanceAdaptor(
        n_replicas=n_replicas,
        adaptation_policy=policy,
        min_acc_prob=adaptation_min_acc_prob,
    )

    remd_runner = LeaderReplicaExchangeRunner(
        n_replicas=n_replicas,
        max_steps=n_steps,
        ladder=ladder,
        adaptor=adaptor,
    )

    communicator = MPICommunicator(
        system.n_atoms, n_replicas=n_replicas, timeout=mpi_timeout
    )

    return REMDInfo(remd_runner, communicator, n_replicas)


def setup_data_store(
    system: ISystem, run_options: RunOptions, remd_info: REMDInfo, block_size: int = 50
):
    """
    Setup the data store

    Args:
        system: The system to be simulated
        run_options: The run options
        remd_info: The replica exchange information
        block_size: The number of steps to store per block
    """
    store = DataStore(
        state_template=system.get_state_template(),
        n_replicas=remd_info.n_replicas,
        pdb_writer=system.get_pdb_writer(),
        block_size=block_size,
    )
    store.initialize(mode="w")
    store.save_system(system)
    store.save_run_options(run_options)
    store.save_remd_runner(remd_info.remd_runner)
    store.save_communicator(remd_info.communicator)
    if run_options.enable_gamd == True:
        integrator = system.integrator.__dict__.copy()
        del integrator["this"]
        store.save_integrator(integrator)

    def _setup_state(index):
        state = system.get_state_template()
        state.alpha = index / (remd_info.n_replicas - 1.0)
        return state

    states = [_setup_state(i) for i in range(remd_info.n_replicas)]
    store.save_states(states, 0)
    store.save_data_store()
