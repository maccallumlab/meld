"""
GaMELD Implementation Module. 

GaMELD combines Gaussian Accelerated Molecular Dynamics (GaMD) and MELD using a new threshold calculation.

References:
    M. Caparotta, A. Perez, When MELD meets GaMD: Accelerating Biomolecular Landscape Exploration.
"""

import logging
from meld import interfaces
import numpy as np
from typing import List
import os

logger = logging.getLogger(__name__)

def change_thresholds(
    step: int,
    system_runner,
    communicator: interfaces.ICommunicator,
    leader: bool,
) -> None:
    """
    Change energy thresholds after cMD and GaMELD Equilibration
    FIXED: Handles interleaved log files, per-replica thresholds, and per-replica k0
    
    Args:
        step: current step
        system_runner: a interfaces.IRunner object to run the simulations
        communicator: a communicator object to talk with workers
        leader: leader (True) or worker (False) indicator
    """

    # Check if it's time to change thresholds
    initial_row: int = 0
    if step == (system_runner._integrator.ntcmd / system_runner._options.timesteps):
        initial_row = int(
            system_runner._integrator.ntcmdprep / system_runner._options.timesteps
        )
        logger.info(f"GaMD threshold update at step {step} (after cMD)")
    elif step == (
        (system_runner._integrator.ntcmd + system_runner._integrator.nteb)
        / system_runner._options.timesteps
    ):
        initial_row = int(
            (system_runner._integrator.ntcmd + system_runner._integrator.ntebprep)
            / system_runner._options.timesteps
        )
        logger.info(f"GaMD threshold update at step {step} (after GaMD equilibration)")

    if initial_row != 0:
        n_replicas = communicator.n_replicas
        n_workers = communicator.n_workers
        replicas_per_worker = n_replicas // n_workers
        worker_rank = communicator.rank
        
        # Determine which replicas this worker handles
        my_replica_indices = list(range(
            worker_rank * replicas_per_worker,
            (worker_rank + 1) * replicas_per_worker
        ))
        
        logger.info(f"Worker {worker_rank} processing {len(my_replica_indices)} replicas")
        
        # Log file for this worker
        gamd_log_filename = os.path.join("Logs", f"gamd_{worker_rank:03d}.log")
        
        if not os.path.exists(gamd_log_filename):
            logger.warning(f"Log file not found: {gamd_log_filename}")
            return
        
        # Collect replica data using helper function
        my_replica_data = collect_replica_data(
            gamd_log_filename, initial_row, my_replica_indices, n_replicas, system_runner
        )
        
        # Gather data from all workers
        if leader:
            if n_workers > 1:
                all_workers_data = communicator.gather_thresholds_from_workers(my_replica_data)
                all_replica_data = []
                for worker_data in all_workers_data:
                    all_replica_data.extend(worker_data)
            else:
                all_replica_data = my_replica_data
            
            all_replica_data.sort(key=lambda x: x[0])
            
            logger.info(f"Collected data from {len(all_replica_data)} replicas")
            
            # Calculate thresholds and k0 values using helper function
            calculate_and_store_gamd_parameters(
                all_replica_data, system_runner
            )
            
            # For multi-process mode, distribute and apply
            if n_workers > 1:
                distribute_and_apply_parameters(
                    all_replica_data, system_runner, communicator,
                    replicas_per_worker, worker_rank, my_replica_indices
                )
            
        else:
            # Worker: send data to leader and receive thresholds
            communicator.send_thresholds_to_leader(my_replica_data)
            my_new_thresholds = communicator.receive_thresholds_from_leader()
            
            # Store for this worker's replicas
            if not hasattr(system_runner, '_gamd_replica_thresholds'):
                system_runner._gamd_replica_thresholds = {}
            
            for i, replica_idx in enumerate(my_replica_indices):
                if i < len(my_new_thresholds):
                    system_runner._gamd_replica_thresholds[replica_idx] = my_new_thresholds[i]
            
            # Apply threshold (for multi-process mode with 1 replica per worker)
            if replicas_per_worker == 1:
                system_runner._simulation.integrator.setGlobalVariableByName(
                    "threshold_energy_Total", my_new_thresholds[0]
                )
                
def compute_energy_width(
    gamd_log_filename: str, initial_row: int, column: int
) -> float:
    """ 
    Compute standard deviation of the potential energies.
    """
    
    energy: List[float] = []
    with open(gamd_log_filename, "r") as file:
        for line in file:
            line_split = line.split()
            if not line_split[0] == "#":
                if int(line_split[1]) > initial_row:
                    energy.append(float(line_split[column]) * 4.184)  # Unboosted-Energy
    return float(np.std(energy))


def new_thresholds(thresholds: List[List[float]]) -> List[float]:
    """ 
    Compute new thresholds.
    """

    new_threshold: List[float] = []
    sd_acum: float = 0
    thresholds.reverse()
    for threshold in range(len(thresholds)):
        if threshold == 0:
            new_threshold.insert(0, thresholds[threshold][0])
        else:
            sd_acum = sd_acum + thresholds[threshold][1]
            new_threshold.insert(threshold, new_threshold[0] - sd_acum)
    new_threshold.reverse()
    return new_threshold

# ============================================================================
# HELPER FUNCTIONS FOR PER-REPLICA GaMD PARAMETERS
# Added to fix interleaved log bug and enable per-replica k0 values
# ============================================================================

def compute_energy_width_per_replica(
    gamd_log_filename: str,
    initial_row: int,
    column: int,
    replica_idx: int,
    n_replicas: int
) -> float:
    """
    Compute standard deviation of potential energies for a SPECIFIC replica.
    Handles interleaved log format (mpirun -n 1) and single-replica logs.
    
    Args:
        gamd_log_filename: Path to GaMD log file
        initial_row: Starting step number (exclude equilibration)
        column: Which column to read (2=Total, 3=Dihedral)
        replica_idx: Which replica to extract (0-indexed)
        n_replicas: Total number of replicas (for interleaving)
    
    Returns:
        Standard deviation of energies for this replica
    """
    energy: List[float] = []
    line_count = 0
    
    with open(gamd_log_filename, "r") as file:
        for line in file:
            line_split = line.split()
            
            if not line_split or line_split[0] == "#":
                continue
            
            step = int(line_split[1])
            
            if step > initial_row:
                current_replica = line_count % n_replicas
                
                if current_replica == replica_idx:
                    energy.append(float(line_split[column]))
            
            line_count += 1
    
    if len(energy) == 0:
        logger.warning(f"No energy data found for replica {replica_idx} after step {initial_row}")
        return 0.0
    
    return float(np.std(energy))


def get_energy_stats_per_replica(
    gamd_log_filename: str,
    initial_row: int,
    column: int,
    replica_idx: int,
    n_replicas: int
) -> tuple:
    """
    Get Vmax, Vmin, Vavg for a specific replica from interleaved log.
    
    Args:
        gamd_log_filename: Path to GaMD log file
        initial_row: Starting step number
        column: Which column to read (2=Total, 3=Dihedral)
        replica_idx: Which replica to extract (0-indexed)
        n_replicas: Total number of replicas
    
    Returns:
        Tuple of (Vmax, Vmin, Vavg)
    """
    energies: List[float] = []
    line_count = 0
    
    with open(gamd_log_filename, "r") as file:
        for line in file:
            line_split = line.split()
            
            if not line_split or line_split[0] == "#":
                continue
            
            step = int(line_split[1])
            
            if step > initial_row:
                current_replica = line_count % n_replicas
                
                if current_replica == replica_idx:
                    energies.append(float(line_split[column]))
            
            line_count += 1
    
    if len(energies) == 0:
        logger.warning(f"No energy data found for replica {replica_idx}")
        return 0.0, 0.0, 0.0
    
    return float(max(energies)), float(min(energies)), float(np.mean(energies))


def collect_replica_data(
    gamd_log_filename: str,
    initial_row: int,
    replica_indices: List[int],
    n_replicas: int,
    system_runner
) -> List[List[float]]:
    """
    Collect [replica_idx, threshold, sigma, Vmax, Vmin, Vavg] for each replica.
    
    Args:
        gamd_log_filename: Path to log file
        initial_row: Starting step
        replica_indices: List of replica indices to process
        n_replicas: Total number of replicas
        system_runner: System runner object
    
    Returns:
        List of [replica_idx, threshold, sigma, Vmax, Vmin, Vavg] for each replica
    """
    replica_data = []
    
    for replica_idx in replica_indices:
        if (
            system_runner._integrator._GamdStageIntegrator__boost_type.name == "TOTAL"
            or system_runner._integrator._GamdStageIntegrator__boost_type.name
            == "DUAL_TOTAL_DIHEDRAL"
        ):
            column = 2  # Unboosted-Total-Energy

            # Calculate sigma for this replica
            tot_sd = compute_energy_width_per_replica(
                gamd_log_filename, initial_row, column, replica_idx, n_replicas
            )
            
            # Get energy statistics
            Vmax, Vmin, Vavg = get_energy_stats_per_replica(
                gamd_log_filename, initial_row, column, replica_idx, n_replicas
            )
            
            # Get current threshold
            current_threshold = system_runner._simulation.integrator.getGlobalVariableByName(
                "threshold_energy_Total"
            )
            
            replica_data.append([replica_idx, current_threshold, tot_sd, Vmax, Vmin, Vavg])
            logger.debug(f"  Replica {replica_idx}: σ={tot_sd:.2f}, Vmax={Vmax:.1f}, Vmin={Vmin:.1f}, Vavg={Vavg:.1f}")
    
    return replica_data


def calculate_k0_value(sigma0: float, sigma_V: float, Vmax: float, Vmin: float, Vavg: float) -> float:
    """
    Calculate k0 using GaMD upper-bound formula.
    
    Args:
        sigma0: Target sigma (from integrator parameter)
        sigma_V: Actual sigma of potential
        Vmax, Vmin, Vavg: Energy statistics
    
    Returns:
        Calculated k0 value
    """
    if sigma_V > 0.001 and abs(Vmax - Vmin) > 0.001 and abs(Vavg - Vmin) > 0.001:
        k0 = (1.0 - sigma0 / sigma_V) * (Vmax - Vmin) / (Vavg - Vmin)
        return k0
    else:
        return 0.0


def calculate_and_store_gamd_parameters(
    all_replica_data: List[List[float]],
    system_runner
) -> None:
    """
    Calculate threshold ladder and k0 values, then store them.
    
    Args:
        all_replica_data: List of [replica_idx, threshold, sigma, Vmax, Vmin, Vavg]
        system_runner: System runner object
    """
    # Extract [threshold, sigma] for ladder calculation
    thresholds_and_sigmas = [[d[1], d[2]] for d in all_replica_data]
    
    # Compute threshold ladder
    new_thresholds_list = new_thresholds(thresholds_and_sigmas)
    
    # Log threshold ladder
    logger.info("New threshold ladder:")
    for i in range(min(3, len(new_thresholds_list))):
        logger.info(f"  Replica {all_replica_data[i][0]}: threshold={new_thresholds_list[i]:.1f} kcal/mol")
    if len(new_thresholds_list) > 6:
        logger.info("  ...")
        for i in range(max(3, len(new_thresholds_list)-3), len(new_thresholds_list)):
            logger.info(f"  Replica {all_replica_data[i][0]}: threshold={new_thresholds_list[i]:.1f} kcal/mol")
    
    # Store thresholds
    if not hasattr(system_runner, '_gamd_replica_thresholds'):
        system_runner._gamd_replica_thresholds = {}
    
    for i, replica_idx in enumerate([d[0] for d in all_replica_data]):
        system_runner._gamd_replica_thresholds[replica_idx] = new_thresholds_list[i]
    
    # Calculate and store k0 values
    if not hasattr(system_runner, '_gamd_replica_k_values'):
        system_runner._gamd_replica_k_values = {}
    
    # Get sigma0 from integrator
    try:
        sigma0 = system_runner._simulation.integrator.getGlobalVariableByName("sigma0_Total")
    except:
        sigma0 = 6.0
        logger.warning("Could not get sigma0 from integrator, using default 6.0")
    
    logger.info("Calculated per-replica k0 values:")
    for i, replica_idx in enumerate([d[0] for d in all_replica_data]):
        sigma_V = all_replica_data[i][2]
        Vmax = all_replica_data[i][3]
        Vmin = all_replica_data[i][4]
        Vavg = all_replica_data[i][5]
        threshold = new_thresholds_list[i]
        
        # Calculate k0
        k0 = calculate_k0_value(sigma0, sigma_V, Vmax, Vmin, Vavg)
        
        # Store k0
        system_runner._gamd_replica_k_values[replica_idx] = k0
        
        logger.info(f"  Replica {replica_idx}: k0={k0:.6f}, σ={sigma_V:.2f}, threshold={threshold:.1f}")
    
    logger.info(f"Stored {len(system_runner._gamd_replica_thresholds)} per-replica thresholds")
    logger.info(f"Stored {len(system_runner._gamd_replica_k_values)} per-replica k0 values")


def distribute_and_apply_parameters(
    all_replica_data: List[List[float]],
    system_runner,
    communicator,
    replicas_per_worker: int,
    worker_rank: int,
    my_replica_indices: List[int]
) -> None:
    """
    Distribute parameters to workers and apply for 1:1 worker:replica ratio.
    
    Args:
        all_replica_data: All replica data
        system_runner: System runner
        communicator: Communicator object
        replicas_per_worker: Replicas per worker
        worker_rank: This worker's rank
        my_replica_indices: Replica indices for this worker
    """
    n_replicas = communicator.n_replicas
    new_thresholds_list = [system_runner._gamd_replica_thresholds[d[0]] for d in all_replica_data]
    
    thresholds_for_workers = [
        new_thresholds_list[i:i+replicas_per_worker] 
        for i in range(0, n_replicas, replicas_per_worker)
    ]
    my_new_thresholds = communicator.distribute_thresholds_to_workers(
        thresholds_for_workers
    )
    
    # For 1:1 worker:replica, apply immediately
    if replicas_per_worker == 1:
        worker_replica_idx = my_replica_indices[0]
        
        # Set threshold
        system_runner._simulation.integrator.setGlobalVariableByName(
            "threshold_energy_Total", my_new_thresholds[0]
        )
        
        # Set k0
        if worker_replica_idx in system_runner._gamd_replica_k_values:
            k0_value = system_runner._gamd_replica_k_values[worker_replica_idx]
            system_runner._simulation.integrator.setGlobalVariableByName(
                "k0_Total", k0_value
            )
            logger.info(f"Worker {worker_rank}: Applied threshold={my_new_thresholds[0]:.1f}, k0={k0_value:.6f}")
        else:
            logger.info(f"Worker {worker_rank}: Applied threshold={my_new_thresholds[0]:.1f}")


def apply_replica_gamd_parameters(system_runner, replica_idx: int) -> None:
    """
    Apply per-replica GaMD threshold and k0 before running a replica.
    Called from leader.py and worker.py.
    
    Args:
        system_runner: System runner object
        replica_idx: Index of replica to apply parameters for
    """
    if hasattr(system_runner, '_gamd_replica_thresholds'):
        if replica_idx in system_runner._gamd_replica_thresholds:
            threshold = system_runner._gamd_replica_thresholds[replica_idx]
            system_runner._simulation.integrator.setGlobalVariableByName(
                "threshold_energy_Total", threshold
            )
            
            # Also set k0 if available
            if hasattr(system_runner, '_gamd_replica_k_values'):
                if replica_idx in system_runner._gamd_replica_k_values:
                    k0 = system_runner._gamd_replica_k_values[replica_idx]
                    system_runner._simulation.integrator.setGlobalVariableByName(
                        "k0_Total", k0
                    )
                    logger.debug(f"Applied threshold {threshold:.1f} and k0 {k0:.6f} to replica {replica_idx}")
                    return
            
            logger.debug(f"Applied threshold {threshold:.1f} to replica {replica_idx}")