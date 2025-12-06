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
    FIXED: Handles interleaved log files and per-replica parameters

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
    elif step == (
        (system_runner._integrator.ntcmd + system_runner._integrator.nteb)
        / system_runner._options.timesteps
    ):
        initial_row = int(
            (system_runner._integrator.ntcmd + system_runner._integrator.ntebprep)
            / system_runner._options.timesteps
        )

    if initial_row != 0:
        # Determine which replicas this worker handles
        n_replicas = communicator.n_replicas
        n_workers = communicator.n_workers
        replicas_per_worker = n_replicas // n_workers
        worker_rank = communicator.rank
        
        my_replica_indices = list(range(
            worker_rank * replicas_per_worker,
            (worker_rank + 1) * replicas_per_worker
        ))
        
        gamd_log_filename = os.path.join("Logs", f"gamd_{worker_rank:03d}.log")

        if (
            system_runner._integrator._GamdStageIntegrator__boost_type.name == "TOTAL"
            or system_runner._integrator._GamdStageIntegrator__boost_type.name
            == "DUAL_TOTAL_DIHEDRAL"
        ):
            column = 2  # Unboosted-Total-Energy

            # Collect data for each replica this worker handles
            my_replica_data = []
            for replica_idx in my_replica_indices:
                # Calculate standard deviation for THIS replica
                tot_sd = compute_energy_width_per_replica(
                    gamd_log_filename, initial_row, column, replica_idx, n_replicas
                )
                
                current_threshold = system_runner._simulation.integrator.getGlobalVariableByName(
                    "threshold_energy_Total"
                )
                
                # Store as [replica_idx, threshold, sigma]
                my_replica_data.append([replica_idx, current_threshold, tot_sd])

            if leader == True:
                # Gather energy thresholds and widths from all workers
                if n_workers > 1:
                    all_workers_data = communicator.gather_thresholds_from_workers(my_replica_data)
                    all_replica_data = []
                    for worker_data in all_workers_data:
                        all_replica_data.extend(worker_data)
                else:
                    all_replica_data = my_replica_data
                
                all_replica_data.sort(key=lambda x: x[0])
                
                # Extract [threshold, sigma] for new_thresholds calculation
                tot_thresholds = [[d[1], d[2]] for d in all_replica_data]
                
                # Set new thresholds
                tot_new_threshold: List[float] = new_thresholds(tot_thresholds)
                
                # Store thresholds and k0 values per-replica
                store_gamd_parameters_total(
                    all_replica_data, tot_new_threshold, system_runner,
                    gamd_log_filename, n_replicas
                )
                
                # Distribute to workers
                if n_workers > 1:
                    thresholds_for_workers = [
                        tot_new_threshold[i:i+replicas_per_worker]
                        for i in range(0, n_replicas, replicas_per_worker)
                    ]
                    tot_threshold = communicator.distribute_thresholds_to_workers(
                        thresholds_for_workers
                    )
                    
                    # Apply for 1:1 worker:replica ratio
                    if replicas_per_worker == 1:
                        apply_parameters_to_integrator(
                            system_runner, my_replica_indices[0], 
                            tot_threshold[0], gamd_log_filename, n_replicas, "Total"
                        )
                else:
                    tot_threshold = tot_new_threshold[0]
            else:
                communicator.send_thresholds_to_leader(my_replica_data)
                tot_threshold = communicator.receive_thresholds_from_leader()
                
                # Store received thresholds
                if not hasattr(system_runner, '_gamd_replica_thresholds'):
                    system_runner._gamd_replica_thresholds = {}
                for i, replica_idx in enumerate(my_replica_indices):
                    if i < len(tot_threshold):
                        system_runner._gamd_replica_thresholds[replica_idx] = tot_threshold[i]
                
                # Apply for 1:1 worker:replica ratio
                if replicas_per_worker == 1:
                    system_runner._simulation.integrator.setGlobalVariableByName(
                        "threshold_energy_Total", tot_threshold[0]
                    )

        if (
            system_runner._integrator._GamdStageIntegrator__boost_type.name
            == "DIHEDRAL"
            or system_runner._integrator._GamdStageIntegrator__boost_type.name
            == "DUAL_TOTAL_DIHEDRAL"
        ):
            column = 3  # Unboosted-Dihedral-Energy

            # Collect data for each replica this worker handles
            my_replica_data = []
            for replica_idx in my_replica_indices:
                # Calculate standard deviation for THIS replica
                dih_sd = compute_energy_width_per_replica(
                    gamd_log_filename, initial_row, column, replica_idx, n_replicas
                )
                
                current_threshold = system_runner._simulation.integrator.getGlobalVariableByName(
                    "threshold_energy_Dihedral"
                )
                
                # Store as [replica_idx, threshold, sigma]
                my_replica_data.append([replica_idx, current_threshold, dih_sd])

            if leader == True:
                # Gather energy thresholds and widths from all workers
                if n_workers > 1:
                    all_workers_data = communicator.gather_thresholds_from_workers(my_replica_data)
                    all_replica_data = []
                    for worker_data in all_workers_data:
                        all_replica_data.extend(worker_data)
                else:
                    all_replica_data = my_replica_data
                
                all_replica_data.sort(key=lambda x: x[0])
                
                # Extract [threshold, sigma] for new_thresholds calculation
                dih_thresholds = [[d[1], d[2]] for d in all_replica_data]
                
                # Set new threshold
                dih_new_threshold = new_thresholds(dih_thresholds)
                
                # Store thresholds and k0 values per-replica
                store_gamd_parameters_dihedral(
                    all_replica_data, dih_new_threshold, system_runner,
                    gamd_log_filename, n_replicas
                )
                
                # Distribute to workers
                if n_workers > 1:
                    thresholds_for_workers = [
                        dih_new_threshold[i:i+replicas_per_worker]
                        for i in range(0, n_replicas, replicas_per_worker)
                    ]
                    dih_threshold = communicator.distribute_thresholds_to_workers(
                        thresholds_for_workers
                    )
                    
                    # Apply for 1:1 worker:replica ratio
                    if replicas_per_worker == 1:
                        apply_parameters_to_integrator(
                            system_runner, my_replica_indices[0],
                            dih_threshold[0], gamd_log_filename, n_replicas, "Dihedral"
                        )
                else:
                    dih_threshold = dih_new_threshold[0]
            else:
                communicator.send_thresholds_to_leader(my_replica_data)
                dih_threshold = communicator.receive_thresholds_from_leader()
                
                # Store received thresholds
                if not hasattr(system_runner, '_gamd_replica_thresholds_dihedral'):
                    system_runner._gamd_replica_thresholds_dihedral = {}
                for i, replica_idx in enumerate(my_replica_indices):
                    if i < len(dih_threshold):
                        system_runner._gamd_replica_thresholds_dihedral[replica_idx] = dih_threshold[i]
                
                # Apply for 1:1 worker:replica ratio
                if replicas_per_worker == 1:
                    system_runner._simulation.integrator.setGlobalVariableByName(
                        "threshold_energy_Dihedral", dih_threshold[0]
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
    
    Replaces compute_energy_width() to fix interleaved log bug.
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
        return 0.0
    
    return float(np.std(energy))


def get_k0_from_log(
    gamd_log_filename: str,
    k0_column: int,
    replica_idx: int,
    n_replicas: int
) -> float:
    """
    Read k0 value from the last logged step for a specific replica.
    The integrator already calculated this k0 during MD.
    
    Args:
        gamd_log_filename: Path to GaMD log file
        k0_column: Column containing k0 (9=Total, 10=Dihedral)
        replica_idx: Which replica to extract
        n_replicas: Total number of replicas
    
    Returns:
        k0 value that integrator calculated for this replica
    """
    k0_value = 0.0
    line_count = 0
    
    with open(gamd_log_filename, "r") as file:
        for line in file:
            line_split = line.split()
            if not line_split or line_split[0] == "#":
                continue
            
            current_replica = line_count % n_replicas
            
            # Keep updating k0 for our replica (last one will be most recent)
            if current_replica == replica_idx:
                try:
                    k0_value = float(line_split[k0_column])
                except (IndexError, ValueError):
                    pass  # Keep previous value if column doesn't exist
            
            line_count += 1
    
    return k0_value


def store_gamd_parameters_total(
    all_replica_data: List[List[float]],
    new_thresholds_list: List[float],
    system_runner,
    gamd_log_filename: str,
    n_replicas: int
) -> None:
    """
    Store per-replica thresholds and k0 values for Total boost.
    Reads k0 directly from log (integrator already calculated it).
    """
    # Initialize storage
    if not hasattr(system_runner, '_gamd_replica_thresholds'):
        system_runner._gamd_replica_thresholds = {}
    if not hasattr(system_runner, '_gamd_replica_k_values'):
        system_runner._gamd_replica_k_values = {}
    
    # Store thresholds and read k0 from log
    for i, data in enumerate(all_replica_data):
        replica_idx = int(data[0])
        threshold = new_thresholds_list[i]
        
        # Store threshold
        system_runner._gamd_replica_thresholds[replica_idx] = threshold
        
        # Read k0 from log (column 9 = Total-Effective-Harmonic-Constant)
        k0 = get_k0_from_log(gamd_log_filename, 9, replica_idx, n_replicas)
        system_runner._gamd_replica_k_values[replica_idx] = k0


def store_gamd_parameters_dihedral(
    all_replica_data: List[List[float]],
    new_thresholds_list: List[float],
    system_runner,
    gamd_log_filename: str,
    n_replicas: int
) -> None:
    """
    Store per-replica thresholds and k0 values for Dihedral boost.
    Reads k0 directly from log (integrator already calculated it).
    """
    # Initialize storage
    if not hasattr(system_runner, '_gamd_replica_thresholds_dihedral'):
        system_runner._gamd_replica_thresholds_dihedral = {}
    if not hasattr(system_runner, '_gamd_replica_k_values_dihedral'):
        system_runner._gamd_replica_k_values_dihedral = {}
    
    # Store thresholds and read k0 from log
    for i, data in enumerate(all_replica_data):
        replica_idx = int(data[0])
        threshold = new_thresholds_list[i]
        
        # Store threshold
        system_runner._gamd_replica_thresholds_dihedral[replica_idx] = threshold
        
        # Read k0 from log (column 10 = Dihedral-Effective-Harmonic-Constant)
        k0 = get_k0_from_log(gamd_log_filename, 10, replica_idx, n_replicas)
        system_runner._gamd_replica_k_values_dihedral[replica_idx] = k0


def apply_parameters_to_integrator(
    system_runner,
    replica_idx: int,
    threshold: float,
    gamd_log_filename: str,
    n_replicas: int,
    boost_type: str
) -> None:
    """Apply threshold and k0 to integrator for given boost type."""
    system_runner._simulation.integrator.setGlobalVariableByName(
        f"threshold_energy_{boost_type}", threshold
    )
    
    # Read and apply k0 from log
    k0_column = 9 if boost_type == "Total" else 10
    k0 = get_k0_from_log(gamd_log_filename, k0_column, replica_idx, n_replicas)
    system_runner._simulation.integrator.setGlobalVariableByName(
        f"k0_{boost_type}", k0
    )


def apply_replica_gamd_parameters(system_runner, replica_idx: int) -> None:
    """
    Apply per-replica GaMD threshold and k0 before running a replica.
    Handles both Total and Dihedral boost types.
    """
    # Apply Total boost parameters
    if hasattr(system_runner, '_gamd_replica_thresholds'):
        if replica_idx in system_runner._gamd_replica_thresholds:
            threshold = system_runner._gamd_replica_thresholds[replica_idx]
            system_runner._simulation.integrator.setGlobalVariableByName(
                "threshold_energy_Total", threshold
            )
            
            if hasattr(system_runner, '_gamd_replica_k_values'):
                if replica_idx in system_runner._gamd_replica_k_values:
                    k0 = system_runner._gamd_replica_k_values[replica_idx]
                    system_runner._simulation.integrator.setGlobalVariableByName(
                        "k0_Total", k0
                    )
    
    # Apply Dihedral boost parameters
    if hasattr(system_runner, '_gamd_replica_thresholds_dihedral'):
        if replica_idx in system_runner._gamd_replica_thresholds_dihedral:
            threshold = system_runner._gamd_replica_thresholds_dihedral[replica_idx]
            system_runner._simulation.integrator.setGlobalVariableByName(
                "threshold_energy_Dihedral", threshold
            )
            
            if hasattr(system_runner, '_gamd_replica_k_values_dihedral'):
                if replica_idx in system_runner._gamd_replica_k_values_dihedral:
                    k0 = system_runner._gamd_replica_k_values_dihedral[replica_idx]
                    system_runner._simulation.integrator.setGlobalVariableByName(
                        "k0_Dihedral", k0
                    )