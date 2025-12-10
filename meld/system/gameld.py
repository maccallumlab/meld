"""
GaMELD Implementation Module. 

GaMELD combines Gaussian Accelerated Molecular Dynamics (GaMD) and MELD using a new threshold calculation.

References:
    M. Caparotta, A. Perez, When MELD meets GaMD: Accelerating Biomolecular Landscape Exploration.
"""

from meld import interfaces
import numpy as np
from typing import Any, List, Optional
import os


def change_thresholds(
    step: int,
    system_runner: Any,
    communicator: interfaces.ICommunicator,
    leader: bool,
    base_replica_index: int = 0,
) -> None:
    """
    Change energy thresholds after cMD and GaMELD Equilibration

    Args:
        step: current step
        system_runner: a interfaces.IRunner object to run the simulations
        communicator: a communicator object to talk with workers
        leader: leader (True) or worker (False) indicator
        base_replica_index: base index for replicas on this worker (for sequential mode)
    """

    # Check if it's time to change thresholds
    initial_row: int = 0
    if step == (system_runner._integrator.ntcmd / system_runner._options.timesteps):
        initial_row = (
            system_runner._integrator.ntcmdprep / system_runner._options.timesteps
        )
    elif step == (
        (system_runner._integrator.ntcmd + system_runner._integrator.nteb)
        / system_runner._options.timesteps
    ):
        initial_row = (
            system_runner._integrator.ntcmd + system_runner._integrator.ntebprep
        ) / system_runner._options.timesteps

    if initial_row != 0:
        replicas_per_worker = communicator.n_replicas // communicator.n_workers
        
        # Determine which boost types to process
        boost_type_name = system_runner._integrator._GamdStageIntegrator__boost_type.name
        process_total = boost_type_name in ["TOTAL", "DUAL_TOTAL_DIHEDRAL"]
        process_dihedral = boost_type_name in ["DIHEDRAL", "DUAL_TOTAL_DIHEDRAL"]
        
        # STEP 1: Collect (threshold, σ) for all local replicas
        tot_threshold_sd_list: List[List[float]] = []
        dih_threshold_sd_list: List[List[float]] = []
        
        for local_idx in range(replicas_per_worker):
            replica_index = base_replica_index + local_idx
            
            # Restore this replica's parameters before calculating
            if hasattr(system_runner, '_restore_gamd_params'):
                system_runner._restore_gamd_params(replica_index)
            
            # Use unified naming: gamd_{ID}.log where ID is replica_index
            gamd_log_filename = os.path.join("Logs", f"gamd_{replica_index:03d}.log")

            # Collect TOTAL boost statistics
            if process_total:
                column = 2  # Unboosted-Total-Energy
                tot_sd = compute_energy_width(gamd_log_filename, initial_row, column)
                tot_threshold = system_runner._simulation.integrator.getGlobalVariableByName(
                    "threshold_energy_Total"
                )
                tot_threshold_sd_list.append([tot_threshold, tot_sd])

            # Collect DIHEDRAL boost statistics
            if process_dihedral:
                column = 3  # Unboosted-Dihedral-Energy
                dih_sd = compute_energy_width(gamd_log_filename, initial_row, column)
                dih_threshold = system_runner._simulation.integrator.getGlobalVariableByName(
                    "threshold_energy_Dihedral"
                )
                dih_threshold_sd_list.append([dih_threshold, dih_sd])
        
        # STEP 2: Gather from all workers and calculate new thresholds
        my_tot_thresholds: Any = None
        my_dih_thresholds: Any = None

        if process_total:
            if leader:
                gathered = communicator.gather_thresholds_from_workers(tot_threshold_sd_list)
                all_tot_thresholds = [item for sublist in gathered for item in sublist]
                tot_new_thresholds = new_thresholds(all_tot_thresholds)
                chunks = [tot_new_thresholds[i:i + replicas_per_worker] 
                         for i in range(0, len(tot_new_thresholds), replicas_per_worker)]
                my_tot_thresholds = communicator.distribute_thresholds_to_workers(chunks)  # type: ignore
            else:
                communicator.send_thresholds_to_leader(tot_threshold_sd_list)
                my_tot_thresholds = communicator.receive_thresholds_from_leader()

        if process_dihedral:
            if leader:
                gathered = communicator.gather_thresholds_from_workers(dih_threshold_sd_list)
                all_dih_thresholds = [item for sublist in gathered for item in sublist]
                dih_new_thresholds = new_thresholds(all_dih_thresholds)
                chunks = [dih_new_thresholds[i:i + replicas_per_worker] 
                         for i in range(0, len(dih_new_thresholds), replicas_per_worker)]
                my_dih_thresholds = communicator.distribute_thresholds_to_workers(chunks)  # type: ignore
            else:
                communicator.send_thresholds_to_leader(dih_threshold_sd_list)
                my_dih_thresholds = communicator.receive_thresholds_from_leader()
        
        # STEP 3: Apply the appropriate thresholds to each local replica
        for local_idx in range(replicas_per_worker):
            replica_index = base_replica_index + local_idx
            
            if hasattr(system_runner, '_restore_gamd_params'):
                system_runner._restore_gamd_params(replica_index)
            
            if process_total and my_tot_thresholds is not None:
                system_runner._simulation.integrator.setGlobalVariableByName(
                    "threshold_energy_Total", my_tot_thresholds[local_idx]
                )
            
            if process_dihedral and my_dih_thresholds is not None:
                system_runner._simulation.integrator.setGlobalVariableByName(
                    "threshold_energy_Dihedral", my_dih_thresholds[local_idx]
                )
            
            if hasattr(system_runner, '_save_gamd_params'):
                system_runner._save_gamd_params(replica_index)

                
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
            if line_split and line_split[0] != "#":
                if int(line_split[1]) > initial_row:
                    energy.append(float(line_split[column]) * 4.184)
    return float(np.std(energy))


def new_thresholds(thresholds: List[List[float]]) -> List[float]:
    """ 
    Compute new thresholds from (threshold, σ) pairs.
    """
    new_threshold: List[float] = []
    sd_acum: float = 0.0
    thresholds_copy = list(reversed(thresholds))
    
    for i, threshold_sd in enumerate(thresholds_copy):
        if i == 0:
            new_threshold.insert(0, threshold_sd[0])
        else:
            sd_acum = sd_acum + thresholds_copy[i][1]
            new_threshold.insert(i, new_threshold[0] - sd_acum)
    
    return list(reversed(new_threshold))