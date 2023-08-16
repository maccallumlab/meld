"""
GaMELD Implementation Module
"""

from meld import interfaces
import numpy as np
from typing import List
import os


def change_thresholds(
    step: int,
    system_runner,
    communicator: interfaces.ICommunicator,
    leader: bool,
) -> None:
    """
    Change energy thresholds after cMD and GaMELD Equilibration

    Args:
        step: current step
        system_runner: a interfaces.IRunner object to run the simulations
        communicator: a communicator object to talk with workers
        leader: leader (True) or worker (False) indicator
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
        column: int = 0
        gamd_log_filename = os.path.join("Logs", f"gamd_{system_runner._rank:03d}.log")

        if (
            system_runner._integrator._GamdStageIntegrator__boost_type.name == "TOTAL"
            or system_runner._integrator._GamdStageIntegrator__boost_type.name
            == "DUAL_TOTAL_DIHEDRAL"
        ):
            column = 2  # Unboosted-Total-Energy

            # calculate standard deviation of the energy (width)
            tot_sd = compute_energy_width(gamd_log_filename, initial_row, column)

            tot_threshold_sd: List[List[float]] = [
                system_runner._simulation.integrator.getGlobalVariableByName(
                    "threshold_energy_Total"
                ),
                tot_sd,
            ]
            if leader == True:
                # gather energy thresholds and widths
                tot_thresholds: List[
                    List[float]
                ] = communicator.gather_thresholds_from_workers(tot_threshold_sd)
                # set new thresholds
                tot_new_threshold: List[float] = new_thresholds(tot_thresholds)
                tot_threshold = communicator.distribute_thresholds_to_workers(
                    tot_new_threshold
                )
            else:
                communicator.send_thresholds_to_leader(tot_threshold_sd)
                tot_threshold = communicator.receive_thresholds_from_leader()
            system_runner._simulation.integrator.setGlobalVariableByName(
                "threshold_energy_Total", tot_threshold
            )

        if (
            system_runner._integrator._GamdStageIntegrator__boost_type.name
            == "DIHEDRAL"
            or system_runner._integrator._GamdStageIntegrator__boost_type.name
            == "DUAL_TOTAL_DIHEDRAL"
        ):
            column = 3  # Unboosted-Dihedral-Energy

            # calculate standard deviation of the energy (width)
            dih_sd = compute_energy_width(gamd_log_filename, initial_row, column)

            dih_threshold_sd: List[List[float]] = [
                system_runner._simulation.integrator.getGlobalVariableByName(
                    "threshold_energy_Dihedral"
                ),
                dih_sd,
            ]
            if leader == True:
                # gather energy thresholds and widths
                dih_thresholds: List[
                    List[float]
                ] = communicator.gather_thresholds_from_workers(dih_threshold_sd)
                # set new threshold
                dih_new_threshold = new_thresholds(dih_thresholds)
                dih_threshold = communicator.distribute_thresholds_to_workers(
                    dih_new_threshold
                )
            else:
                communicator.send_thresholds_to_leader(dih_threshold_sd)
                dih_threshold = communicator.receive_thresholds_from_leader()
            system_runner._simulation.integrator.setGlobalVariableByName(
                "threshold_energy_Dihedral", dih_threshold
            )


def compute_energy_width(
    gamd_log_filename: str, initial_row: int, column: int
) -> float:
    # Compute standard deviation
    energy: List[float] = []
    with open(gamd_log_filename, "r") as file:
        for line in file:
            line_split = line.split()
            if not line_split[0] == "#":
                if int(line_split[1]) > initial_row:
                    energy.append(float(line_split[column]) * 4.184)  # Unboosted-Energy
    return np.std(energy)


def new_thresholds(thresholds: List[List[float]]) -> List[float]:
    # Compute new thresholds
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
