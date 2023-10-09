#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
A module for launching replica exchange runs
"""

import meld
from meld import util
from meld import vault
from meld import runner
from openmm import version as mm_version  # type: ignore

import os
import logging
import socket
from typing import Union


Handler = Union[logging.StreamHandler, logging.FileHandler]

logger = logging.getLogger(__name__)


def log_versions() -> None:
    """Record version numbers to log"""
    logger.info("Meld version is %s", meld.__version__)
    logger.info("OpenMM_Meld version is %s", mm_version.full_version)


def launch(
    platform: str,
    console_handler: Handler,
    debug: bool = False,
    console_log: bool = False,
) -> None:
    """
    Launch a replica exchange run

    Args:
        platform: platform to run on [Reference, CPU, CUDA]
        console_handler: log handler for console logging
        debug: log debugging information
        console_log: display logging on console
    """
    logger.info("loading data store")
    store = vault.DataStore.load_data_store()

    logger.info("initializing communicator")
    communicator = store.load_communicator()
    communicator.initialize()

    #
    # setup logging
    #
    hostname = socket.gethostname()
    hostid = f"{hostname}:{communicator.rank:03d}"

    meld_logger = logging.getLogger("meld")
    # this filter adds the hostid to each logging record so
    # it gets printed out as part of the logging output
    hostid_log_filter = util.HostNameContextFilter(hostid)

    # remove the console handler, so that
    # we can add a new handler below without
    # duplicate logging messages
    meld_logger.removeHandler(console_handler)

    if not console_log:
        # setup file
        log_path = os.path.join(store.log_dir, f"remd_{communicator.rank:03d}.log")
        handler: Handler = logging.FileHandler(
            filename=log_path,
            mode="a",
        )
    else:
        fmt = "%(hostid)s %(asctime)s %(levelname)s %(name)s: %(message)s"
        fmt = fmt.format(hostid)
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

    meld_logger.addHandler(handler)
    handler.addFilter(hostid_log_filter)
    level = logging.DEBUG if debug else logging.INFO
    handler.setLevel(level)
    meld_logger.setLevel(level)
    meld_logger.propagate = False

    if communicator.is_leader():
        logger.info("Launching replica exchange on leader")
    else:
        logger.info("Launching replica exchange on worker")
    log_versions()

    logger.info("Loading system")
    system = store.load_system()

    logger.info("Loading run options")
    options = store.load_run_options()

    if options.enable_gamd == True:
        integrator_var = store.load_integrator()
        for key, value in integrator_var.items():
            setattr(system._integrator, key, value)

    system_runner = runner.get_runner(
        system, options, comm=communicator, platform=platform
    )

    if communicator.is_leader():
        store.initialize(mode="a")
        remd_runner = store.load_remd_runner()
        remd_runner.run(communicator, system_runner, store)
    else:
        remd_runner = store.load_remd_runner().to_worker()
        remd_runner.run(communicator, system_runner)
