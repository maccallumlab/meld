import logging
from meld import vault
from meld.system import get_runner
from meld import version as meld_version
from simtk.openmm import version as mm_version
from meld.remd import multiplex_runner


logger = logging.getLogger(__name__)


def log_versions():
    logger.info('Meld version is %s', meld_version.full_version)
    logger.info('OpenMM_Meld version is %s', mm_version.full_version)


def launch(debug=False):
    store = vault.DataStore.load_data_store()

    communicator = store.load_communicator()
    communicator.initialize()

    if communicator.is_master():
        level = logging.DEBUG if debug else logging.INFO
        format = '%(asctime)s  %(name)s: %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        logging.basicConfig(filename='remd.log', level=level, format=format,
                            datefmt=datefmt)
        logger.info('Launching replica exchange')
        log_versions()
    else:
        if debug:
            format = '%(asctime)s  %(name)s: %(message)s'
            datefmt = '%Y-%m-%d %H:%M:%S'
            logging.basicConfig(level=logging.DEBUG, format=format,
                                datefmt=datefmt)
            logger.info('Launching replica exchange')
            log_versions()

    system = store.load_system()
    options = store.load_run_options()

    system_runner = get_runner(system, options, communicator)

    if communicator.is_master():
        store.initialize(mode='a')
        remd_runner = store.load_remd_runner()
        remd_runner.run(communicator, system_runner, store)
    else:
        remd_runner = store.load_remd_runner().to_slave()
        remd_runner.run(communicator, system_runner)


def launch_multiplex(debug=False):
    store = vault.DataStore.load_data_store()

    level = logging.DEBUG if debug else logging.INFO
    format = '%(asctime)s  %(name)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(filename='remd.log', level=level, format=format,
                        datefmt=datefmt)
    logger.info('Launching replica exchange')
    log_versions()

    system = store.load_system()
    options = store.load_run_options()

    system_runner = get_runner(system, options, None)

    store.initialize(mode='a')
    remd_runner = store.load_remd_runner()
    runner = multiplex_runner.MultiplexReplicaExchangeRunner(remd_runner.n_replicas, remd_runner.max_steps,
                                                             remd_runner.ladder, remd_runner.adaptor,
                                                             remd_runner.ramp_steps)

    runner.run(system_runner, store)
