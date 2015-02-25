import logging
import meld
from meld import vault
from meld.system import get_runner
from simtk.openmm import version as mm_version
from meld.remd import multiplex_runner
import socket


logger = logging.getLogger(__name__)


def log_versions():
    logger.info('Meld version is %s', meld.__version__)
    logger.info('OpenMM_Meld version is %s', mm_version.full_version)


def launch(console_handler, debug=False):
    logger.info('loading data store')
    store = vault.DataStore.load_data_store()

    logger.info('initializing communicator')
    communicator = store.load_communicator()
    communicator.initialize()

    #
    # setup logging
    #
    hostname = socket.gethostname()
    hostid = '{}:{}'.format(hostname, communicator.rank)
    format = '{:10} %(asctime)s  %(name)s: %(message)s'.format(hostid)
    datefmt = '%Y-%m-%d %H:%M:%S'

    meld_logger = logging.getLogger('meld')
    formatter = logging.Formatter(fmt=format, datefmt=datefmt)
    # remove the console handler, so that
    # we can add a new handler below without
    # duplicate logging messages
    meld_logger.removeHandler(console_handler)
    level = logging.DEBUG if debug else logging.INFO
    print logging.getLevelName(level)
    if communicator.is_master():
        handler = logging.FileHandler(filename='remd.log', mode='a')
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(level)
    meld_logger.addHandler(handler)
    meld_logger.setLevel(level)

    if communicator.is_master():
        logger.info('Launching replica exchange on master')
    else:
        logger.info('Launching replica exchange on slave')
    log_versions()

    logger.info('Loading system')
    system = store.load_system()

    logger.info('Loading run options')
    options = store.load_run_options()

    system_runner = get_runner(system, options, communicator)

    if communicator.is_master():
        store.initialize(mode='a')
        remd_runner = store.load_remd_runner()
        remd_runner.run(communicator, system_runner, store)
    else:
        remd_runner = store.load_remd_runner().to_slave()
        remd_runner.run(communicator, system_runner)


def launch_multiplex(console_handler, debug=False):
    logger.info('Loading data store')
    store = vault.DataStore.load_data_store()

    #
    # Setup logging
    #
    level = logging.DEBUG if debug else logging.INFO
    fmt = '%(asctime)s  %(name)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'

    meld_logger = logging.getLogger('meld')
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    # remove the console handler, so that
    # we can add a new handler below without
    # duplicate logging messages
    meld_logger.removeHandler(console_handler)
    handler = logging.FileHandler(filename='remd.log', mode='a')
    handler.setFormatter(formatter)
    handler.setLevel(level)
    meld_logger.addHandler(handler)
    meld_logger.setLevel(level)
    logger.info('Launching replica exchange')
    log_versions()

    system = store.load_system()
    options = store.load_run_options()

    system_runner = get_runner(system, options, None)

    store.initialize(mode='a')
    remd_runner = store.load_remd_runner()
    runner = multiplex_runner.MultiplexReplicaExchangeRunner(remd_runner.n_replicas, remd_runner.max_steps,
                                                             remd_runner.ladder, remd_runner.adaptor,
                                                             remd_runner._step)

    runner.run(system_runner, store)
