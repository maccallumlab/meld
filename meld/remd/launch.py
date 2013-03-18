import cPickle as pickle
import collections


RemdSavedState = collections.namedtuple('RemdSavedState', 'communicator system_runner remd_runner store')


def launch():
    result = pickle.load('restart.dat')

    communicator = result.communicator
    communicator.initialize()

    system_runner = result.system_runner
    system_runner.initialize()

    if communicator.is_master():
        store = result.store
        store.initialize()
        remd_runner = result.remd_runner
        remd_runner.run(communicator, system_runner, store)
    else:
        remd_runner = result.remd_runner.to_slave()
        remd_runner.run(communicator, system_runner)
