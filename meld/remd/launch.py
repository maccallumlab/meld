import collections
from meld import vault



def launch():
    store = vault.DataStore.load_data_store()

    communicator = store.load_communicator()
    communicator.initialize()

    system = store.load_system()
    system_runner = system.get_runner()
    system_runner.initialize()

    if communicator.is_master():
        store.initialize(mode='existing')
        remd_runner = store.load_remd_runner()
        remd_runner.run(communicator, system_runner, store)
    else:
        remd_runner = store.load_remd_runner().to_slave()
        remd_runner.run(communicator, system_runner)
