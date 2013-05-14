from meld import vault
from meld.system import get_runner


def launch():
    store = vault.DataStore.load_data_store()

    communicator = store.load_communicator()
    communicator.initialize()

    system = store.load_system()
    options = store.load_run_options()

    system_runner = get_runner(system, options)

    if communicator.is_master():
        store.initialize(mode='existing')
        remd_runner = store.load_remd_runner()
        remd_runner.run(communicator, system_runner, store)
    else:
        remd_runner = store.load_remd_runner().to_slave()
        remd_runner.run(communicator, system_runner)
