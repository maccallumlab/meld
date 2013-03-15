class SlaveReplicaExchangeRunner(object):
    '''
    This class coordinates running replica exchange on the slaves.

    '''

    def __init__(self, step, max_steps):
        self._step = step
        self._max_steps = max_steps

    @classmethod
    def from_master(cls, master):
        '''
        Initialize a new SlaveReplicaExchangeRunner from a MasterReplicaExchangeRunner.
        '''
        new_slave = cls(master.step, master.max_steps)
        return new_slave

    @property
    def step(self):
        return self._step

    @property
    def max_steps(self):
        return self._max_steps

    def run(self, communicator, replica_runner):
        '''
        Continue running slave jobs until done.

        Parameters
            communicator -- a communicator object for talking to the master
            replica_runner -- a replica_runner object for actually running the simulations

        '''
        my_lambda = None

        while self._step <= self._max_steps:
            # update simulation conditions
            new_lambda = communicator.recieve_lambda()
            if not new_lambda == my_lambda:
                my_lambda = new_lambda
                replica_runner.set_lambda(my_lambda)

            # do one round of simulation
            state = communicator.recieve_state()
            if self._step == 1:
                state = replica_runner.minimize_then_run(state)
            else:
                state = replica_runner.run(state)
            communicator.send_state(state)

            # compute energies
            states = communicator.recieve_states_for_energy_calc()
            energies = []
            for state in states:
                energy = replica_runner.get_energy(state)
                energies.append(energy)
            communicator.send_energies(energies)
            self._step += 1
