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
        Initialize a new slave from a master.

        Parameters
            master -- a MasterReplicaExchangeRunner to serve as a template
        Returns
            a SlaveReplicaExchangeRunner

        '''
        new_slave = cls(master.step, master.max_steps)
        return new_slave

    @property
    def step(self):
        return self._step

    @property
    def max_steps(self):
        return self._max_steps

    def run(self, communicator, system_runner):
        '''
        Continue running slave jobs until done.

        Parameters
            communicator -- a communicator object for talking to the master
            system_runner -- a system_runner object for actually running the simulations

        '''
        my_lambda = None

        while self._step <= self._max_steps:
            # update simulation conditions
            new_lambda = communicator.recieve_lambda_from_master()
            if not new_lambda == my_lambda:
                my_lambda = new_lambda
                system_runner.set_lambda(my_lambda)

            # do one round of simulation
            state = communicator.recieve_state_from_master()
            if self._step == 1:
                state = system_runner.minimize_then_run(state)
            else:
                state = system_runner.run(state)
            communicator.send_state_to_master(state)

            # compute energies
            states = communicator.recieve_states_for_energy_calc_from_master()
            energies = []
            for state in states:
                energy = system_runner.get_energy(state)
                energies.append(energy)
            communicator.send_energies_to_master(energies)
            self._step += 1
