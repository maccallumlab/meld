class SlaveReplicaExchangeRunner(object):
    '''
    This class coordinates running replica exchange on the slaves.

    '''

    def __init__(self, step, max_steps, ramp_steps=None):
        self._step = step
        self._max_steps = max_steps
        self._ramp_steps = ramp_steps

    @classmethod
    def from_master(cls, master):
        '''
        Initialize a new slave from a master.

        Parameters
            master -- a MasterReplicaExchangeRunner to serve as a template
        Returns
            a SlaveReplicaExchangeRunner

        '''
        new_slave = cls(master.step, master.max_steps, master.ramp_steps)
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
        my_alpha = None

        while self._step <= self._max_steps:
            # update simulation conditions
            new_alpha = communicator.receive_alpha_from_master()

            state = communicator.receive_state_from_master()

            ramp_weight = self._compute_ramp_weight()
            my_alpha = new_alpha
            system_runner.set_alpha(my_alpha, ramp_weight)
            state.alpha = my_alpha

            # do one round of simulation
            if self._step == 1:
                state = system_runner.minimize_then_run(state)
            else:
                state = system_runner.run(state)

            # compute energies
            states = communicator.exchange_states_for_energy_calc(state)

            energies = []
            for state in states:
                energy = system_runner.get_energy(state)
                energies.append(energy)
            communicator.send_energies_to_master(energies)

            self._step += 1

    def _compute_ramp_weight(self):
        if self._ramp_steps is None:
            return 1.0
        else:
            if self._step > self._ramp_steps:
                return 1.0
            else:
                return (float(self.step + 1) / float(self._ramp_steps)) ** 4
