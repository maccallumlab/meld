class MasterReplicaExchangeRunner(object):
    '''
    Class to coordinate running of replica exchange

    This class doesn't really know much about the calculation that is happening,
    but it's the glue that holds everything together.
    '''

    #
    # read only properties
    #

    @property
    def n_replicas(self):
        return self._n_replicas

    @property
    def lambdas(self):
        return self._lambdas

    @property
    def step(self):
        return self._step

    @property
    def max_steps(self):
        return self._max_steps

    #
    # public methods
    #

    def __init__(self, n_replicas, max_steps, ladder, adaptor):
        '''
        Initialize a MasterReplicaExchangeRunner

        Parameters
            n_replicas -- number of replicas
            max_steps -- maximum number of steps to run
            ladder -- Ladder object to handle exchanges
            adaptor -- Adaptor object to handle lambda adaptation

        '''
        self._n_replicas = n_replicas
        self._max_steps = max_steps
        self._step = 1
        self.ladder = ladder
        self.adaptor = adaptor

        self._lambdas = None
        self._setup_lambdas()

    def to_slave(self):
        '''
        Return a SlaveReplicaExchangeRunner based on self.

        '''
        return slave_runner.SlaveReplicaExchangeRunner.from_master(self)

    def run(self, communicator, replica_runner, store):
        '''
        Run replica exchange until finished

        Parameters
            communicator -- A communicator object to talk with slaves
            replica_runner -- a ReplicaRunner object to run the simulations
            store -- a Store object to handle storing data to disk

        '''
        # check to make sure n_replicas matches
        assert self._n_replicas == communicator.n_replicas
        assert self._n_replicas == store.n_replicas

        # load previous state from the store
        states = store.load_states(step=self.step - 1)

        # the master is always at lambda = 0, so set that here
        replica_runner.set_lambda(0.)

        while self._step <= self._max_steps:
            # update lambdas
            self._lambdas = self.adaptor.adapt(self._lambdas, self._step)
            communicator.broadcast_lambdas_to_slaves(self._lambdas)

            # do one step
            my_state = communicator.broadcast_states_to_slaves(states)
            if self._step == 1:
                my_state = replica_runner.minimize_then_run(my_state)
            else:
                my_state = replica_runner.run(my_state)

            # gather all of the states
            states = communicator.gather_states_from_slaves(my_state)

            # send them to the slaves
            communicator.broadcast_states_for_energy_calc_to_slaves(states)

            # compute our energy for each state
            my_energies = self._compute_energies(states, replica_runner)
            energies = communicator.gather_energies_from_slaves(my_energies)

            # ask the ladder how to permute things
            permutation_vector = self.ladder.compute_exchanges(energies, self.adaptor)
            states = self._permute_states(permutation_vector, states)

            # store everything
            store.save_states(states, self.step)
            store.append_traj(states[0])
            store.save_lambdas(self._lambdas, self.step)
            store.save_permutation_vector(permutation_vector, self.step)

            # on to the next step!
            self._step += 1
            store.save_remd_runner(self)
            store.backup(self.step - 1)

    #
    # private helper methods
    #

    @staticmethod
    def _compute_energies(states, replica_runner):
        my_energies = []
        for state in states:
            my_energies.append(replica_runner.get_energy(state))
        return my_energies

    @staticmethod
    def _permute_states(permutation_matrix, states):
        new_states = []
        for index in permutation_matrix:
            new_states.append(states[index])
        return new_states

    def _setup_lambdas(self):
        delta = 1.0 / (self._n_replicas - 1.0)
        self._lambdas = [i * delta for i in range(self._n_replicas)]
