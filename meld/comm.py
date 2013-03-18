class MPICommunicator(object):
    def initialize(self):
        pass

    def is_master(self):
        '''
        Is this the master node?

        Returns
            True if we are the master, otherwise False

        '''
        pass

    def broadcast_lambdas_to_slaves(self, lambdas):
        '''
        Send the lambda values to the slaves

        Parameters
            lambdas -- a list of lambda values, one for each replica
        Returns
            None

        The master node's lambda value should be included in this list.
        The master node will always be at lambda=0.0

        '''
        pass

    def recieve_lambda_from_master(self):
        '''
        Recieve lambda value from master node

        Returns
            a floating point value for lambda in [0,1]

        '''
        pass

    def broadcast_states_to_slaves(self, states):
        '''
        Send a state to each slave

        Parameters
            states -- a list of states
        Returns
            the state to run on the master node

        The list of states should include the state for the master node. These are the
        states that will be simulated on each replica for each step.

        '''
        pass

    def recieve_state_from_master(self):
        '''
        Get state to run for this step

        Returns
            the state to run for this step

        '''
        pass

    def gather_states_from_slaves(self, state_on_master):
        '''
        Recieve states from all slaves

        Parameters
            state_on_master -- the state on the master after simulating
        Returns
            a list of states, one from each replica

        The returned states are the states after simulating.

        '''
        pass

    def send_state_to_master(self, state):
        '''
        Send state to master

        Parameters
            state -- state to send to master
        Returns
            None

        This is the state after simulating this step.

        '''
        pass

    def broadcast_states_for_energy_calc_to_slaves(self, states):
        '''
        Broadcast states to all slaves

        Parameters
            states -- a list of states
        Returns
            None

        Send all results from this step to every slave so that we can calculate
        the energies and do replica exchange.

        '''
        pass

    def recieve_states_for_energy_calc_from_master(self):
        '''
        Recieve all states from master

        Returns
            a list of states to calculate the energy of

        '''
        pass

    def gather_energies_from_slaves(self, energies_on_master):
        '''
        Recieve a list of energies from each slave

        Parameters
            energies_on_master -- a list of energies from the master
        Returns
            a square matrix of every state on every replica to be used for replica exchange

        '''
        pass

    def send_energies_to_master(self, energies):
        '''
        Send a list of energies to the master

        Parameters
            energies -- a list of energies to send to the master
        Returns
            None

        '''
        pass

    @property
    def n_replicas(self):
        pass
