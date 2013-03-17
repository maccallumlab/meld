class MPICommunicator(object):
    def initialize(self):
        pass

    def is_master(self):
        pass

    def recieve_lambda_from_master(self):
        pass

    def recieve_state_from_master(self):
        pass

    def send_state_to_master(self, state):
        pass

    def recieve_states_for_energy_calc_from_master(self):
        pass

    def send_energies_to_master(self, energies):
        pass
