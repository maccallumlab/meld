class SystemState(object):
    '''
    Class to hold the state of a system
    '''
    def __init__(self, positions, velocities, spring_states, lam, energy, spring_energies):
        '''
        Initialize a SystemState object

        Parameters
            positions -- coordinates of structure, numpy array (n_atoms, 3)
            velocities -- velocities for structure, same as coords
            spring_states -- state of each spring, numpy array (n_springs)
            lam -- lambda value, within [0, 1]
            energy -- total potential energy, including restraints
            spring_energies -- energy for each spring, numpy array (n_springs)

        Note that spring_energies contains the energy that the spring would have it was active,
        regardless of if it was actually active or not. To get the actual energy for each spring,
        you should take spring_states * spring_energies.

        '''
        self.positions = positions
        self.n_atoms = positions.shape[0]
        self.velocities = velocities
        self.spring_states = spring_states
        if self.spring_states is None:
            self.n_springs = 0
        else:
            self.n_springs = spring_states.shape
        self.lam = lam
        self.energy = energy
        self.spring_energies = spring_energies

        self._validate()

    #
    # private methods
    #
    def _validate(self):
        # check positions
        if not len(self.positions.shape) == 2:
            raise RuntimeError('positions should be a 2D array')
        if not self.positions.shape[1] == 3:
            raise RuntimeError('positions should be (n_atoms, 3) array')

        # check velocities
        if not self.positions.shape == self.velocities.shape:
            raise RuntimeError('velocities must have the same shape as positions')

        # check lambda
        if self.lam < 0 or self.lam > 1:
            raise RuntimeError('lam must be in [0,1]')

        # check springs
        if not (self.spring_states is None):
            if not len(self.spring_energies.shape) == 1:
                raise RuntimeError('spring_states must be 1D or None')

            if not self.spring_energies.shape == self.spring_states.shape:
                raise RuntimeError('spring_energies must have same shape as spring_states')
        else:
            if not (self.spring_energies is None):
                raise RuntimeError('if spring_states is None, spring_energies must be None')
