class SystemState(object):
    """
    Class to hold the state of a system.

    :param positions: coordinates of structure, ``numpy.array(n_atoms, 3)``
    :param velocities: velocities for structure, same as coords
    :param alpha: alpha value, within ``[0, 1]``
    :param energy: total potential energy, including restraints
    
    """
    def __init__(self, positions, velocities, alpha, energy):
        self.positions = positions
        self.n_atoms = positions.shape[0]
        self.velocities = velocities
        self.alpha = alpha
        self.energy = energy

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

        # check alpha
        if self.alpha < 0 or self.alpha > 1:
            raise RuntimeError('alpha must be in [0,1]')
