#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld.system.param_sampling import ParameterState
from typing import Optional
import numpy as np  # type: ignore


class SystemState:
    """
    Class to hold the state of a system.

    :param positions: coordinates of structure, ``numpy.array(n_atoms, 3)``
    :param velocities: velocities for structure, same as coords
    :param alpha: alpha value, within ``[0, 1]``
    :param energy: total potential energy, including restraints
    :param box_vector: box vectors, ``numpy.array(3)``
    """

    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        alpha: float,
        energy: float,
        box_vector: np.ndarray,
        parameters: Optional[ParameterState] = None,
    ) -> None:
        self.positions = positions
        self.velocities = velocities
        self.box_vector = box_vector
        self.n_atoms = positions.shape[0]
        self.alpha = alpha
        self.energy = energy
        if parameters is None:
            self.parameters = ParameterState(
                discrete=np.array([], dtype=np.int32),
                continuous=np.array([], dtype=np.float64),
            )
        else:
            self.parameters = parameters

        self._validate()

    #
    # private methods
    #
    def _validate(self):
        # check positions
        if not len(self.positions.shape) == 2:
            raise RuntimeError("positions should be a 2D array")
        if not self.positions.shape[1] == 3:
            raise RuntimeError("positions should be (n_atoms, 3) array")

        # check velocities
        if not self.positions.shape == self.velocities.shape:
            raise RuntimeError("velocities must have the same shape as positions")

        # check box vectors
        if self.box_vector is not None:
            if not len(self.box_vector) == 3:
                raise RuntimeError("len(box_vectors) != 3")

        # check alpha
        if self.alpha < 0 or self.alpha > 1:
            raise RuntimeError("alpha must be in [0,1]")
