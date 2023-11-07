"""
System specifications

A system specification describes many details about the system to
be modeled. A `SystemSpec` can be modified by a patcher to produce
a new `SystemSpec`. Then the final `SystemSpec` can be turned into
a `System` by calling `finalize`.
"""

from typing import Optional

import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import app

from meld.system.indexing import setup_indexing
from meld.system.meld_system import System


class SystemSpec:
    """
    A system specification.

    Attributes:
        solvation: implicit or explicit
        system: the system to be modeled
        topology: the topology of the system
        integrator: the integrator to be used
        barostat: the barostat to be used
        coordinates: the initial coordinates of the system
        velocities: the initial velocities of the system
        box_vectors: the box vectors of the system
        builder_info: extra information about how the system was built
    """

    solvation: str
    system: mm.System
    topology: app.Topology
    integrator: mm.LangevinIntegrator
    barostat: Optional[mm.MonteCarloBarostat]
    coordinates: np.ndarray
    velocities: np.ndarray
    box_vectors: Optional[np.ndarray]
    builder_info: dict

    def __init__(
        self,
        solvation: str,
        system: mm.System,
        topology: app.Topology,
        integrator: mm.LangevinIntegrator,
        barostat: Optional[mm.MonteCarloBarostat],
        coordinates: np.ndarray,
        velocities: np.ndarray,
        box_vectors: Optional[np.ndarray],
        builder_info: Optional[dict] = None,
    ):
        assert system.getNumParticles() == coordinates.shape[0]
        assert system.getNumParticles() == velocities.shape[0]
        assert coordinates.shape[1] == 3
        assert velocities.shape[1] == 3

        self.solvation = solvation
        self.system = system
        self.topology = topology
        self.integrator = integrator
        self.barostat = barostat
        self.coordinates = coordinates
        self.velocities = velocities
        self.box_vectors = box_vectors
        self.builder_info = builder_info if builder_info is not None else {}
        self.index = setup_indexing(self.topology)

    def finalize(self) -> System:
        """
        Finalize the system specification.

        Returns:
            The system described by this specification.
        """
        return System(
            self.solvation,
            self.system,
            self.topology,
            self.integrator,
            self.barostat,
            self.coordinates,
            self.velocities,
            self.box_vectors,
            self.builder_info,
        )
