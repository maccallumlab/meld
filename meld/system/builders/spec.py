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
from ..indexing import setup_indexing
from ..meld_system import System


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
    """

    solvation: str
    system: mm.System
    topology: app.Topology
    integrator: mm.LangevinIntegrator
    barostat: Optional[mm.MonteCarloBarostat]
    coordinates: np.ndarray
    velocities: np.ndarray
    box_vectors: Optional[np.ndarray]

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
        )


class AmberSystemSpec(SystemSpec):
    """
    An Amber system specification

    Attributes:
        solvation: implicit or explicit
        system: the system to be modeled
        topology: the topology of the system
        integrator: the integrator to be used
        barostat: the barostat to be used
        coordinates: the initial coordinates of the system
        velocities: the initial velocities of the system
        box_vectors: the box vectors of the system
        implict_solvent_model: the implicit solvent model used
    """

    solvation: str
    system: mm.System
    topology: app.Topology
    integrator: mm.LangevinIntegrator
    barostat: Optional[mm.MonteCarloBarostat]
    coordinates: np.ndarray
    velocities: np.ndarray
    box_vectors: Optional[np.ndarray]
    implicit_solvent_model: str

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
        implicit_solvent_model: str,
    ):
        super().__init__(
            solvation,
            system,
            topology,
            integrator,
            barostat,
            coordinates,
            velocities,
            box_vectors,
        )
        self.implicit_solvent_model = implicit_solvent_model
