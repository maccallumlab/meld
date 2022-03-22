from ..indexing import setup_indexing
from ..meld_system import System


class SystemSpec:
    def __init__(
        self,
        solvation,
        system,
        topology,
        integrator,
        barostat,
        coordinates,
        velocities,
        box_vectors,
    ):
        self.solvation = solvation
        self.system = system
        self.topology = topology
        self.integrator = integrator
        self.barostat = barostat
        self.coordinates = coordinates
        self.velocities = velocities
        self.box_vectors = box_vectors
        self.index = setup_indexing(self.topology)

    def finalize(self):
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
    def __init__(
        self,
        solvation,
        system,
        topology,
        integrator,
        barostat,
        coordinates,
        velocities,
        box_vectors,
        implicit_solvent_model,
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
