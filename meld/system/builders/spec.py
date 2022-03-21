from ..indexing import setup_indexing
from ..meld_system import System


class OpenMMSpec:
    def __init__(self, system, topology, integrator, barostat, coordinates):
        self.system = system
        self.topology = topology
        self.integrator = integrator
        self.barostat = barostat
        self.coordinates = coordinates
        self.index = setup_indexing(self.topology)

    def finalize(self):
        return System(
            self.system, self.topology, self.integrator, self.barostat, self.coordinates
        )
