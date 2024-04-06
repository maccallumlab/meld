"""
Freeze atoms in the system
"""

from typing import List, Optional

from meld.system import indexing
from meld.system.builders.spec import SystemSpec


def freeze_atoms(
    spec: SystemSpec, atoms: Optional[List[indexing.AtomIndex]] = None
) -> SystemSpec:
    """
    Freeze atoms in the system

    Args:
        spec: system specification to modify
        atoms: the atoms to freeze, defaults to all atoms

    Returns:
        the modified system specification
    """
    n_particles = spec.system.getNumParticles()

    if atoms is None:
        indices = [i for i in range(n_particles)]

    else:
        indices = [int(i) for i in atoms]

    for i in range(n_particles):
        if i in indices:
            spec.system.setParticleMass(i, 0.0)

    return SystemSpec(
        spec.solvation,
        spec.system,
        spec.topology,
        spec.integrator,
        spec.barostat,
        spec.coordinates,
        spec.velocities,
        spec.box_vectors,
        spec.builder_info,
    )
