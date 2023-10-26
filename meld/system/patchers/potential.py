"""
Remove potential and freeze the system
"""

import openmm as mm  # type: ignore

from meld.system.builders.spec import SystemSpec


def remove_potential(spec: SystemSpec) -> SystemSpec:
    """
    Remove the potential from the system and freeze atoms.

    This can be useful when using peak mapping to try to
    infer assignments based on a known or predicted structure.

    Args:
        spec: system specification to modify

    Returns:
        the modified system specification
    """
    new_system = mm.System()
    for _ in range(spec.system.getNumParticles()):
        new_system.addParticle(0.0)

    return SystemSpec(
        spec.solvation,
        new_system,
        spec.topology,
        spec.integrator,
        spec.barostat,
        spec.coordinates,
        spec.velocities,
        spec.box_vectors,
        spec.builder_info,
    )
