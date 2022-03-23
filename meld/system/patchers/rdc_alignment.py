"""
Add RDC alignments to the system
"""

import openmm as mm  # type: ignore
from meld.system.builders.spec import SystemSpec, AmberSystemSpec
from .rdc_integrator import CustomRDCIntegrator


def add_rdc_alignment(
    spec: SystemSpec, num_alignments: int, alignment_mass: float = 1e4
) -> SystemSpec:
    """
    Add RDC alignments to the system

    Args:
        spec: system specification to modify
        num_alignments: number of alignments to add
        alignment_step_size: step size for alignment tensor update
    """
    assert num_alignments > 0, "Must add at least one alignment"
    assert alignment_mass > 0, "Alignment mass must be positive"
    if not isinstance(spec.integrator, mm.LangevinIntegrator):
        raise ValueError(
            f"When adding RDC alignment, expected a LangevinIntegrator, but found a {type(spec.integrator)}"
        )

    old_integrator = spec.integrator
    new_integrator = CustomRDCIntegrator(
        old_integrator.getTemperature(),
        old_integrator.getFriction(),
        old_integrator.getStepSize(),
        num_alignments,
        alignment_mass,
    )

    if isinstance(spec, AmberSystemSpec):
        return AmberSystemSpec(
            spec.solvation,
            spec.system,
            spec.topology,
            new_integrator,
            spec.barostat,
            spec.coordinates,
            spec.velocities,
            spec.box_vectors,
            spec.implicit_solvent_model,
        )

    raise ValueError("Only AmberSystemSpec is supported")
