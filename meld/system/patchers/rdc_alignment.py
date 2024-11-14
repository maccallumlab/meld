"""
Add RDC alignments to the system
"""

import math

import openmm as mm  # type: ignore
from openmm import unit as u

from meld.system.builders.spec import SystemSpec

GAS_CONSTANT = 8.314e-3


def add_rdc_alignment(
    spec: SystemSpec,
    num_alignments: int,
    alignment_mass: float = 1000.0,
    scale_factor: float = 1.0e-4,
) -> SystemSpec:
    """
    Add RDC alignments to the system

    Args:
        spec: system specification to modify
        num_alignments: number of alignments to add
        alignment_step_size: step size for alignment tensor update
        scale_factor: scale factor for the alignment tensor

    Returns:
        the modified system specification

    Note:
        The alignment tensor is encoded as five independent components,
        s1 through s5. The components can be combined to produce the
        aligmnet tensor as follows::

            s1 - s2      s3        s4
            s3        -s1 - s2     s5
            s4           s5      2*s2

    Note:
        The values of the alignment tensor are typically on the order
        of 1e-4. Setting `scale_factor=1e-4` serves to scale these values
        to be on the order of unity.

    Note:
        The mass of alignment tensor determines how quickly the alignment
        tensor responds to changes in the structure. A mass that is too
        small will result instability and crashes. A mass that is too large
        will lead to slow exploration as the alignment tensor will change
        very slowly. The default value of 1000.0 is a reasonable compromise
        that results in the alignment tensor changing over ~10-20 ps.

    """
    assert num_alignments > 0, "Must add at least one alignment"
    assert alignment_mass > 0, "Alignment mass must be positive"
    if not isinstance(spec.integrator, mm.LangevinIntegrator):
        raise ValueError(
            f"When adding RDC alignment, expected a LangevinIntegrator, but found a {type(spec.integrator)}"
        )

    old_integrator = spec.integrator
    new_integrator = _create_rdc_integrator(
        num_alignments,
        old_integrator.getTemperature(),
        old_integrator.getFriction(),
        old_integrator.getStepSize(),
        alignment_mass,
    )

    builder_info = spec.builder_info
    assert "has_alignments" not in builder_info
    builder_info["has_alignments"] = True
    builder_info["num_alignments"] = num_alignments
    builder_info["alignment_scale_factor"] = scale_factor

    return SystemSpec(
        spec.solvation,
        spec.system,
        spec.topology,
        new_integrator,
        spec.barostat,
        spec.coordinates,
        spec.velocities,
        spec.box_vectors,
        builder_info,
    )


def _create_rdc_integrator(
    num_alignments, temperature, friction, timestep, alignment_mass
):
    assert num_alignments > 0

    if isinstance(temperature, u.Quantity):
        temperature = temperature.value_in_unit(u.kelvin)
    assert temperature > 0

    if isinstance(friction, u.Quantity):
        friction = friction.value_in_unit((1 / u.picosecond).unit)
    assert friction > 0

    if isinstance(timestep, u.Quantity):
        timestep = timestep.value_in_unit(u.picosecond)
    assert timestep > 0

    integrator = mm.CustomIntegrator(timestep)

    # Setup variables
    integrator.addGlobalVariable("a", math.exp(-friction * timestep))
    integrator.addGlobalVariable("b", math.sqrt(1 - math.exp(-2 * friction * timestep)))
    integrator.addGlobalVariable("kT", GAS_CONSTANT * temperature)
    # Add in the alignment tensor components
    for i in range(num_alignments):
        for j in range(5):
            integrator.addGlobalVariable(f"rdc_{i}_s{j + 1}_vel", 0.0)
    integrator.addPerDofVariable("x1", 0)

    # Steps during integration
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v + dt*f/m")
    for i in range(num_alignments):
        for j in range(5):
            integrator.addComputeGlobal(
                f"rdc_{i}_s{j + 1}_vel",
                f"rdc_{i}_s{j + 1}_vel - dt * deriv(energy, rdc_{i}_s{j + 1}) / {alignment_mass}",
            )
    integrator.addConstrainVelocities()
    integrator.addComputePerDof("x", "x + 0.5*dt*v")
    for i in range(num_alignments):
        for j in range(5):
            integrator.addComputeGlobal(
                f"rdc_{i}_s{j + 1}",
                f"rdc_{i}_s{j + 1} + 0.5*dt*rdc_{i}_s{j + 1}_vel",
            )
    integrator.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
    for i in range(num_alignments):
        for j in range(5):
            integrator.addComputeGlobal(
                f"rdc_{i}_s{j + 1}_vel",
                f"a*rdc_{i}_s{j + 1}_vel + b*sqrt(kT/{alignment_mass})*gaussian",
            )
    integrator.addComputePerDof("x", "x + 0.5*dt*v")
    for i in range(num_alignments):
        for j in range(5):
            integrator.addComputeGlobal(
                f"rdc_{i}_s{j + 1}",
                f"rdc_{i}_s{j + 1} + 0.5*dt*rdc_{i}_s{j + 1}_vel",
            )
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions()
    integrator.addComputePerDof("v", "v + (x-x1)/dt")

    return integrator
