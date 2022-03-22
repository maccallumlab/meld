"""
Add RDC alignment tensor particles to the system.
"""

import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import app
from openmm import unit as u  # type: ignore
from meld.system import indexing
from meld.system.builders.spec import SystemSpec, AmberSystemSpec
from typing import Tuple


def add_rdc_alignment(spec: SystemSpec) -> Tuple[SystemSpec, indexing.ResidueIndex]:
    """
    Adds an RDC alignment term to the system.

    Args:
        spec: system specification to modify

    Returns:
        modified system specification
    """
    if isinstance(spec, AmberSystemSpec):
        return _add_rdc_alignment_amber(spec)
    else:
        raise ValueError("Unsupported system spec type for rdc alignment")


def _add_rdc_alignment_amber(
    spec: AmberSystemSpec,
) -> Tuple[AmberSystemSpec, indexing.ResidueIndex]:
    system = spec.system
    topology = spec.topology

    # add two particles
    p1 = system.addParticle(12.0 * u.amu)
    p2 = system.addParticle(12.0 * u.amu)

    # add nonbonded interactions
    _update_nb_force_rdc(system)
    if spec.solvation == "implicit":
        _update_gb_force_rdc(spec.implicit_solvent_model, system)

    # add residues to topology
    last_chain = list(topology.chains())[-1]
    residue = topology.addResidue("SDM", last_chain)
    # add atoms to residues
    a1 = topology.addAtom("S1", app.Element.getByMass(12.0), residue)
    a2 = topology.addAtom("S2", app.Element.getByMass(12.0), residue)

    assert p1 == a1.index
    assert p2 == a2.index

    # Add new coordinates and velocities
    new_coords = np.concatenate([spec.coordinates, np.random.randn(2, 3)])
    new_vels = np.concatenate([spec.velocities, np.zeros((2, 3))])

    new_spec = AmberSystemSpec(
        spec.solvation,
        spec.system,
        spec.topology,
        spec.integrator,
        spec.barostat,
        new_coords,
        new_vels,
        spec.box_vectors,
        spec.implicit_solvent_model,
    )

    return new_spec, indexing.ResidueIndex(residue.index)


def _update_gb_force_rdc(implicit_solvation_model: str, system: mm.System):
    if implicit_solvation_model == "vacuum":
        return
    elif implicit_solvation_model == "obc":
        _update_gb_force_rdc_obc(system)
    elif implicit_solvation_model == "gbNeck" or implicit_solvation_model == "gbNeck2":
        _update_gb_force_rdc_gbneck(system)
    else:
        raise ValueError(
            f"Unsupported implicit solvent model {implicit_solvation_model}"
        )


def _update_gb_force_rdc_obc(system: mm.System):
    force = _get_obc_force(system)

    # Add two new particles with zeros for all parameters.
    # It's not clear if this will work, but it is not
    # possible to add exclusions with GBSAOBCForce.
    force.addParticle(0.0, 0.0, 0.0)
    force.addParticle(0.0, 0.0, 0.0)


def _update_gb_force_rdc_gbneck(system: mm.System):
    force = _get_customgb_force(system)
    n_particles = force.getNumParticles()

    # Add two new particles. We use the same parameters
    # as for the first atom in the system. These parameters
    # will not matter, as all of the interactions involving
    # the new particles will be excluded.
    params = force.getParticleParameters(0)
    p1 = force.addParticle(params)
    p2 = force.addParticle(params)

    # Add exclusions between all particles and the new ones
    for i in range(0, n_particles):
        force.addExclusion(i, p1)
        force.addExclusion(i, p2)
    force.addExclusion(p1, p2)


def _get_customgb_force(system: mm.System) -> mm.CustomGBForce:
    for force in system.getForces():
        if isinstance(force, mm.CustomGBForce):
            return force
    raise ValueError("No CustomGBForce found in system")


def _get_obc_force(system: mm.System) -> mm.GBSAOBCForce:
    for force in system.getForces():
        if isinstance(force, mm.GBSAOBCForce):
            return force
    raise ValueError("No GBSAOBCForce found in system")


def _update_nb_force_rdc(system: mm.System):
    nb_force = _get_nb_force(system)
    p1 = nb_force.addParticle(0.0, 0.0, 0.0)
    p2 = nb_force.addParticle(0.0, 0.0, 0.0)
    for i in range(0, p1):
        nb_force.addException(i, p1, 0.0, 0.0, 0.0)
        nb_force.addException(i, p2, 0.0, 0.0, 0.0)
    nb_force.addException(p1, p2, 0.0, 0.0, 0.0)


def _get_nb_force(system: mm.System) -> mm.NonbondedForce:
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            return force
    raise ValueError("No nonbonded force found in system")
