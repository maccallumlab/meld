"""
Add virtual spin labels to the system.
"""

from copy import copy

import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import app
from openmm import unit as u  # type: ignore

from meld.system import indexing
from meld.system.builders.spec import SystemSpec


def add_virtual_spin_label(
    spec: SystemSpec, residue: indexing.ResidueIndex, label_type="OND", trials=20
) -> SystemSpec:
    """
    Adds a virtual spin label to the system.

    Args:
        spec: The system specification to modify.
        residue: The residue to add the spin label to.
        label_type: The type of spin label to add.
        trials: The number of Monte Carlo trials to use when adding the spin label.

    Returns:
        A modified system specification with an added spin label.
    """
    if spec.builder_info["builder"] == "amber":
        return _add_virtual_spin_label_amber(spec, residue, label_type, trials)
    else:
        raise ValueError("Unsupported SystemSpec type for virtual spin label")


def _add_virtual_spin_label_amber(
    spec: SystemSpec, residue: app.Residue, label_type: str, trials: int
):
    assert label_type == "OND"

    # Find the insertion point
    residue_ind = int(residue)
    residues = list(spec.topology.residues())
    residue = residues[residue_ind]
    insertion_point = max(a.index for a in residue.atoms()) + 1

    system = _create_spin_label_system(
        spec.system, spec.topology, residue, insertion_point, label_type
    )
    coords = _create_spin_label_coords(
        spec.coordinates,
        system,
        residue,
        insertion_point,
        trials,
    )
    vels = _create_spin_label_vels(spec.velocities, insertion_point)
    topology = _create_spin_label_topology(spec.topology, residue, insertion_point)

    return SystemSpec(
        spec.solvation,
        system,
        topology,
        spec.integrator,
        spec.barostat,
        coords,
        vels,
        spec.box_vectors,
        spec.builder_info,
    )


def _create_spin_label_system(
    old_system: mm.System,
    topology: app.Topology,
    residue: app.Residue,
    insertion_point: int,
    label_type: str,
):
    system = mm.System()

    # Add the particles to the system
    for i in range(insertion_point):
        mass = old_system.getParticleMass(i)
        system.addParticle(mass)
    system.addParticle(12.0)  # Add the spin label
    for i in range(insertion_point, old_system.getNumParticles()):
        mass = old_system.getParticleMass(i)
        system.addParticle(mass)

    # Update the constraints
    for i in range(old_system.getNumConstraints()):
        p1, p2, dist = old_system.getConstraintParameters(i)
        system.addConstraint(
            p1 + 1 if p1 >= insertion_point else p1,
            p2 + 1 if p2 >= insertion_point else p2,
            dist,
        )

    forces = old_system.getForces()

    # Create the custom angle force for the restricted angle potential
    # if one does not exist.
    force_names = [f.getName() for f in old_system.getForces()]
    if not "virtual_spin_label_angle" in force_names:
        custom_angle = mm.CustomAngleForce("0.5*k*(theta - theta0)^2 / sin(theta)^2")
        custom_angle.setName("virtual_spin_label_angle")
        custom_angle.addPerAngleParameter("theta0")
        custom_angle.addPerAngleParameter("k")
        forces.append(custom_angle)

    for force in forces:
        if isinstance(force, mm.NonbondedForce):
            _handle_spin_label_nonbonded(force, system, insertion_point, label_type)
        # elif isinstance(force, mm.GBSAOBCForce):
        #     _handle_spin_label_obc(force, system, insertion_point, label_type)
        elif isinstance(force, mm.CustomGBForce):
            _handle_spin_label_customgb(
                force, system, topology, insertion_point, label_type
            )
        elif isinstance(force, mm.GBSAOBCForce):
            _handle_spin_label_obc(force, system, topology, insertion_point, label_type)
        elif isinstance(force, mm.HarmonicBondForce):
            _handle_spin_label_harmonic_bond(
                force, system, residue, insertion_point, label_type
            )
        elif isinstance(force, mm.HarmonicAngleForce):
            _handle_spin_label_harmonic_angle(
                force, system, residue, insertion_point, label_type
            )
        elif (
            isinstance(force, mm.CustomAngleForce)
            and force.getName() == "virtual_spin_label_angle"
        ):
            _handle_spin_label_custom_angle(
                force, system, residue, insertion_point, label_type
            )
        elif isinstance(force, mm.PeriodicTorsionForce):
            _handle_spin_label_periodic_torsion(
                force, system, residue, insertion_point, label_type
            )
        elif isinstance(force, mm.CMMotionRemover):
            _handle_spin_label_cmmotionremover(force, system)
        elif isinstance(force, mm.CMAPTorsionForce):
            _handle_spin_label_cmap(force, system, insertion_point)
        else:
            raise RuntimeError(f"Unsupported force type {force}")

    return system


def _handle_spin_label_nonbonded(
    force: mm.NonbondedForce, system: mm.System, insertion_point: int, label_type: str
):
    new_force = mm.NonbondedForce()

    for i in range(insertion_point):
        q, sigma, eps = force.getParticleParameters(i)
        new_force.addParticle(q, sigma, eps)

    # We should be using the label type here, but we currently
    # only support OND, so we can hard code.
    new_force.addParticle(
        0.0,
        4.0 * u.angstrom * 2 ** (-1 / 6),  # convert from rmin to sigma
        0.05 * u.kilocalorie_per_mole,
    )

    for i in range(insertion_point, force.getNumParticles()):
        q, sigma, eps = force.getParticleParameters(i)
        new_force.addParticle(q, sigma, eps)

    # Update exceptions
    for i in range(force.getNumExceptions()):
        ind1, ind2, qq, sigma, eps = force.getExceptionParameters(i)
        new_force.addException(
            ind1 + 1 if ind1 >= insertion_point else ind1,
            ind2 + 1 if ind2 >= insertion_point else ind2,
            qq,
            sigma,
            eps,
        )

    # Note that this does not handle all of the possible functionality of
    # NonbondedForce. We only handle what MELD currently uses, so this may
    # need updating in the future.

    # Update parameters
    new_force.setExceptionsUsePeriodicBoundaryConditions(
        force.getExceptionsUsePeriodicBoundaryConditions()
    )
    new_force.setCutoffDistance(force.getCutoffDistance())
    new_force.setEwaldErrorTolerance(force.getEwaldErrorTolerance())
    new_force.setNonbondedMethod(force.getNonbondedMethod())
    new_force.setReactionFieldDielectric(force.getReactionFieldDielectric())
    new_force.setSwitchingDistance(force.getSwitchingDistance())
    new_force.setUseSwitchingFunction(force.getUseSwitchingFunction())
    new_force.setUseDispersionCorrection(force.getUseDispersionCorrection())
    alpha, nx, ny, nz = force.getPMEParameters()
    new_force.setPMEParameters(alpha, nx, ny, nz)

    system.addForce(new_force)


def _handle_spin_label_customgb(
    force: mm.CustomGBForce,
    system: mm.System,
    topology: app.Topology,
    insertion_point: int,
    label_type: str,
):
    new_force = mm.CustomGBForce()

    # Copy tabulated functions
    for i in range(force.getNumTabulatedFunctions()):
        name = force.getTabulatedFunctionName(i)
        func = copy(force.getTabulatedFunction(i))
        new_force.addTabulatedFunction(name, func)

    # Copy per-particle parameters
    for i in range(force.getNumPerParticleParameters()):
        name = force.getPerParticleParameterName(i)
        new_force.addPerParticleParameter(name)

    # Copy computed values
    for i in range(force.getNumComputedValues()):
        name, expr, val_type = force.getComputedValueParameters(i)
        new_force.addComputedValue(name, expr, val_type)

    # Copy energy terms
    for i in range(force.getNumEnergyTerms()):
        expr, val_type = force.getEnergyTermParameters(i)
        new_force.addEnergyTerm(expr, val_type)

    # Copy particles
    for i in range(insertion_point):
        new_force.addParticle(force.getParticleParameters(i))

    # Add the spin label. This should be based on label_type, but we only
    # support OND for now, so we can hard code. We base the parameters
    # on oxygen, so we need to find an oxygen to copy the parameters
    # from.
    oxygen_index = _find_oxygen(topology)
    oxygen_params = list(force.getParticleParameters(oxygen_index))
    oxygen_params[0] = 0.0  # set the charge to zero
    new_force.addParticle(oxygen_params)

    # Copy remaining particles
    for i in range(insertion_point, force.getNumParticles()):
        new_force.addParticle(force.getParticleParameters(i))

    # Update exclusions
    for i in range(force.getNumExclusions()):
        ind1, ind2 = force.getExclusionParticles(i)
        new_force.addExclusion(
            ind1 + 1 if ind1 >= insertion_point else ind1,
            ind2 + 1 if ind2 >= insertion_point else ind2,
        )

    # Copy over settings
    new_force.setCutoffDistance(force.getCutoffDistance())
    new_force.setNonbondedMethod(force.getNonbondedMethod())

    system.addForce(new_force)


def _handle_spin_label_obc(
    force: mm.GBSAOBCForce,
    system: mm.System,
    topology: app.Topology,
    insertion_point: int,
    label_type: str,
):
    new_force = mm.GBSAOBCForce()

    # Copy particles
    for i in range(insertion_point):
        q, r, s = force.getParticleParameters(i)
        new_force.addParticle(q, r, s)

    # Add the spin label. This should be based on label_type, but we only
    # support OND for now, so we can hard code. We base the parameters
    # on oxygen, so we need to find an oxygen to copy the parameters
    # from.
    oxygen_index = _find_oxygen(topology)
    _, r, s = force.getParticleParameters(oxygen_index)
    new_force.addParticle(0.0, r, s)

    # Copy remaining particles
    for i in range(insertion_point, force.getNumParticles()):
        q, r, s = force.getParticleParameters(i)
        new_force.addParticle(q, r, s)

    # Update settings
    new_force.setCutoffDistance(force.getCutoffDistance())
    new_force.setNonbondedMethod(force.getNonbondedMethod())
    new_force.setSolventDielectric(force.getSolventDielectric())
    new_force.setSoluteDielectric(force.getSoluteDielectric())
    new_force.setSurfaceAreaEnergy(force.getSurfaceAreaEnergy())

    system.addForce(new_force)


def _handle_spin_label_harmonic_bond(
    force: mm.HarmonicBondForce,
    system: mm.System,
    residue: app.Residue,
    insertion_point: int,
    label_type: str,
):
    new_force = mm.HarmonicBondForce()
    for i in range(force.getNumBonds()):
        ind1, ind2, length, k = force.getBondParameters(i)
        new_force.addBond(
            ind1 + 1 if ind1 >= insertion_point else ind1,
            ind2 + 1 if ind2 >= insertion_point else ind2,
            length,
            k,
        )

    # Add in the bond. We should use the label type, but
    # we only support OND, so we can hard code.
    ca_index = _find_atom_by_name(residue, "CA").index
    new_force.addBond(
        ca_index,
        insertion_point,
        7.9 * u.angstrom,
        1.0
        * u.kilocalorie_per_mole
        / u.angstrom**2,  # CHARMM does not have factor of 0.5
    )

    new_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    system.addForce(new_force)


def _handle_spin_label_harmonic_angle(
    force: mm.HarmonicAngleForce,
    system: mm.System,
    residue: app.Residue,
    insertion_point: int,
    label_type: str,
):
    new_force = mm.HarmonicAngleForce()
    for i in range(force.getNumAngles()):
        ind1, ind2, ind3, angle, k = force.getAngleParameters(i)
        new_force.addAngle(
            ind1 + 1 if ind1 >= insertion_point else ind1,
            ind2 + 1 if ind2 >= insertion_point else ind2,
            ind3 + 1 if ind3 >= insertion_point else ind3,
            angle,
            k,
        )

    new_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    system.addForce(new_force)


def _handle_spin_label_custom_angle(
    force: mm.HarmonicAngleForce,
    system: mm.System,
    residue: app.Residue,
    insertion_point: int,
    label_type: str,
):
    new_force = mm.CustomAngleForce("0.5*k*(theta - theta0)^2 / sqrt(sin(theta))")
    new_force.setName("virtual_spin_label_angle")
    new_force.addPerAngleParameter("theta0")
    new_force.addPerAngleParameter("k")

    # Add all of the old angles
    for i in range(force.getNumAngles()):
        ind1, ind2, ind3, params = force.getAngleParameters(i)
        new_force.addAngle(
            ind1 + 1 if ind1 >= insertion_point else ind1,
            ind2 + 1 if ind2 >= insertion_point else ind2,
            ind3 + 1 if ind3 >= insertion_point else ind3,
            params,
        )

    # Add in the angle. We should use the label type, but
    # we only support OND, so we can hard code.
    ca_index = _find_atom_by_name(residue, "CA").index
    cb_index = _find_atom_by_name(residue, "CB").index
    new_force.addAngle(
        cb_index,
        ca_index,
        insertion_point,
        [
            46.0 * u.degree,
            2.0
            * u.kilocalorie_per_mole
            / u.radian**2,  # CHARMM does not have factor of 0.5
        ],
    )

    new_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    system.addForce(new_force)


def _handle_spin_label_periodic_torsion(
    force: mm.PeriodicTorsionForce,
    system: mm.System,
    residue: app.Residue,
    insertion_point: int,
    label_type: str,
):
    new_force = mm.PeriodicTorsionForce()
    for i in range(force.getNumTorsions()):
        ind1, ind2, ind3, ind4, period, phase, k = force.getTorsionParameters(i)
        new_force.addTorsion(
            ind1 + 1 if ind1 >= insertion_point else ind1,
            ind2 + 1 if ind2 >= insertion_point else ind2,
            ind3 + 1 if ind3 >= insertion_point else ind3,
            ind4 + 1 if ind4 >= insertion_point else ind4,
            period,
            phase,
            k,
        )

    # Add in the bond. We should use the label type, but
    # we only support OND, so we can hard code.
    n_index = _find_atom_by_name(residue, "N").index
    ca_index = _find_atom_by_name(residue, "CA").index
    cb_index = _find_atom_by_name(residue, "CB").index
    new_force.addTorsion(
        insertion_point,
        cb_index,
        ca_index,
        n_index,
        1,
        43.0 * u.degree,
        1.9 * u.kilocalorie_per_mole,
    )

    new_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    system.addForce(new_force)


def _handle_spin_label_cmmotionremover(force: mm.CMMotionRemover, system: mm.System):
    new_force = mm.CMMotionRemover()
    new_force.setFrequency(force.getFrequency())
    system.addForce(new_force)


def _handle_spin_label_cmap(
    force: mm.CMAPTorsionForce, system: mm.System, insertion_point: int
):
    new_force = mm.CMAPTorsionForce()

    # Copy over the maps
    for i in range(force.getNumMaps()):
        size, energy = force.getMapParameters(i)
        new_force.addMap(size, energy)

    for i in range(force.getNumTorsions()):
        map, a1, a2, a3, a4, b1, b2, b3, b4 = force.getTorsionParameters(i)
        new_force.addTorsion(
            map,
            a1 + 1 if a1 >= insertion_point else a1,
            a2 + 1 if a2 >= insertion_point else a2,
            a3 + 1 if a3 >= insertion_point else a3,
            a4 + 1 if a4 >= insertion_point else a4,
            b1 + 1 if b1 >= insertion_point else b1,
            b2 + 1 if b2 >= insertion_point else b2,
            b3 + 1 if b3 >= insertion_point else b3,
            b4 + 1 if b4 >= insertion_point else b4,
        )

    new_force.setUsesPeriodicBoundaryConditions(force.usesPeriodicBoundaryConditions())
    system.addForce(new_force)


def _find_atom_by_name(residue: app.Residue, name: str) -> app.Atom:
    for atom in residue.atoms():
        if atom.name == name:
            return atom

    raise RuntimeError("Could not find CA atom in residue")


def _find_oxygen(topology: app.Topology) -> int:
    for atom in topology.atoms():
        if atom.element.symbol == "O":
            return atom.index
    raise RuntimeError("Could not find oxygen to copy parameters from.")


def _create_spin_label_coords(
    old_coords: np.ndarray,
    system: mm.System,
    residue: app.Residue,
    insertion_point: int,
    trials: int,
) -> np.ndarray:
    ca_index = _find_atom_by_name(residue, "CA").index
    ca_coords = old_coords[ca_index, :]

    # Decide on the new position using MC
    integrator = mm.LangevinIntegrator(300.0, 1.0, 1.0)
    context = mm.Context(system, integrator)
    best_energy = 9e99
    best_coords = None
    for _ in range(trials):
        direction = np.random.normal(0.0, 1.0, 3)
        direction = direction / np.linalg.norm(direction)
        magnitude = np.random.normal(0.79, 0.125)
        label_position = ca_coords + magnitude * direction
        trial_coords = np.concatenate(
            [
                old_coords[:insertion_point, :],
                label_position.reshape(1, 3),
                old_coords[insertion_point:, :],
            ],
            axis=0,
        )
        context.setPositions(trial_coords)
        state = context.getState(getEnergy=True)
        energy = state.getPotentialEnergy().value_in_unit(u.kilojoules_per_mole)
        if energy < best_energy:
            best_energy = energy
            best_coords = trial_coords
    assert best_coords is not None
    return best_coords


def _create_spin_label_vels(old_vels: np.ndarray, insertion_point: int) -> np.ndarray:
    return np.concatenate(
        [
            old_vels[:insertion_point, :],
            np.zeros((1, 3)),
            old_vels[insertion_point:, :],
        ],
        axis=0,
    )


def _create_spin_label_topology(
    old_topology: app.Topology, label_residue: app.Residue, insertion_point: int
) -> app.Topology:
    new_topology = app.Topology()

    # Re-create all chains
    chain_map = {}
    for chain in old_topology.chains():
        new_chain = new_topology.addChain(id=chain.id)
        chain_map[chain] = new_chain

    # Re-create all residues
    res_map = {}
    for residue in old_topology.residues():
        new_residue = new_topology.addResidue(
            residue.name, chain_map[residue.chain], insertionCode=residue.insertionCode
        )
        res_map[residue] = new_residue

    # Add atoms before label
    atoms = list(old_topology.atoms())
    atom_map = {}
    for atom in atoms[:insertion_point]:
        new_atom = new_topology.addAtom(atom.name, atom.element, res_map[atom.residue])
        atom_map[atom] = new_atom
    # Add in spin label
    spin_label = new_topology.addAtom(
        "OND", app.Element.getBySymbol("C"), res_map[label_residue]
    )
    # Add remaining atoms
    for atom in atoms[insertion_point:]:
        new_atom = new_topology.addAtom(atom.name, atom.element, res_map[atom.residue])
        atom_map[atom] = new_atom

    # Re-create all the bonds
    for bond in old_topology.bonds():
        new_topology.addBond(
            atom_map[bond.atom1], atom_map[bond.atom2], order=bond.order, type=bond.type
        )
    new_topology.addBond(spin_label, atom_map[_find_atom_by_name(label_residue, "CA")])

    return new_topology
