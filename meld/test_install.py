"""
Test MELD installation
"""

import random  # type: ignore
import sys  # type: ignore

import meldplugin  # type: ignore
import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import unit as u  # type: ignore

import meld  # type: ignore


def _create_test_system_and_coords():
    force = meldplugin.MeldForce(0, 1.0)

    n_particles = random.randint(4, 256)
    indices = list(range(0, n_particles))

    system = mm.System()
    for _ in range(n_particles):
        system.addParticle(1.0)
    coords = 5.0 * np.random.normal(size=(n_particles, 3))

    n_collections = random.randint(1, 16)
    for i in range(n_collections):
        n_groups = random.randint(1, 256)
        n_active_collection = random.randint(1, n_groups)

        groups = []
        for j in range(n_groups):
            n_rests = random.randint(1, 256)
            n_active_group = random.randint(1, n_rests)

            restraints = []
            for k in range(n_rests):
                if random.random() < 5:  # distance
                    i, j = random.sample(indices, 2)
                    force_const = random.uniform(0.0, 2500.0)
                    r1, r2, r3, r4 = sorted([random.uniform(0, 10) for _ in range(4)])
                    rest = force.addDistanceRestraint(i, j, r1, r2, r3, r4, force_const)
                    restraints.append(rest)
                else:  # torsion
                    pass

            group = force.addGroup(restraints, n_active_group)
            groups.append(group)
        collection = force.addCollection(groups, n_active_collection)

    system.addForce(force)
    return system, coords


def test_install():
    """
    Test this installation of MELD.

    This script will test the installation of MELD by constructing
    a random system and using it to compare forces on all
    available platforms.
    """

    print()
    print("openmm version:", mm.Platform.getOpenMMVersion())
    print("openmm git revision:", mm.version.git_revision)
    print("Meld Version:", meld.__version__)
    print("meldplugin version:", meldplugin.__version__)
    print()

    system, coords = _create_test_system_and_coords()

    n_platform = mm.Platform.getNumPlatforms()
    print("There are", n_platform, "platforms available:")
    print()

    forces = [None] * n_platform
    energies = [None] * n_platform
    platform_errors = dict()

    for i in range(n_platform):
        platform = mm.Platform.getPlatform(i)
        print(i + 1, platform.getName(), end=" ")
        integrator = mm.LangevinIntegrator(300.0, 1.0, 0.002)

        try:
            context = mm.Context(system, integrator, platform)
            context.setPositions(coords)
            state = context.getState(getForces=True, getEnergy=True)
            forces[i] = state.getForces(asNumpy=True).value_in_unit(
                u.kilojoule_per_mole / u.nanometer
            )
            energies[i] = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            print("- successfully computed forces.")
        except:
            print("- failed.")
            platform_errors[platform.getName()] = sys.exc_info()[1]

    for platform, error in platform_errors.items():
        print()
        print(platform, "platform error:", error)

    if n_platform > 1:
        print()
        print("Median differences between platforms:")
        print()
        for i in range(n_platform):
            if forces[i] is None:
                continue
            for j in range(i + 1, n_platform):
                if forces[j] is None:
                    continue

                forces_close = np.allclose(forces[i], forces[j], atol=1e-4, rtol=1e-4)
                median_force_error = np.median(np.abs(forces[i] - forces[j]))
                max_force_error = np.max(np.abs(forces[i] - forces[j]))

                energies_close = np.allclose(energies[i], energies[j])
                energy_error = np.abs(energies[i] - energies[j])

                name_i = mm.Platform.getPlatform(i).getName()
                name_j = mm.Platform.getPlatform(j).getName()

                if not forces_close or not energies_close:
                    msg = "** LARGE DIFFERENCE **"
                else:
                    msg = "** OK **"

                print(f"{name_i} vs {name_j}")
                print(f"\tenergy diff: {energy_error:g}")
                print(f"\tenegy good: {energies_close}")
                print(f"\tmedian force diff: {median_force_error:g}")
                print(f"\tmax force diff: {max_force_error:g}")
                print(f"\tforce good: {forces_close}")
                print(f"\t{msg}")
                print()


if __name__ == "__main__":
    test_install()
