"""
This module implements patchers classes that can modify
the system to include new atoms and/or paramters.
"""

import numpy as np  # type: ignore
import openmm as mm  # type: ignore
from openmm import app  # type: ignore
from openmm import unit as u  # type: ignore
from meld.system import indexing
from meld.system.builders.spec import SystemSpec, AmberSystemSpec


def add_rdc_alignment(spec: SystemSpec) -> SystemSpec:
    """
    Adds an RDC alignment term to the system.
    """
    if isinstance(spec, AmberSystemSpec):
        return _add_rdc_alignment_amber(spec)
    else:
        raise ValueError("Unsupported system spec type for rdc alignment")


def _add_rdc_alignment_amber(spec: AmberSystemSpec):
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


def _update_gb_force_rdc(implicit_solvation_model, system):
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


def _update_gb_force_rdc_obc(system):
    force = _get_obc_force(system)

    # Add two new particles with zeros for all parameters.
    # It's not clear if this will work, but it is not
    # possible to add exclusions with GBSAOBCForce.
    force.addParticle(0.0, 0.0, 0.0)
    force.addParticle(0.0, 0.0, 0.0)


def _update_gb_force_rdc_gbneck(system):
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


def _get_customgb_force(system):
    for force in system.getForces():
        if isinstance(force, mm.CustomGBForce):
            return force
    raise ValueError("No CustomGBForce found in system")


def _get_obc_force(system):
    for force in system.getForces():
        if isinstance(force, mm.GBSAOBCForce):
            return force
    raise ValueError("No GBSAOBCForce found in system")


def _update_nb_force_rdc(system):
    nb_force = _get_nb_force(system)
    p1 = nb_force.addParticle(0.0, 0.0, 0.0)
    p2 = nb_force.addParticle(0.0, 0.0, 0.0)
    for i in range(0, p1):
        nb_force.addException(i, p1, 0.0, 0.0, 0.0)
        nb_force.addException(i, p2, 0.0, 0.0, 0.0)
    nb_force.addException(p1, p2, 0.0, 0.0, 0.0)


def _get_nb_force(system):
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            return force
    raise ValueError("No nonbonded force found in system")


# TODO These need to be re-written

# class VirtualSpinLabelPatcher(PatcherBase):
#     """
#     Patch residues to include virtual spin label sites.
#     """

#     ALLOWED_TYPES = ["OND"]

#     bond_params = {
#         "OND": (1.2 * u.kilocalorie_per_mole / u.angstrom ** 2, 8.0 * u.angstrom)
#     }
#     angle_params = {
#         "OND": (2.0 * u.kilocalorie_per_mole / u.radian ** 2, 46.0 * u.degrees)
#     }
#     tors_params = {"OND": (1.9 * u.kilocalorie_per_mole, 1, 240.0 * u.degrees)}
#     lj_params = {"OND": (4.0 * u.angstrom / 2.0, 0.05 * u.kilocalorie_per_mole)}

#     def __init__(self, params: Dict[indexing.ResidueIndex, str], explicit_solvent: bool = False):
#         """
#         Initialize a VirtualSpinLabelPatcher

#         Args:
#             params: A dictionary of resdiue indices and corresponding spin label types.
#             explicit_solvent: A flag indicating if this is an explicit solvent simulation.

#         .. note::
#            Currently, 'OND' is the only supported spin label type, with parameters
#            taken from:
#            Islam, Stein, Mchaourab, and Roux, J. Phys. Chem. B 2013, 117, 4740-4754.
#         """
#         for key in params:
#             assert isinstance(key, indexing.ResidueIndex)
#             assert params[key] in self.ALLOWED_TYPES

#         self.params = params
#         self.explicit = explicit_solvent

#     def patch(self, top_string: str, crd_string: str) -> Tuple[str, str]:
#         INTOP = "in.top"
#         INRST = "in.rst"
#         OUTTOP = "out.top"
#         OUTRST = "out.rst"

#         with util.in_temp_dir():
#             with open(INTOP, "wt") as outfile:
#                 outfile.write(top_string)
#             with open(INRST, "wt") as outfile:
#                 outfile.write(crd_string)

#             topol = pmd.load_file(INTOP)
#             crd = pmd.load_file(INRST)
#             topol.coordinates = crd.coordinates

#             self._add_particles(topol)

#             topol.write_parm(OUTTOP)
#             topol.write_rst7(OUTRST)
#             with open(OUTTOP, "rt") as infile:
#                 top_string = infile.read()
#             with open(OUTRST, "rt") as infile:
#                 crd_string = infile.read()
#         return top_string, crd_string

#     def finalize(self, system: interfaces.ISystem):
#         for res_index in self.params:
#             site_type = self.params[res_index]

#             # find the atoms
#             n_index = system.index.atom(int(res_index), "N")
#             ca_index = system.index.atom(int(res_index), "CA")
#             cb_index = system.index.atom(int(res_index), "CB")
#             ond_index = system.index.atom(int(res_index), "OND")

#             # add the extra parameters
#             system.add_extra_bond(
#                 i=ca_index,
#                 j=ond_index,
#                 length=self.bond_params[site_type][1],
#                 force_constant=self.bond_params[site_type][0],
#             )
#             system.add_extra_angle(
#                 i=cb_index,
#                 j=ca_index,
#                 k=ond_index,
#                 angle=self.angle_params[site_type][1],
#                 force_constant=self.angle_params[site_type][0],
#             )
#             system.add_extra_torsion(
#                 i=n_index,
#                 j=ca_index,
#                 k=cb_index,
#                 l=ond_index,
#                 phase=self.tors_params[site_type][1],
#                 multiplicity=1,
#                 energy=self.tors_params[site_type][0],
#             )

#     def _add_particles(self, topol):
#         err_msg = "Unknown spin label type {{}}. Allowed values are: {}"
#         err_msg = err_msg.format(", ".join(self.ALLOWED_TYPES))

#         # we use the same radius and screen as for oxygen
#         if not self.explicit:
#             radius, screen = self._find_radius_and_screen(topol)

#         # find all the unique types of spin labels
#         types = set(self.params.values())

#         for key in self.params:
#             if self.params[key] not in self.ALLOWED_TYPES:
#                 raise ValueError(err_msg.format(self.params[key]))

#             # create the particle
#             atom = pmd.Atom(None, 8, "OND", "OND", 0.0, 16.00)
#             if not self.explicit:
#                 atom.solvent_radius = radius
#                 atom.screen = screen

#             # add to system
#             topol.add_atom_to_residue(atom, topol.residues[int(key)])

#             # find the other atoms
#             ca = topol.view[f":{int(key)+1},@CA"].atoms[0]
#             cb = topol.view[f":{int(key)+1},@CB"].atoms[0]
#             n = topol.view[f":{int(key)+1},@N"].atoms[0]

#             # Mark that the spin label and CA are connected.
#             # This will not actually add a bond to the potential,
#             # but will mark the connectivity between the atoms.
#             atom.bond_to(ca)

#             # set position
#             ca_pos = np.array((ca.xx, ca.xy, ca.xz))
#             n_pos = np.array((n.xx, n.xy, n.xz))
#             cb_pos = np.array((cb.xx, cb.xy, cb.xz))

#             direction = (ca_pos - cb_pos) / np.linalg.norm(ca_pos - cb_pos)
#             new_pos = (
#                 ca_pos
#                 - self.bond_params[self.params[key]][1].value_in_unit(u.nanometer)
#                 * direction
#                 + ca_pos
#                 - n_pos
#             )

#             atom.xx = new_pos[0]
#             atom.xy = new_pos[1]
#             atom.xz = new_pos[2]
#         topol.remake_parm()

#         # setup the new non-bonded parameters
#         for t in types:
#             indices = [index for index in self.params if self.params[index] == t]
#             selection_string = "(:{residue_mask})&(@{atom_name})".format(
#                 residue_mask=",".join(str(i) for i in indices), atom_name=t
#             )
#             action = pmd.tools.addLJType(
#                 topol,
#                 selection_string,
#                 radius=self.lj_params[t][0].value_in_unit(u.angstrom),
#                 epsilon=self.lj_params[t][1].value_in_unit(u.kilocalorie_per_mole),
#             )
#             action.execute()

#     def _find_radius_and_screen(self, topol):
#         for atom in topol.atoms:
#             if atom.atomic_number == 8:
#                 return atom.solvent_radius, atom.screen
