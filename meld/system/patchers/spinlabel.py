'''
This module implements dummy spin label sites
for EPR and PRE experiments.

'''

import parmed as pmd
from meld import util
from parmed import unit as u
import numpy as np


class VirtualSpinLabel(object):
    ALLOWED_TYPES = ['OND']

    bond_params = {'OND': (0.6 * u.kilocalorie_per_mole / u.angstrom**2, 8. * u.angstrom)}
    angle_params = {'OND': (1.0 * u.kilocalorie_per_mole / u.radian**2, 52. * u.degrees)}
    tors_params = {'OND': (1.9 * u.kilocalorie_per_mole, 1, 60. * u.degrees)}
    lj_params = {'OND': (4.0 * u.angstrom / 2.0, 0.05 * u.kilocalorie_per_mole)}

    def __init__(self, params, explicit_solvent=False):
        self.params = params
        self.explicit = explicit_solvent

    def patch(self, top_string, crd_string):
        INTOP = 'in.top'
        INRST = 'in.rst'
        OUTTOP = 'out.top'
        OUTRST = 'out.rst'

        with util.in_temp_dir():
            with open(INTOP, 'wt') as outfile:
                outfile.write(top_string)
            with open(INRST, 'wt') as outfile:
                outfile.write(crd_string)

            topol = pmd.load_file(INTOP)
            crd = pmd.load_file(INRST)
            topol.coordinates = crd.coordinates

            self._add_particles(topol)

            topol.write_parm(OUTTOP)
            topol.write_rst7(OUTRST)
            with open(OUTTOP, 'rt') as infile:
                top_string = infile.read()
            with open(OUTRST, 'rt') as infile:
                crd_string = infile.read()
        return top_string, crd_string

    def _add_particles(self, topol):
        err_msg = 'Unknown spin label type {{}}. Allowed values are: {}'
        err_msg = err_msg.format(', '.join(self.ALLOWED_TYPES))

        # we use the same radius and screen as for oxygen
        if not self.explicit:
            radius, screen = self._find_radius_and_screen(topol)

        # find all the unique types of spin labels
        types = set(self.params.values())

        # create the bond types
        bond_types = {}
        for t in types:
            bond_k, bond_r = self.bond_params[t]
            topol.bond_types.append(
                pmd.BondType(bond_k, bond_r, list=topol.bond_types))
            bt = topol.bond_types[-1]
            bond_types[t] = bt

        # create the angle types
        angle_types = {}
        for t in types:
            angle_k, angle_theta = self.angle_params[t]
            topol.angle_types.append(
                pmd.AngleType(angle_k, angle_theta, list=topol.angle_types))
            at = topol.angle_types[-1]
            angle_types[t] = at

        # create the torsion types
        tors_types = {}
        for t in types:
            tors_k, tors_per, tors_phase = self.tors_params[t]
            topol.dihedral_types.append(
                    pmd.DihedralType(tors_k,
                                     tors_per,
                                     tors_phase,
                                     list=topol.dihedral_types))
            tt = topol.dihedral_types[-1]
            tors_types[t] = tt

        for key in self.params:
            if self.params[key] not in self.ALLOWED_TYPES:
                raise ValueError(err_msg.format(self.params[key]))

            # create the particle
            atom = pmd.Atom(None, 8, 'OND', 'OND', 0.0, 16.00)
            if not self.explicit:
                atom.radii = radius
                atom.screen = screen

            # add to system
            topol.add_atom_to_residue(atom, topol.residues[key])

            # find the other atoms
            ca = topol.view[':{},@CA'.format(key + 1)].atoms[0]
            cb = topol.view[':{},@CB'.format(key + 1)].atoms[0]
            n = topol.view[':{},@N'.format(key + 1)].atoms[0]

            # add bond
            topol.bonds.append(
                pmd.Bond(atom, ca, bond_types[self.params[key]]))

            # add angle
            topol.angles.append(
                pmd.Angle(cb, ca, atom, angle_types[self.params[key]]))

            # add torsion
            topol.dihedrals.append(
                pmd.Dihedral(n, ca, cb, atom, type=tors_types[self.params[key]]))

            # set position
            ca_pos = np.array((ca.xx, ca.xy, ca.xz))
            n_pos = np.array((n.xx, n.xy, n.xz))
            cb_pos = np.array((cb.xx, cb.xy, cb.xz))

            direction = np.linalg.norm(ca_pos - n_pos)
            new_pos = cb_pos - self.bond_params[self.params[key]][1].value_in_unit(u.angstrom) * direction

            atom.xx = new_pos[0]
            atom.xy = new_pos[1]
            atom.xz = new_pos[2]
        topol.remake_parm()

        # setup the new non-bonded parameters
        for t in types:
            indices = [index+1 for index in self.params
                       if self.params[index] == t]
            selection_string = '(:{residue_mask})&(@{atom_name})'.format(
                residue_mask=','.join(str(i) for i in indices),
                atom_name=t)
            print topol.LJ_radius
            action = pmd.tools.addLJType(
                topol,
                selection_string,
                radius=self.lj_params[t][0].value_in_unit(u.angstrom),
                epsilon=self.lj_params[t][1].value_in_unit(u.kilocalorie_per_mole)
            )
            action.execute()
            print topol.LJ_radius

    def _find_radius_and_screen(self, topol):
        for atom in topol.atoms:
            if atom.atomic_number == 8:
                return atom.radii, atom.screen
