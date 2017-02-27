'''
This module implements `Patcher` classes that can modify
the system to include new atoms and/or paramters.
'''

import parmed as pmd
from meld import util
from parmed import unit as u
import numpy as np


class PatcherBase(object):
    def patch(self, top_string, crd_string):
        '''
        Called before `System` is created.

        Parameters
        ----------
        top_string : string
            The output topology from tleap or the previous Patcher
        crd_string : string
            The output mdcrd string from tleap or the previous Patcher

        Returns
        -------
        (new_top, new_crd) : (string, string)
            Returns new topology and mdcrd strings that will be used
            by the next Patcher and eventually to create the System.

        Notes
        -----
        This method is called with the topology and mdcrd strings before
        the system is created. It can add new atoms, bonds, etc, and
        modify an parameters present in the topology. The Patchers
        are chained together to produce a final topology and crd
        that are used to instantiate the System.
        '''
        pass

    def finalize(self, system):
        '''
        Called after `System` is created.

        Parameters
        ----------
        system : meld.system.System
            The system to be patched

        Notes
        -----
        This method is called after the system has been created. It can
        modify the system object in any way, for example by adding restraints.
        '''
        pass




class VirtualSpinLabelPatcher(PatcherBase):
    ALLOWED_TYPES = ['OND']

    bond_params = {'OND': (1.2 * u.kilocalorie_per_mole / u.angstrom**2, 8. * u.angstrom)}
    angle_params = {'OND': (2.0 * u.kilocalorie_per_mole / u.radian**2, 46. * u.degrees)}
    tors_params = {'OND': (1.9 * u.kilocalorie_per_mole, 1, 240. * u.degrees)}
    lj_params = {'OND': (4.0 * u.angstrom / 2.0, 0.05 * u.kilocalorie_per_mole)}

    def __init__(self, params, explicit_solvent=False):
        '''
        Patch residues to include virtual spin label sites.

        Parameters
        ----------
        params : {int: string}
            A dictionary with 1-based resdiue indices
            and corresponding spin label types.
        explicit_solvent : bool
            A flag indicating if this is an explicit
            solvent simulation.

        Notes
        -----
        Currently, 'OND' is the only supported spin label type, with parameters
        taken from:
        Islam, Stein, Mchaourab, and Roux, J. Phys. Chem. B 2013, 117, 4740-4754.
        '''
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

    def finalize(self, system):
        for res_index in self.params:
            site_type = self.params[res_index]

            # find the atoms
            n_index = system.index_of_atom(res_index, 'N')
            ca_index = system.index_of_atom(res_index, 'CA')
            cb_index = system.index_of_atom(res_index, 'CB')
            ond_index = system.index_of_atom(res_index, 'OND')

            # add the extra parameters
            system.add_extra_bond(i=ca_index, j=ond_index,
                                  length=self.bond_params[site_type][1],
                                  force_constant=self.bond_params[site_type][0])
            system.add_extra_angle(i=cb_index, j=ca_index, k=ond_index,
                                   angle=self.angle_params[site_type][1],
                                   force_constant=self.angle_params[site_type][0])
            system.add_extra_torsion(i=n_index, j=ca_index, k=cb_index, l=ond_index,
                                     phase=self.tors_params[site_type][1],
                                     multiplicity=1,
                                     energy=self.tors_params[site_type][0])

    def _add_particles(self, topol):
        err_msg = 'Unknown spin label type {{}}. Allowed values are: {}'
        err_msg = err_msg.format(', '.join(self.ALLOWED_TYPES))

        # we use the same radius and screen as for oxygen
        if not self.explicit:
            radius, screen = self._find_radius_and_screen(topol)

        # find all the unique types of spin labels
        types = set(self.params.values())

        for key in self.params:
            if self.params[key] not in self.ALLOWED_TYPES:
                raise ValueError(err_msg.format(self.params[key]))

            # create the particle
            atom = pmd.Atom(None, 8, 'OND', 'OND', 0.0, 16.00)
            if not self.explicit:
                atom.solvent_radius = radius
                atom.screen = screen

            # add to system
            topol.add_atom_to_residue(atom, topol.residues[key - 1])

            # find the other atoms
            ca = topol.view[':{},@CA'.format(key)].atoms[0]
            cb = topol.view[':{},@CB'.format(key)].atoms[0]
            n = topol.view[':{},@N'.format(key)].atoms[0]

            # set position
            ca_pos = np.array((ca.xx, ca.xy, ca.xz))
            n_pos = np.array((n.xx, n.xy, n.xz))
            cb_pos = np.array((cb.xx, cb.xy, cb.xz))

            direction = (ca_pos - cb_pos) / np.linalg.norm(ca_pos - cb_pos)
            new_pos = (ca_pos -
                       self.bond_params[self.params[key]][1].value_in_unit(u.angstrom) * direction +
                       ca_pos - n_pos)

            atom.xx = new_pos[0]
            atom.xy = new_pos[1]
            atom.xz = new_pos[2]
        topol.remake_parm()

        # setup the new non-bonded parameters
        for t in types:
            indices = [index for index in self.params
                       if self.params[index] == t]
            selection_string = '(:{residue_mask})&(@{atom_name})'.format(
                residue_mask=','.join(str(i) for i in indices),
                atom_name=t)
            action = pmd.tools.addLJType(
                topol,
                selection_string,
                radius=self.lj_params[t][0].value_in_unit(u.angstrom),
                epsilon=self.lj_params[t][1].value_in_unit(u.kilocalorie_per_mole)
            )
            action.execute()

    def _find_radius_and_screen(self, topol):
        for atom in topol.atoms:
            if atom.atomic_number == 8:
                return atom.solvent_radius, atom.screen
