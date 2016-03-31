#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import util
from .system import System
import subprocess


class SystemBuilder(object):
    def __init__(self, forcefield='ff12sb', gb_radii='mbondi3'):
        self._forcefield = None
        self._set_forcefield(forcefield)
        self._gb_radii = None
        self._set_gb_radii(gb_radii)

    def build_system_from_molecules(self, molecules):
        with util.in_temp_dir():
            leap_cmds = []
            mol_ids = []
            leap_cmds.extend(self._generate_leap_header())
            for index, mol in enumerate(molecules):
                mol_id = 'mol_{}'.format(index)
                mol_ids.append(mol_id)
                mol.prepare_for_tleap(mol_id)
                leap_cmds.extend(mol.generate_tleap_input(mol_id))
            leap_cmds.extend(self._generate_leap_footer(mol_ids))
            with open('tleap.in', 'w') as tleap_file:
                tleap_string = '\n'.join(leap_cmds)
                tleap_file.write(tleap_string)
                print tleap_string
            subprocess.check_call('tleap -f tleap.in > tleap.out', shell=True)
            with open('system.top') as top_file:
                top = top_file.read()
            with open('system.mdcrd') as crd_file:
                crd = crd_file.read()
            return System(top, crd)

    def _set_forcefield(self, forcefield):
        ff_dict = {'ff12sb': 'leaprc.ff12SB','ff14sb': 'leaprc.ff14SB', 'ff14sbside':  'leaprc.ff14SBonlysc'}
        try:
            self._forcefield = ff_dict[forcefield]
        except KeyError:
            raise RuntimeError('Unknown forcefield: {}'.format(forcefield))

    def _set_gb_radii(self, gb_radii):
        allowed = ['mbondi2', 'mbondi3']
        if not gb_radii in allowed:
            raise RuntimeError('Unknown gb_radii: {}'.format(gb_radii))
        else:
            self._gb_radii = gb_radii

    def _generate_leap_header(self):
        leap_cmds = []
        leap_cmds.append('set default PBradii {}'.format(self._gb_radii))
        leap_cmds.append('source {}'.format(self._forcefield))
        return leap_cmds

    def _generate_leap_footer(self, mol_ids):
        leap_cmds = []
        list_of_mol_ids = ''
        for mol_id in mol_ids:
            list_of_mol_ids += '{} '.format(mol_id)
        leap_cmds.append('sys = combine {{ {} }}'.format(list_of_mol_ids))
        leap_cmds.append('check sys')
        leap_cmds.append('saveAmberParm sys system.top system.mdcrd')
        leap_cmds.append('quit')
        return leap_cmds
