#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import util
from meld.system.system import System
import subprocess
from six import string_types


def load_amber_system(top_filename, crd_filename, patchers=None):
    if patchers is None:
        patchers = []

    with open(top_filename, "rt") as topfile:
        top = topfile.read()
    with open(crd_filename) as crdfile:
        crd = crdfile.read()

    for patcher in patchers:
        top, crd = patcher.patch(top, crd)

    system = System(top, crd)

    for patcher in patchers:
        patcher.finalize(system)

    return system


class SystemBuilder(object):
    def __init__(self, forcefield="ff14sbside", gb_radii="mbondi3"):
        self._forcefield = None
        self._set_forcefield(forcefield)
        self._gb_radii = None
        self._set_gb_radii(gb_radii)

    def build_system_from_molecules(
        self, molecules, patchers=None, leap_header_cmds=None
    ):
        if patchers is None:
            patchers = []
        if leap_header_cmds is None:
            leap_header_cmds = []
        if isinstance(leap_header_cmds, string_types):
            leap_header_cmds = [leap_header_cmds]

        with util.in_temp_dir():
            leap_cmds = []
            mol_ids = []
            leap_cmds.extend(self._generate_leap_header())
            leap_cmds.extend(leap_header_cmds)
            for index, mol in enumerate(molecules):
                mol_id = "mol_{}".format(index)
                mol_ids.append(mol_id)
                mol.prepare_for_tleap(mol_id)
                leap_cmds.extend(mol.generate_tleap_input(mol_id))
            leap_cmds.extend(self._generate_leap_footer(mol_ids))
            with open("tleap.in", "w") as tleap_file:
                tleap_string = "\n".join(leap_cmds)
                tleap_file.write(tleap_string)
            subprocess.check_call("tleap -f tleap.in > tleap.out", shell=True)

            return load_amber_system("system.top", "system.mdcrd", patchers)

    def _set_forcefield(self, forcefield):
        ff_dict = {
            "ff12sb": "leaprc.ff12SB",
            "ff14sb": "leaprc.protein.ff14SB",
            "ff14sbside": "leaprc.protein.ff14SBonlysc",
        }
        try:
            self._forcefield = ff_dict[forcefield]
        except KeyError:
            raise RuntimeError("Unknown forcefield: {}".format(forcefield))

    def _set_gb_radii(self, gb_radii):
        allowed = ["mbondi2", "mbondi3"]
        if gb_radii not in allowed:
            raise RuntimeError("Unknown gb_radii: {}".format(gb_radii))
        else:
            self._gb_radii = gb_radii

    def _generate_leap_header(self):
        leap_cmds = []
        leap_cmds.append("set default PBradii {}".format(self._gb_radii))
        leap_cmds.append("source {}".format(self._forcefield))
        return leap_cmds

    def _generate_leap_footer(self, mol_ids):
        leap_cmds = []
        list_of_mol_ids = ""
        for mol_id in mol_ids:
            list_of_mol_ids += "{} ".format(mol_id)
        leap_cmds.append("sys = combine {{ {} }}".format(list_of_mol_ids))
        leap_cmds.append("check sys")
        leap_cmds.append("saveAmberParm sys system.top system.mdcrd")
        leap_cmds.append("quit")
        return leap_cmds
