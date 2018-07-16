#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

from meld import util
from meld.system.system import System
import subprocess


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


class SystemBuilder:
    def __init__(self, forcefield="ff14sbside", gb_radii="mbondi3",
        explicit_solvent=False, solvent_forcefield="tip3p", solvent_distance=6
    ):
        self._forcefield = None
        self._set_forcefield(forcefield)
        self._gb_radii = None
        self._set_gb_radii(gb_radii)
        self._explicit_solvent = explicit_solvent
        if self._explicit_solvent:
            self._set_solvent_forcefield(solvent_forcefield)
            self._solvent_dist = solvent_distance

    def build_system_from_molecules(
        self, molecules, patchers=None, leap_header_cmds=None
    ):
        if patchers is None:
            patchers = []
        if leap_header_cmds is None:
            leap_header_cmds = []
        if isinstance(leap_header_cmds, str):
            leap_header_cmds = [leap_header_cmds]

        with util.in_temp_dir():
            leap_cmds = []
            mol_ids = []
            leap_cmds.extend(self._generate_leap_header())
            leap_cmds.extend(leap_header_cmds)
            for index, mol in enumerate(molecules):
                mol_id = f"mol_{index}"
                mol_ids.append(mol_id)
                mol.prepare_for_tleap(mol_id)
                leap_cmds.extend(mol.generate_tleap_input(mol_id))
            if self._explicit_solvent:
                leap_cmds.extend(self._generate_solvent(mol_ids))
                leap_cmds.extend(self._generate_leap_footer([f"solute"]))
            else:
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
            raise RuntimeError(f"Unknown forcefield: {forcefield}")

    def _set_gb_radii(self, gb_radii):
        allowed = ["mbondi2", "mbondi3"]
        if gb_radii not in allowed:
            raise RuntimeError(f"Unknown gb_radii: {gb_radii}")
        else:
            self._gb_radii = gb_radii

    def _set_solvent_forcefield(self, solvent_forcefield):
        ff_dict = {
            "spce": "leaprc.water.spce",
            "spceb": "leaprc.water.spceb",
            "opc":  "leaprc.water.opc",
            "tip3p": "leaprc.water.tip3p",
            "tip4pew": "leaprc.water.tip4pew",
        }
        box_dict = {
            "spce": "SPCBOX",
            "spceb": "SPCBOX",
            "opc":  "OPCBOX",
            "tip3p": "TIP3PBOX",
            "tip4pew": "TIP4PEWBOX",
        }
        try:
            self._solvent_forcefield = ff_dict[solvent_forcefield]
            self._solvent_box = box_dict[solvent_forcefield]
        except KeyError:
            raise RuntimeError(f"Unknown solvent_model: {solvent_model}")

    def _generate_leap_header(self):
        leap_cmds = []
        leap_cmds.append(f"set default PBradii {self._gb_radii}")
        leap_cmds.append(f"source {self._forcefield}")
        if self._explicit_solvent:
            leap_cmds.append(f"source {self._solvent_forcefield}")
        return leap_cmds

    def _generate_solvent(self, mol_ids):
        leap_cmds = []
        list_of_mol_ids = ""
        for mol_id in mol_ids:
            list_of_mol_ids += f"{mol_id} "
        leap_cmds.append(f"solute = combine {{ {list_of_mol_ids} }}")
        leap_cmds.append(f"solvateBox solute {self._solvent_box} {self._solvent_dist}")
        return leap_cmds

    def _generate_leap_footer(self, mol_ids):
        leap_cmds = []
        list_of_mol_ids = ""
        for mol_id in mol_ids:
            list_of_mol_ids += f"{mol_id} "
        leap_cmds.append(f"sys = combine {{ {list_of_mol_ids} }}")
        leap_cmds.append("check sys")
        leap_cmds.append("saveAmberParm sys system.top system.mdcrd")
        leap_cmds.append("quit")
        return leap_cmds
