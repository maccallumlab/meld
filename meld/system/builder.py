#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to build a System from SubSystems
"""

from meld import util
from .system import _load_amber_system, System
from .indexing import _ChainInfo
from .subsystem import _SubSystem
from .patchers import PatcherBase
from typing import List, Optional
import subprocess


class SystemBuilder:
    r"""
    Class to handle building a System from SubSystems.
    """

    def __init__(
        self,
        forcefield: str = "ff14sbside",
        gb_radii: str = "mbondi3",
        explicit_solvent: bool = False,
        solvent_forcefield: str = "tip3p",
        solvent_distance: float = 6,
        explicit_ions: bool = False,
        p_ion: str = "Na+",
        p_ioncount: int = 0,
        n_ion: str = "Cl-",
        n_ioncount: int = 0,
    ):
        """
        Initialize a SystemBuilder

        Args:
            forcefield: the forcefield to use [ff12sb, ff14sb, ff14sbside]
            gb_radii: the generalized Born radii [mbondi2, mbondi3]
            explicit_solvent: use explicit solvent?
            solvent_forcefield: explicit force field [spce, spceb, opc, tip3p, tip4pew]
            solvent_distance: distance between protein and edge of solvent box, Angstrom
            explicit_ions: include explicit ions?
            p_ion: name of positive ion [Na+, K+, Li+, Rb+, Cs+, Mg+]
            p_ioncount: number of positive ions
            n_ion: name of negative ion [Cl-, I-, Br-, F-]
            n_ioncount: number of negative ions
        """
        self._forcefield = None
        self._set_forcefield(forcefield)
        self._gb_radii = None
        self._set_gb_radii(gb_radii)

        self._explicit_solvent = explicit_solvent
        self._explicit_ions = explicit_ions
        if self._explicit_solvent:
            self._set_solvent_forcefield(solvent_forcefield)
            self._solvent_dist = solvent_distance

            if self._explicit_ions:
                self._set_positive_ion_type(p_ion)
                self._p_ioncount = p_ioncount
                self._set_negative_ion_type(n_ion)
                self._n_ioncount = n_ioncount

    def build_system(
        self,
        subsystems: List[_SubSystem],
        patchers: Optional[List[PatcherBase]] = None,
        leap_header_cmds: Optional[List[str]] = None,
    ) -> System:
        """
        Build the system from SubSystems

        Args:
            subsystems: component subsystems that make up the system
            patchers: an optional list of patchers to modify system
            leap_header_cmds: an optional list of leap commands

        Returns:
            the MELD system
        """
        if not subsystems:
            raise ValueError("len(subsystems) must be > 0")

        if patchers is None:
            patchers = []
        if leap_header_cmds is None:
            leap_header_cmds = []
        if isinstance(leap_header_cmds, str):
            leap_header_cmds = [leap_header_cmds]

        with util.in_temp_dir():
            mol_ids = []
            chains = []
            current_res_index = 0
            leap_cmds = []
            leap_cmds.extend(self._generate_leap_header())
            leap_cmds.extend(leap_header_cmds)
            for index, sub in enumerate(subsystems):
                # First we'll update the indexing for this subsystem
                for chain in sub._info.chains:
                    residues_with_offset = {
                        k: v + current_res_index for k, v in chain.residues.items()
                    }
                    chains.append(_ChainInfo(residues_with_offset))
                current_res_index += sub._info.n_residues

                # now add the leap commands for this subsystem
                mol_id = f"mol_{index}"
                mol_ids.append(mol_id)
                sub.prepare_for_tleap(mol_id)
                leap_cmds.extend(sub.generate_tleap_input(mol_id))

            if self._explicit_solvent:
                leap_cmds.extend(self._generate_solvent(mol_ids))
                leap_cmds.extend(self._generate_leap_footer([f"solute"]))
            else:
                leap_cmds.extend(self._generate_leap_footer(mol_ids))

            with open("tleap.in", "w") as tleap_file:
                tleap_string = "\n".join(leap_cmds)
                tleap_file.write(tleap_string)
            try:
                subprocess.check_call("tleap -f tleap.in > tleap.out", shell=True)
            except subprocess.CalledProcessError:
                print("Call to tleap failed.")
                print()
                print()
                print()
                print("=========")
                print("tleap.in")
                print("=========")
                print(open("tleap.in").read())
                print()
                print()
                print()
                print("=========")
                print("tleap.out")
                print("=========")
                print(open("tleap.out").read())
                print()
                print()
                print()
                print("========")
                print("leap.log")
                print("========")
                print(open("leap.log").read())
                print()
                print()
                print()
                raise

            return _load_amber_system("system.top", "system.mdcrd", chains, patchers)

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
            "opc": "leaprc.water.opc",
            "tip3p": "leaprc.water.tip3p",
            "tip4pew": "leaprc.water.tip4pew",
        }
        box_dict = {
            "spce": "SPCBOX",
            "spceb": "SPCBOX",
            "opc": "OPCBOX",
            "tip3p": "TIP3PBOX",
            "tip4pew": "TIP4PEWBOX",
        }
        try:
            self._solvent_forcefield = ff_dict[solvent_forcefield]
            self._solvent_box = box_dict[solvent_forcefield]
        except KeyError:
            raise RuntimeError(f"Unknown solvent_model: {solvent_forcefield}")

    def _set_positive_ion_type(self, ion_type):
        allowed = ["Na+", "K+", "Li+", "Rb+", "Cs+", "Mg+"]
        if ion_type not in allowed:
            raise RuntimeError(f"Unknown ion_type: {ion_type}")
        else:
            self._p_ion = ion_type

    def _set_negative_ion_type(self, ion_type):
        allowed = ["Cl-", "I-", "Br-", "F-"]
        if ion_type not in allowed:
            raise RuntimeError(f"Unknown ion_type: {ion_type}")
        else:
            self._n_ion = ion_type

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
        if self._explicit_ions:
            leap_cmds.append(f"addIons solute {self._p_ion} {self._p_ioncount}")
            leap_cmds.append(f"addIons solute {self._n_ion} {self._n_ioncount}")
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
