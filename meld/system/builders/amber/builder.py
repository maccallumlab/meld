#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to build a System from AmberSubSystems
"""

from meld import util
from ..spec import SystemSpec
from ... import indexing
from . import subsystem
from . import amap

from typing import List, Optional
from dataclasses import dataclass
from functools import partial
import subprocess
import logging

logger = logging.getLogger(__name__)

from openmm import app  # type: ignore
from openmm.app import forcefield as ff  # type: ignore
import openmm as mm  # type: ignore
from openmm import unit as u  # type: ignore
import numpy as np  # type: ignore

try:
    from gamd.integrator_factory import *

    has_gamd = True
except:
    has_gamd = False


@partial(dataclass, frozen=True)
class AmberOptions:
    default_temperature: float = 300.0 * u.kelvin
    forcefield: str = "ff14sbside"
    solvation: str = "implicit"
    gb_radii: str = "mbondi3"
    implicit_solvent_model: str = "gbNeck2"
    solute_dielectric: Optional[float] = None
    solvent_dielectric: Optional[float] = None
    implicit_solvent_salt_conc: Optional[float] = None
    solvent_forcefield: str = "tip3p"
    solvent_distance: float = 0.6
    explicit_ions: bool = False
    p_ion: str = "Na+"
    p_ioncount: int = 0
    n_ion: str = "Cl-"
    n_ioncount: int = 0
    enable_pme: bool = False
    pme_tolerance: float = 0.0005
    enable_pressure_coupling: bool = False
    pressure: float = 1.01325 * u.bar
    pressure_coupling_update_steps: int = 25
    cutoff: Optional[float] = None
    remove_com: bool = True
    use_big_timestep: bool = False
    use_bigger_timestep: bool = False
    enable_amap: bool = False
    amap_alpha_bias: float = 1.0
    amap_beta_bias: float = 1.0
    enable_gamd: bool = False
    boost_type_str: str = "upper-total"
    conventional_md_prep: int = 5
    conventional_md: int = 50
    gamd_equilibration_prep: int = 15
    gamd_equilibration: int = 150
    total_simulation_length: int = 1000
    averaging_window_interval: int = 2500
    sigma0p: float = 6.0
    sigma0d: float = 6.0
    random_seed: int = 0
    friction_coefficient: float = 1.0

    def __post_init__(self):
        # Sanity checks for implicit and explicit solvent
        if self.solvation == "implicit":
            if self.enable_pme:
                raise ValueError("Using implicit solvation, but `enable_pme` is True.")
            if self.enable_pressure_coupling:
                raise ValueError(
                    "Using implicit solvation, but `enable_pressure_coupling` is True."
                )
        elif self.solvation == "explicit":
            if not self.enable_pme:
                raise ValueError("Using explicit solvation, but `enable_pme` is False.")
            if not self.enable_pressure_coupling:
                raise ValueError(
                    "Using explicit solvation, but `enable_pressure_coupling` is False."
                )
            if self.enable_amap:
                raise ValueError("Using explicit solvation, but `enable_amap` is True.")
            if self.cutoff is None:
                raise ValueError("Using explicit solvation, but `cutoff` is None.")
        else:
            raise ValueError(f"Unknown solvation model {self.solvation}")

        if self.forcefield not in ["ff12sb", "ff14sb", "ff14sbside"]:
            raise ValueError(f"Unknown forcefield {self.forcefield}")

        if self.gb_radii not in ["mbondi2", "mbondi3"]:
            raise ValueError(f"Unknown gb_radii {self.gb_radii}")

        if self.solvent_forcefield not in ["spce", "spceb", "opc", "tip3p", "tip4pew"]:
            raise ValueError(f"Unknown solvent_forcefield {self.solvent_forcefield}")

        if self.p_ion not in ["Na+", "K+", "Li+", "Rb+", "Cs+", "Mg+"]:
            raise ValueError(f"Unknown p_ion {self.p_ion}")

        if self.n_ion not in ["Cl-", "I-", "Br-", "F-"]:
            raise ValueError(f"Unknown n_ion {self.n_ion}")

        if isinstance(self.default_temperature, u.Quantity):
            object.__setattr__(
                self,
                "default_temperature",
                self.default_temperature.value_in_unit(u.kelvin),
            )
        if self.default_temperature < 0:
            raise ValueError(f"default_temperature must be >= 0")

        if isinstance(self.pressure, u.Quantity):
            object.__setattr__(self, "pressure", self.pressure.value_in_unit(u.bar))
        if self.pressure < 0:
            raise ValueError(f"pressure must be >= 0")


class AmberSystemBuilder:
    r"""
    Class to handle building a System from SubSystems.
    """

    options: AmberOptions

    def __init__(self, options: AmberOptions):
        """
        Initialize a SystemBuilder

        Args:
            options: Options for building the system
        """
        self.options = options
        self._set_forcefield()

        if self.options.solvation == "explicit":
            self._set_solvent_forcefield()
            self._solvent_dist = self.options.solvent_distance * 10.0  # nm to angstrom

    def build_system(
        self,
        subsystems: List[subsystem._AmberSubSystem],
        leap_header_cmds: Optional[List[str]] = None,
    ) -> SystemSpec:
        """
        Build the system from AmberSubSystems
        """
        if not subsystems:
            raise ValueError("len(subsystems) must be > 0")

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
                    chains.append(indexing._ChainInfo(residues_with_offset))
                current_res_index += sub._info.n_residues

                # now add the leap commands for this subsystem
                mol_id = f"mol_{index}"
                mol_ids.append(mol_id)
                sub.prepare_for_tleap(mol_id)
                leap_cmds.extend(sub.generate_tleap_input(mol_id))

            if self.options.solvation == "explicit":
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

            prmtop = app.AmberPrmtopFile("system.top")
            crd = app.AmberInpcrdFile("system.mdcrd")

        topology = prmtop.topology
        topology = _add_chains(topology, chains)

        system, barostat = _create_openmm_system(
            prmtop,
            self.options.solvation,
            self.options.cutoff,
            self.options.use_big_timestep,
            self.options.use_bigger_timestep,
            self.options.implicit_solvent_model,
            self.options.enable_pme,
            self.options.pme_tolerance,
            self.options.enable_pressure_coupling,
            self.options.pressure,
            self.options.pressure_coupling_update_steps,
            self.options.remove_com,
            self.options.default_temperature,
            self.options.implicit_solvent_salt_conc,
            self.options.solute_dielectric,
            self.options.solvent_dielectric,
        )

        if self.options.enable_amap:
            amap.add_amap(
                system,
                topology,
                self.options.amap_alpha_bias,
                self.options.amap_beta_bias,
            )

        # Create integrator based on GaMD options
        if self.options.enable_gamd:
            assert (
                has_gamd == True
            ), "Couldn't find library integrator_factory. Please, install GaMD for OpenMM"
            allowed_modes = [
                "upper-dual",
                "lower-dual",
                "upper-total",
                "lower-total",
                "lower-dihedral",
                "upper-dihedral",
            ]
            if self.options.boost_type_str in allowed_modes:
                integrator = _create_gamd_integrator(
                    self.options,
                    system,
                )
            else:
                raise Exception(
                    f"{self.options.boost_type_str} mode not supported. Check your boost_type_str option."
                )
        else:
            integrator = _create_integrator(
                self.options.default_temperature,
                self.options.use_big_timestep,
                self.options.use_bigger_timestep,
            )

        coords = crd.getPositions(asNumpy=True).value_in_unit(u.nanometer)
        try:
            vels = crd.getVelocities(asNumpy=True)
        except AttributeError:
            print("WARNING: No velocities found, setting to zero")
            vels = np.zeros_like(coords)
        try:
            box = crd.getBoxVectors(asNumpy=True)
            # We only support orthorhombic boxes
            box_a = box[0][0].value_in_unit(u.nanometer)
            assert box[0][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            assert box[0][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            box_b = box[1][1].value_in_unit(u.nanometer)
            assert box[1][0] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            assert box[1][2] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            box_c = box[2][2].value_in_unit(u.nanometer)
            assert box[2][0] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            assert box[2][1] == 0.0 * u.nanometer, "Only orthorhombic boxes supported"
            box = np.array([box_a, box_b, box_c])
        except AttributeError:
            box = None

        return SystemSpec(
            self.options.solvation,
            system,
            topology,
            integrator,
            barostat,
            coords,
            vels,
            box,
            {
                "solvation": self.options.solvation,
                "builder": "amber",
                "implicit_solvent_model": self.options.implicit_solvent_model,
            },
        )

    def _set_forcefield(self):
        ff_dict = {
            "ff12sb": "leaprc.ff12SB",
            "ff14sb": "leaprc.protein.ff14SB",
            "ff14sbside": "leaprc.protein.ff14SBonlysc",
        }
        self._forcefield = ff_dict[self.options.forcefield]

    def _set_solvent_forcefield(self):
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

        self._solvent_forcefield = ff_dict[self.options.solvent_forcefield]
        self._solvent_box = box_dict[self.options.solvent_forcefield]

    def _generate_leap_header(self):
        leap_cmds = []
        leap_cmds.append(f"set default PBradii {self.options.gb_radii}")
        leap_cmds.append(f"source {self._forcefield}")
        if self.options.solvation == "explicit":
            leap_cmds.append(f"source {self._solvent_forcefield}")
        return leap_cmds

    def _generate_solvent(self, mol_ids):
        leap_cmds = []
        list_of_mol_ids = ""
        for mol_id in mol_ids:
            list_of_mol_ids += f"{mol_id} "
        leap_cmds.append(f"solute = combine {{ {list_of_mol_ids} }}")
        leap_cmds.append(f"solvateBox solute {self._solvent_box} {self._solvent_dist}")
        if self.options.explicit_ions:
            leap_cmds.append(
                f"addIons solute {self.options.p_ion} {self.options.p_ioncount}"
            )
            leap_cmds.append(
                f"addIons solute {self.options.n_ion} {self.options.n_ioncount}"
            )
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


def _create_openmm_system(
    parm_object,
    solvation_type,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
    implicit_solvent,
    enable_pme,
    pme_tolerance,
    enable_pressure_coupling,
    pressure,
    pressure_coupling_update_steps,
    remove_com,
    default_temperature,
    implicitSolventSaltConc,
    soluteDielectric,
    solventDielectric,
):
    if solvation_type == "implicit":
        logger.info("Creating implicit solvent system")
        system = _create_openmm_system_implicit(
            parm_object,
            cutoff,
            use_big_timestep,
            use_bigger_timestep,
            implicit_solvent,
            remove_com,
            implicitSolventSaltConc,
            soluteDielectric,
            solventDielectric,
        )
        baro = None
    elif solvation_type == "explicit":
        logger.info("Creating explicit solvent system")
        system, baro = _create_openmm_system_explicit(
            parm_object,
            cutoff,
            use_big_timestep,
            use_bigger_timestep,
            enable_pme,
            pme_tolerance,
            enable_pressure_coupling,
            pressure,
            pressure_coupling_update_steps,
            remove_com,
            default_temperature,
        )
    else:
        raise ValueError(f"unknown value for solvation_type: {solvation_type}")

    return system, baro


def _get_hydrogen_mass_and_constraints(use_big_timestep, use_bigger_timestep):
    if use_big_timestep:
        logger.info("Enabling hydrogen mass=3, constraining all bonds")
        constraint_type = ff.AllBonds
        hydrogen_mass = 3.0 * u.gram / u.mole
    elif use_bigger_timestep:
        logger.info("Enabling hydrogen mass=4, constraining all bonds")
        constraint_type = ff.AllBonds
        hydrogen_mass = 4.0 * u.gram / u.mole
    else:
        logger.info("Enabling hydrogen mass=1, constraining bonds with hydrogen")
        constraint_type = ff.HBonds
        hydrogen_mass = None
    return hydrogen_mass, constraint_type


def _create_openmm_system_implicit(
    parm_object,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
    implicit_solvent,
    remove_com,
    implicitSolventSaltConc,
    soluteDielectric,
    solventDielectric,
):
    if cutoff is None:
        logger.info("Using no cutoff")
        cutoff_type = ff.NoCutoff
        cutoff_dist = 999.0
    else:
        logger.info(f"Using a cutoff of {cutoff}")
        cutoff_type = ff.CutoffNonPeriodic
        cutoff_dist = cutoff

    hydrogen_mass, constraint_type = _get_hydrogen_mass_and_constraints(
        use_big_timestep, use_bigger_timestep
    )

    if implicit_solvent == "obc":
        logger.info('Using "OBC" implicit solvent')
        implicit_type = app.OBC2
    elif implicit_solvent == "gbNeck":
        logger.info('Using "gbNeck" implicit solvent')
        implicit_type = app.GBn
    elif implicit_solvent == "gbNeck2":
        logger.info('Using "gbNeck2" implicit solvent')
        implicit_type = app.GBn2
    elif implicit_solvent == "vacuum" or implicit_solvent is None:
        logger.info("Using vacuum instead of implicit solvent")
        implicit_type = None
    else:
        RuntimeError("Should never get here")

    if implicitSolventSaltConc is None:
        implicitSolventSaltConc = 0.0
    if soluteDielectric is None:
        soluteDielectric = 1.0
    if solventDielectric is None:
        solventDielectric = 78.5

    sys = parm_object.createSystem(
        nonbondedMethod=cutoff_type,
        nonbondedCutoff=cutoff_dist,
        constraints=constraint_type,
        implicitSolvent=implicit_type,
        removeCMMotion=remove_com,
        hydrogenMass=hydrogen_mass,
        implicitSolventSaltConc=implicitSolventSaltConc,
        soluteDielectric=soluteDielectric,
        solventDielectric=solventDielectric,
    )
    return sys


def _create_openmm_system_explicit(
    parm_object,
    cutoff,
    use_big_timestep,
    use_bigger_timestep,
    enable_pme,
    pme_tolerance,
    enable_pressure_coupling,
    pressure,
    pressure_couping_update_steps,
    remove_com,
    default_temperature,
):
    if cutoff is None:
        raise ValueError("cutoff must be set for explicit solvent, but got None")
    else:
        if enable_pme:
            logger.info(f"Using PME with tolerance {pme_tolerance}")
            cutoff_type = ff.PME
        else:
            logger.info("Using reaction field")
            cutoff_type = ff.CutoffPeriodic

        logger.info(f"Using a cutoff of {cutoff}")
        cutoff_dist = cutoff

    hydrogen_mass, constraint_type = _get_hydrogen_mass_and_constraints(
        use_big_timestep, use_bigger_timestep
    )

    s = parm_object.createSystem(
        nonbondedMethod=cutoff_type,
        nonbondedCutoff=cutoff_dist,
        constraints=constraint_type,
        implicitSolvent=None,
        removeCMMotion=remove_com,
        hydrogenMass=hydrogen_mass,
        rigidWater=True,
        ewaldErrorTolerance=pme_tolerance,
    )

    baro = None
    if enable_pressure_coupling:
        logger.info("Enabling pressure coupling")
        logger.info(f"Pressure is {pressure}")
        logger.info(
            f"Volume moves attempted every {pressure_couping_update_steps} steps"
        )
        baro = mm.MonteCarloBarostat(
            pressure, default_temperature, pressure_couping_update_steps
        )
        s.addForce(baro)

    return s, baro


def _create_integrator(temperature, use_big_timestep, use_bigger_timestep):
    if use_big_timestep:
        logger.info("Creating integrator with 3.5 fs timestep")
        timestep = 3.5 * u.femtosecond
    elif use_bigger_timestep:
        logger.info("Creating integrator with 4.5 fs timestep")
        timestep = 4.5 * u.femtosecond
    else:
        logger.info("Creating integrator with 2.0 fs timestep")
        timestep = 2.0 * u.femtosecond
    return mm.LangevinIntegrator(temperature * u.kelvin, 1.0 / u.picosecond, timestep)


def _create_gamd_integrator(options, system):
    gamdIntegratorFactory = GamdIntegratorFactory()
    if options.use_big_timestep:
        logger.info("Creating custom integrator with 3.5 fs timestep")
        timestep = 3.5 * u.femtosecond
    elif options.use_bigger_timestep:
        logger.info("Creating custom integrator with 4.5 fs timestep")
        timestep = 4.5 * u.femtosecond
    else:
        logger.info("Creating custom integrator with 2.0 fs timestep")
        timestep = 2.0 * u.femtosecond
    result = gamdIntegratorFactory.get_integrator(
        options.boost_type_str,
        system,
        options.default_temperature,
        timestep,
        options.conventional_md_prep,
        options.conventional_md,
        options.gamd_equilibration_prep,
        options.gamd_equilibration,
        options.total_simulation_length,
        options.averaging_window_interval,
        options.sigma0p,
        options.sigma0d,
    )
    [
        first_boost_group,
        second_boost_group,
        integrator,
        first_boost_type,
        second_boost_type,
    ] = result
    integrator.first_boost_group = first_boost_group
    integrator.second_boost_group = second_boost_group
    integrator.first_boost_type = first_boost_type
    integrator.second_boost_type = second_boost_type
    integrator.setRandomNumberSeed(options.random_seed)
    integrator.setFriction(options.friction_coefficient)
    return integrator


def _add_chains(topology, chain_list):
    # Verify that the input from Amber only has one chain
    assert len(list(topology.chains())) == 1

    newtop = app.Topology()

    # Add the chains to the new topology and
    # create a map between residues and chains.
    chain_map = {}
    for chain_info in chain_list:
        chain = newtop.addChain()
        for res in chain_info.residues.values():
            chain_map[res] = chain

    # Now we'll create a final chain for solvent, etc
    # and everything left to this last chain.
    last_chain = newtop.addChain()
    for res in topology.residues():
        if res.index not in chain_map:
            chain_map[res.index] = last_chain

    # Add all of the residues to the new topology, while
    # correcting the chain index.
    # Create a map between the old and new residues so
    # that we can add the atoms.
    residue_map = {}
    for residue in topology.residues():
        chain = chain_map[residue.index]
        new_residue = newtop.addResidue(residue.name, chain, residue.index)
        residue_map[residue] = new_residue

    # Now add back all of the atoms with tne new residues.
    # We keep a map between the old and new atoms so that
    # we can add the bonds.
    atom_map = {}
    for atom in topology.atoms():
        new_atom = newtop.addAtom(
            atom.name, atom.element, residue_map[atom.residue], atom.index
        )
        atom_map[atom] = new_atom

    # Now we add all of the bonds
    for bond in topology.bonds():
        newtop.addBond(atom_map[bond[0]], atom_map[bond[1]])

    return newtop
