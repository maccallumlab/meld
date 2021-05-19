"""
This module implements Patcher classes that can modify
the system to include new atoms and/or paramters.
"""

from meld import util
from meld import interfaces
from meld.system import indexing
import parmed as pmd  # type: ignore
from parmed import unit as u

import numpy as np  # type: ignore
from typing import Tuple, Dict, List


class PatcherBase:
    """
    Base class for Patcher objects
    """
    def patch(self, top_string: str, crd_string: str) -> Tuple[str, str]:
        """
        Called before `System` is created.

        Args:
            top_string: the output topology from tleap or the previous Patcher
            crd_string: the output mdcrd string from tleap or the previous Patcher

        Returns:
            Returns new topology and mdcrd strings that will be used
            by the next Patcher and eventually to create the System.

        .. note::
           This method is called with the topology and mdcrd strings before
           the system is created. It can add new atoms, bonds, etc, and
           modify an parameters present in the topology. The Patchers
           are chained together to produce a final topology and crd
           that are used to instantiate the System.
        """
        pass

    def finalize(self, system: interfaces.ISystem):
        """
        Called after `System` is created.

        Args:
            system: the system to be patched

        .. note::
           This method is called after the system has been created. It can
           modify the system object in any way, for example by adding restraints.
        """
        pass


class RdcAlignmentPatcher(PatcherBase):
    """
    Patch system to include extra dummy atoms to encode RDC alignment tensors.
    """

    def __init__(self, n_tensors: int):
        """
        Initialize an RdcAlignmentPatcher

        Args:
            n_tensors: number of alignment tensors to add
        """
        self.n_tensors = n_tensors
        self.resids: List[int] = []

    def patch(self, top_string: str, crd_string: str) -> Tuple[str, str]:
        INTOP = "in.top"
        INRST = "in.rst"
        OUTTOP = "out.top"
        OUTRST = "out.rst"

        with util.in_temp_dir():
            with open(INTOP, "wt") as outfile:
                outfile.write(top_string)
            with open(INRST, "wt") as outfile:
                outfile.write(crd_string)

            base = pmd.load_file(INTOP)
            crd = pmd.load_file(INRST)
            base.coordinates = crd.coordinates

            # create a new structure to add our dummy atoms to
            parm = pmd.Structure()

            # add in atom type for our dummy particles
            atype = pmd.AtomType("SDUM", 0, mass=12.0, charge=0.0)
            atype.set_lj_params(0.0, 0.0)

            for i in range(self.n_tensors):
                a1 = pmd.Atom(
                    name="S1",
                    atomic_number=atype.atomic_number,
                    type=str(atype),
                    charge=atype.charge,
                    mass=atype.mass,
                    solvent_radius=1.0,
                    screen=0.5,
                )
                a1.atom_type = atype

                a2 = pmd.Atom(
                    name="S2",
                    atomic_number=atype.atomic_number,
                    type=str(atype),
                    charge=atype.charge,
                    mass=atype.mass,
                    solvent_radius=1.0,
                    screen=0.5,
                )
                a2.atom_type = atype

                parm.add_atom(a1, resname="SDM", resnum=i)
                parm.add_atom(a2, resname="SDM", resnum=i)

            # we add noise here because we'll get NaN if the particles ever
            # end up exactly on top of each other
            parm.positions = np.zeros((2 * self.n_tensors, 3))

            # combine the old system with the new dummy atoms
            comb = base + parm
            last_index = comb.residues[-1].idx
            self.resids = list(range(last_index - self.n_tensors + 1, last_index + 1))

            comb.write_parm(OUTTOP)
            comb.write_rst7(OUTRST)
            with open(OUTTOP, "rt") as infile:
                top_string = infile.read()
            with open(OUTRST, "rt") as infile:
                crd_string = infile.read()
        return top_string, crd_string


class VirtualSpinLabelPatcher(PatcherBase):
    """
    Patch residues to include virtual spin label sites.
    """

    ALLOWED_TYPES = ["OND"]

    bond_params = {
        "OND": (1.2 * u.kilocalorie_per_mole / u.angstrom ** 2, 8.0 * u.angstrom)
    }
    angle_params = {
        "OND": (2.0 * u.kilocalorie_per_mole / u.radian ** 2, 46.0 * u.degrees)
    }
    tors_params = {"OND": (1.9 * u.kilocalorie_per_mole, 1, 240.0 * u.degrees)}
    lj_params = {"OND": (4.0 * u.angstrom / 2.0, 0.05 * u.kilocalorie_per_mole)}

    def __init__(self, params: Dict[indexing.ResidueIndex, str], explicit_solvent: bool = False):
        """
        Initialize a VirtualSpinLabelPatcher

        Args:
            params: A dictionary of resdiue indices and corresponding spin label types.
            explicit_solvent: A flag indicating if this is an explicit solvent simulation.

        .. note::
           Currently, 'OND' is the only supported spin label type, with parameters
           taken from:
           Islam, Stein, Mchaourab, and Roux, J. Phys. Chem. B 2013, 117, 4740-4754.
        """
        for key in params:
            assert isinstance(key, indexing.ResidueIndex)
            assert params[key] in self.ALLOWED_TYPES

        self.params = params
        self.explicit = explicit_solvent

    def patch(self, top_string: str, crd_string: str) -> Tuple[str, str]:
        INTOP = "in.top"
        INRST = "in.rst"
        OUTTOP = "out.top"
        OUTRST = "out.rst"

        with util.in_temp_dir():
            with open(INTOP, "wt") as outfile:
                outfile.write(top_string)
            with open(INRST, "wt") as outfile:
                outfile.write(crd_string)

            topol = pmd.load_file(INTOP)
            crd = pmd.load_file(INRST)
            topol.coordinates = crd.coordinates

            self._add_particles(topol)

            topol.write_parm(OUTTOP)
            topol.write_rst7(OUTRST)
            with open(OUTTOP, "rt") as infile:
                top_string = infile.read()
            with open(OUTRST, "rt") as infile:
                crd_string = infile.read()
        return top_string, crd_string

    def finalize(self, system: interfaces.ISystem):
        for res_index in self.params:
            site_type = self.params[res_index]

            # find the atoms
            n_index = system.atom_index(int(res_index), "N")
            ca_index = system.atom_index(int(res_index), "CA")
            cb_index = system.atom_index(int(res_index), "CB")
            ond_index = system.atom_index(int(res_index), "OND")

            # add the extra parameters
            system.add_extra_bond(
                i=ca_index,
                j=ond_index,
                length=self.bond_params[site_type][1],
                force_constant=self.bond_params[site_type][0],
            )
            system.add_extra_angle(
                i=cb_index,
                j=ca_index,
                k=ond_index,
                angle=self.angle_params[site_type][1],
                force_constant=self.angle_params[site_type][0],
            )
            system.add_extra_torsion(
                i=n_index,
                j=ca_index,
                k=cb_index,
                l=ond_index,
                phase=self.tors_params[site_type][1],
                multiplicity=1,
                energy=self.tors_params[site_type][0],
            )

    def _add_particles(self, topol):
        err_msg = "Unknown spin label type {{}}. Allowed values are: {}"
        err_msg = err_msg.format(", ".join(self.ALLOWED_TYPES))

        # we use the same radius and screen as for oxygen
        if not self.explicit:
            radius, screen = self._find_radius_and_screen(topol)

        # find all the unique types of spin labels
        types = set(self.params.values())

        for key in self.params:
            if self.params[key] not in self.ALLOWED_TYPES:
                raise ValueError(err_msg.format(self.params[key]))

            # create the particle
            atom = pmd.Atom(None, 8, "OND", "OND", 0.0, 16.00)
            if not self.explicit:
                atom.solvent_radius = radius
                atom.screen = screen

            # add to system
            topol.add_atom_to_residue(atom, topol.residues[int(key)])

            # find the other atoms
            ca = topol.view[f":{int(key)+1},@CA"].atoms[0]
            cb = topol.view[f":{int(key)+1},@CB"].atoms[0]
            n = topol.view[f":{int(key)+1},@N"].atoms[0]

            # Mark that the spin label and CA are connected.
            # This will not actually add a bond to the potential,
            # but will mark the connectivity between the atoms.
            atom.bond_to(ca)

            # set position
            ca_pos = np.array((ca.xx, ca.xy, ca.xz))
            n_pos = np.array((n.xx, n.xy, n.xz))
            cb_pos = np.array((cb.xx, cb.xy, cb.xz))

            direction = (ca_pos - cb_pos) / np.linalg.norm(ca_pos - cb_pos)
            new_pos = (
                ca_pos
                - self.bond_params[self.params[key]][1].value_in_unit(u.angstrom)
                * direction
                + ca_pos
                - n_pos
            )

            atom.xx = new_pos[0]
            atom.xy = new_pos[1]
            atom.xz = new_pos[2]
        topol.remake_parm()

        # setup the new non-bonded parameters
        for t in types:
            indices = [index for index in self.params if self.params[index] == t]
            selection_string = "(:{residue_mask})&(@{atom_name})".format(
                residue_mask=",".join(str(i) for i in indices), atom_name=t
            )
            action = pmd.tools.addLJType(
                topol,
                selection_string,
                radius=self.lj_params[t][0].value_in_unit(u.angstrom),
                epsilon=self.lj_params[t][1].value_in_unit(u.kilocalorie_per_mole),
            )
            action.execute()

    def _find_radius_and_screen(self, topol):
        for atom in topol.atoms:
            if atom.atomic_number == 8:
                return atom.solvent_radius, atom.screen
