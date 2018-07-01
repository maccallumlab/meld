#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

'''
This module implements a transformer that implements REST2.

We use the updated version of Replica Exchange with Solute Scaling [1]_.

Limitations
-----------

Currently, the REST2 implmentation in MELD has two limitations:
1. MELD forces do not work with periodic boundary conditions
2. Any CMAP / AMAP potentials are not scaled.


References
----------
.. [1] L. Wang, R.A. Friesner, B.J. Berne, Replica exchange with solute scaling: a
   more efficient version of replica exchange with solute tempering.
'''

from meld.system.openmm_runner.transform import TransformerBase
from simtk import openmm as mm
import math


class REST2Transformer(TransformerBase):
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        self.active = False
        self.protein_nonbonded_params = {}
        self.protein_exception_params = {}
        self.protein_dihedrals = {}
        self.scaler = None
        self.ramp = None
        self.nb_force = None
        self.dihedral_force = None

        if options.use_rest2:
            self.active = True
            self.scaler = options.rest2_scaler

            if options.solvation != 'explicit':
                raise ValueError('Cannot use REST2 without explicit solvent')

    def finalize(self, system, topology):
        if self.active:
            nonsolvent_atoms = self._find_nonsolvent_atoms(topology)
            self._find_nb_force(system)
            self._find_dihedral_force(system)
            self._gather_nonbonded_params(nonsolvent_atoms)
            self._gather_dihedral_params(nonsolvent_atoms, topology)

    def update(self, simulation, alpha, timestep):
        if self.active:
            scale = self.scaler(alpha)
            self._update_nonbonded(simulation, scale)
            self._update_dihedrals(simulation, scale)

    def _find_nonsolvent_atoms(self, topology):
        solvent_residue_names = ['WAT', 'SOL', 'H2O', 'HOH']
        nonsolvent_atoms = []
        for atom in topology.atoms():
            if not atom.residue.name in solvent_residue_names:
                nonsolvent_atoms.append(atom.index)
        return nonsolvent_atoms

    def _gather_nonbonded_params(self, nonsolvent_atoms):
        # gather the nonbonded parameters
        for index in nonsolvent_atoms:
            self.protein_nonbonded_params[index] = self.nb_force.getParticleParameters(index)

        # gather the exception parameters
        for param_index in range(self.nb_force.getNumExceptions()):
            params = self.nb_force.getExceptionParameters(param_index)
            if params[0] in nonsolvent_atoms and params[1] in nonsolvent_atoms:
                self.protein_exception_params[param_index] = params

    def _gather_dihedral_params(self, nonsolvent_atoms, topology):
        bond_idxs = [sorted([i.index, j.index]) for i,j in topology.bonds()]
        for parm_index in range(self.dihedral_force.getNumTorsions()):
            params = self.dihedral_force.getTorsionParameters(parm_index)
            i, j, k, l, mult, phi, fc = params

            not_solvent = (i in nonsolvent_atoms and
                           j in nonsolvent_atoms and
                           k in nonsolvent_atoms and
                           l in nonsolvent_atoms)

            not_improper = (sorted([i,j]) in bond_idxs and
                            sorted([j,k]) in bond_idxs and
                            sorted([k,l]) in bond_idxs)

            # only modify dihedrals involving non-solvent atoms
            # and those where sequential atoms are bonded (proper dihedrals)
            if not_solvent and not_improper:
                self.protein_dihedrals[parm_index] = params

    def _find_nb_force(self, system):
        forces = [system.getForce(i) for i in range(system.getNumForces())]
        nb_forces = [f for f in forces if isinstance(f, mm.NonbondedForce)]

        if not nb_forces:
            raise RuntimeError('REST2 could not find NonbondedForce')
        if len(nb_forces) > 1:
            raise RuntimeError('REST2 found more than one NonbondedForce')

        self.nb_force = nb_forces[0]

    def _find_dihedral_force(self, system):
        forces = [system.getForce(i) for i in range(system.getNumForces())]
        dihed_forces = [f for f in forces if isinstance(f, mm.PeriodicTorsionForce)]

        if not dihed_forces:
            raise RuntimeError('REST2 could not find PeriodicTorsionForce')
        if len(dihed_forces) > 1:
            raise RuntimeError('REST2 found more than one PeriodicTorsionForce')

        self.dihedral_force = dihed_forces[0]

    def _update_nonbonded(self, simulation, scale):
        for index in self.protein_nonbonded_params:
            params = self.protein_nonbonded_params[index]
            q, sigma, eps = params
            self.nb_force.setParticleParameters(index, q * math.sqrt(scale), sigma, eps * scale)

        for index in self.protein_exception_params:
            params = self.protein_exception_params[index]
            i, j, q, sigma, eps = params
            self.nb_force.setExceptionParameters(index, i, j, q * scale, sigma, eps * scale)
        self.nb_force.updateParametersInContext(simulation.context)

    def _update_dihedrals(self, simulation, scale):
        for index in self.protein_dihedrals:
            params = self.protein_dihedrals[index]
            i, j, k, l, mult, phi, fc = params
            self.dihedral_force.setTorsionParameters(index, i, j, k, l, mult, phi, fc * scale)
        self.dihedral_force.updateParametersInContext(simulation.context)
