#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
from unittest import mock  #type: ignore
from meld.system.openmm_runner import OpenMMRunner
from meld.system import protein, builder, ConstantTemperatureScaler, RunOptions
from meld.system.openmm_runner.runner import (
    _parm_top_from_string,
    _create_openmm_system,
    _create_integrator,
    PressureCouplingParams,
    PMEParams,
)
from simtk.openmm.app import AmberPrmtopFile, OBC2, GBn, GBn2  #type: ignore
from simtk.openmm.app import forcefield as ff
from simtk.openmm import LangevinIntegrator, MonteCarloBarostat  #type: ignore
from simtk.unit import kelvin, picosecond, femtosecond, mole, gram, atmosphere  #type: ignore
from meld.system.restraints import (
    SelectableRestraint,
    NonSelectableRestraint,
    DistanceRestraint,
    TorsionRestraint,
    LinearScaler,
    RestraintGroup,
    SelectivelyActiveCollection,
    ConstantRamp,
)


class TestOpenMMRunner(unittest.TestCase):
    def setUp(self):
        p = protein.ProteinMoleculeFromSequence("NALA ALA CALA")
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p])
        self.system.temperature_scaler = ConstantTemperatureScaler(300.)

    def test_raises_when_system_has_no_temperature_scaler(self):
        self.system.temperature_scaler = None
        with self.assertRaises(RuntimeError):
            OpenMMRunner(self.system, RunOptions())


class TestPrmTopFromString(unittest.TestCase):
    def test_should_call_openmm(self):
        with mock.patch(
            "meld.system.openmm_runner.runner.AmberPrmtopFile"
        ) as mock_parm:
            _parm_top_from_string("ABCD")

            self.assertEqual(mock_parm.call_count, 1)


class TestCreateOpenMMSystemImplicit(unittest.TestCase):
    def setUp(self):
        self.mock_parm = mock.Mock(spec=AmberPrmtopFile)
        self.TEMP = 300 * kelvin
        self.pcouple_params = PressureCouplingParams(
            temperature=300 * kelvin, pressure=1.0 * atmosphere, steps=25, enable=False
        )
        self.pme_params = PMEParams(enable=False, tolerance=0.005)

    def test_no_cutoff_should_set_correct_method(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=None,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="obc",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.NoCutoff,
            nonbondedCutoff=999.,
            constraints=ff.HBonds,
            implicitSolvent=OBC2,
            hydrogenMass=None,
            implicitSolventSaltConc=0.0,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

    def test_cutoff_sets_correct_method(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=1.5,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="obc",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.CutoffNonPeriodic,
            nonbondedCutoff=1.5,
            constraints=ff.HBonds,
            implicitSolvent=OBC2,
            hydrogenMass=None,
            implicitSolventSaltConc=0.0,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

    def test_salt_shielding_sets_correct_method(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=1.5,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="obc",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=0.150,
            implicitSolventSaltConcitSolventKappa=2.,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.CutoffNonPeriodic,
            nonbondedCutoff=1.5,
            constraints=ff.HBonds,
            implicitSolvent=OBC2,
            hydrogenMass=None,
            implicitSolventSaltConc=0.150,
            implicitSolventSaltConcitSolventKappa=2.,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

    def test_soulte_dielectric_sets_correct_method(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=1.5,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="obc",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=0.150,
            implicitSolventSaltConcitSolventKappa=2.,
            soluteDielectric=2.,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.CutoffNonPeriodic,
            nonbondedCutoff=1.5,
            constraints=ff.HBonds,
            implicitSolvent=OBC2,
            hydrogenMass=None,
            implicitSolventSaltConc=0.150,
            implicitSolventSaltConcitSolventKappa=2.,
            soluteDielectric=2.0,
            solventDielectric=78.5,
        )

    def test_solvent_dielectric_sets_correct_method(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=1.5,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="obc",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=0.150,
            implicitSolventSaltConcitSolventKappa=2.,
            soluteDielectric=None,
            solventDielectric=5.,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.CutoffNonPeriodic,
            nonbondedCutoff=1.5,
            constraints=ff.HBonds,
            implicitSolvent=OBC2,
            hydrogenMass=None,
            implicitSolventSaltConc=0.150,
            implicitSolventSaltConcitSolventKappa=2.,
            soluteDielectric=1.0,
            solventDielectric=5.,
        )

    def test_salt_concentration_sets_correct_method(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=1.5,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="obc",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=0.150,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.CutoffNonPeriodic,
            nonbondedCutoff=1.5,
            constraints=ff.HBonds,
            implicitSolvent=OBC2,
            hydrogenMass=None,
            implicitSolventSaltConc=0.150,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

    def test_big_timestep_sets_allbonds_and_hydrogen_masses(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=None,
            use_big_timestep=True,
            use_bigger_timestep=False,
            implicit_solvent="obc",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.NoCutoff,
            nonbondedCutoff=999.,
            constraints=ff.AllBonds,
            implicitSolvent=OBC2,
            hydrogenMass=3.0 * (gram / mole),
            implicitSolventSaltConc=0.0,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

    def test_gbneck_sets_correct_solvent_model(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=None,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="gbNeck",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.NoCutoff,
            nonbondedCutoff=999.,
            constraints=ff.HBonds,
            implicitSolvent=GBn,
            hydrogenMass=None,
            implicitSolventSaltConc=0.0,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )

    def test_gbneck2_sets_correct_solvent_model(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="implicit",
            cutoff=None,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="gbNeck2",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=False,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=False,
            nonbondedMethod=ff.NoCutoff,
            nonbondedCutoff=999.,
            constraints=ff.HBonds,
            implicitSolvent=GBn2,
            hydrogenMass=None,
            implicitSolventSaltConc=0.0,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=1.0,
            solventDielectric=78.5,
        )


class TestCreateOpenMMSystemExplicitNoPCouple(unittest.TestCase):
    def setUp(self):
        self.mock_parm = mock.Mock(spec=AmberPrmtopFile)
        self.TEMP = 450.
        self.pcouple_params = PressureCouplingParams(
            temperature=300 * kelvin, pressure=1.0 * atmosphere, steps=25, enable=False
        )
        self.pme_params = PMEParams(enable=True, tolerance=0.0001)

    def test_no_pme_uses_cutoffs(self):
        pme_params = PMEParams(enable=False, tolerance=0.0001)

        _create_openmm_system(
            self.mock_parm,
            solvation_type="explicit",
            cutoff=1.0,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="vacuum",
            pme_params=pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=True,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=True,
            nonbondedMethod=ff.CutoffPeriodic,
            nonbondedCutoff=1.0,
            constraints=ff.HBonds,
            implicitSolvent=None,
            rigidWater=True,
            hydrogenMass=None,
            ewaldErrorTolerance=pme_params.tolerance,
        )

    def test_enables_pme_and_cutoffs(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="explicit",
            cutoff=1.0,
            use_big_timestep=False,
            use_bigger_timestep=False,
            implicit_solvent="vacuum",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=True,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=True,
            nonbondedMethod=ff.PME,
            nonbondedCutoff=1.0,
            constraints=ff.HBonds,
            implicitSolvent=None,
            rigidWater=True,
            hydrogenMass=None,
            ewaldErrorTolerance=self.pme_params.tolerance,
        )

    def test_big_timestep_sets_allbonds_and_hydrogen_masses(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="explicit",
            cutoff=1.0,
            use_big_timestep=True,
            use_bigger_timestep=False,
            implicit_solvent="vacuum",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=True,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=True,
            nonbondedMethod=ff.PME,
            nonbondedCutoff=1.0,
            constraints=ff.AllBonds,
            implicitSolvent=None,
            rigidWater=True,
            hydrogenMass=3.0 * gram / mole,
            ewaldErrorTolerance=self.pme_params.tolerance,
        )

    def test_bigger_timestep_sets_allbonds_and_hydrogen_masses(self):
        _create_openmm_system(
            self.mock_parm,
            solvation_type="explicit",
            cutoff=1.0,
            use_big_timestep=False,
            use_bigger_timestep=True,
            implicit_solvent="vacuum",
            pme_params=self.pme_params,
            pcouple_params=self.pcouple_params,
            remove_com=True,
            temperature=self.TEMP,
            extra_bonds=[],
            extra_restricted_angles=[],
            extra_torsions=[],
            implicitSolventSaltConc=None,
            implicitSolventSaltConcitSolventKappa=None,
            soluteDielectric=None,
            solventDielectric=None,
        )
        self.mock_parm.createSystem.assert_called_with(
            removeCMMotion=True,
            nonbondedMethod=ff.PME,
            nonbondedCutoff=1.0,
            constraints=ff.AllBonds,
            implicitSolvent=None,
            rigidWater=True,
            hydrogenMass=4.0 * gram / mole,
            ewaldErrorTolerance=self.pme_params.tolerance,
        )


class TestCreateOpenMMSystemExplicitPCouple(unittest.TestCase):
    def setUp(self):
        self.mock_parm = mock.Mock(spec=AmberPrmtopFile)
        self.TEMP = 450.

    def test_pressure_coupling_should_add_barostat(self):
        PRESS = 2.0 * atmosphere
        STEPS = 50
        pcouple_params = PressureCouplingParams(
            enable=True, temperature=self.TEMP, pressure=PRESS, steps=STEPS
        )
        pme_params = PMEParams(enable=True, tolerance=0.0005)

        with mock.patch(
            "meld.system.openmm_runner.runner.MonteCarloBarostat",
            spec=MonteCarloBarostat,
        ) as mock_baro:
            mock_baro.return_value = mock.sentinel.baro_force

            _create_openmm_system(
                self.mock_parm,
                solvation_type="explicit",
                cutoff=1.0,
                use_big_timestep=False,
                use_bigger_timestep=False,
                implicit_solvent="vacuum",
                pme_params=pme_params,
                pcouple_params=pcouple_params,
                remove_com=True,
                temperature=self.TEMP,
                extra_bonds=[],
                extra_restricted_angles=[],
                extra_torsions=[],
                implicitSolventSaltConc=None,
                implicitSolventSaltConcitSolventKappa=None,
                soluteDielectric=None,
                solventDielectric=None,
            )

            mock_baro.assert_called_with(PRESS, self.TEMP, STEPS)
            mock_sys = self.mock_parm.createSystem.return_value
            mock_sys.addForce.assert_called_with(mock.sentinel.baro_force)


class TestCreateIntegrator(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch(
            "meld.system.openmm_runner.runner.LangevinIntegrator",
            spec=LangevinIntegrator,
        )
        self.MockIntegrator = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_sets_correct_temperature(self):
        _create_integrator(
            temperature=300., use_big_timestep=False, use_bigger_timestep=False
        )
        self.MockIntegrator.assert_called_with(
            300. * kelvin, 1.0 / picosecond, 2 * femtosecond
        )

    def test_big_timestep_should_set_correct_timestep(self):
        _create_integrator(
            temperature=300., use_big_timestep=True, use_bigger_timestep=False
        )
        self.MockIntegrator.assert_called_with(
            300. * kelvin, 1.0 / picosecond, 3.5 * femtosecond
        )
