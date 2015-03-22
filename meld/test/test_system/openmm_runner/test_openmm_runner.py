import unittest
import mock
from meld.system import OpenMMRunner, RunOptions
from meld.system.openmm_runner.runner import _parm_top_from_string, _create_openmm_system, _create_integrator
from meld.system.openmm_runner.runner import _add_always_active_restraints, _add_selectively_active_restraints
from meld.system import protein, builder, ConstantTemperatureScaler
from meldplugin import MeldForce
from simtk.openmm.app import AmberPrmtopFile, OBC2, GBn, GBn2
from simtk.openmm.app import forcefield as ff
from simtk.openmm import LangevinIntegrator
from simtk.unit import kelvin, picosecond, femtosecond, mole, gram
from meld.system.restraints import SelectableRestraint, NonSelectableRestraint, DistanceRestraint
from meld.system.restraints import TorsionRestraint, LinearScaler, RestraintGroup, SelectivelyActiveCollection
from meld.system.restraints import ConstantRamp


class TestOpenMMRunner(unittest.TestCase):
    def setUp(self):
        p = protein.ProteinMoleculeFromSequence('NALA ALA CALA')
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p])
        self.system.temperature_scaler = ConstantTemperatureScaler(300.)

    def test_raises_when_system_has_no_temperature_scaler(self):
        self.system.temperature_scaler = None
        with self.assertRaises(RuntimeError):
            OpenMMRunner(self.system, RunOptions())


class TestPrmTopFromString(unittest.TestCase):
    def test_should_call_openmm(self):
        with mock.patch('meld.system.openmm_runner.runner.AmberPrmtopFile') as mock_parm:
            _parm_top_from_string('ABCD')

            self.assertEqual(mock_parm.call_count, 1)

class TestCreateOpenMMSystem(unittest.TestCase):
    def setUp(self):
        self.mock_parm = mock.Mock(spec=AmberPrmtopFile)

    def test_no_cutoff_should_set_correct_method(self):
        _create_openmm_system(self.mock_parm, cutoff=None, use_big_timestep=False, implicit_solvent='obc', remove_com=False)
        self.mock_parm.createSystem.assert_called_with(removeCMMotion=False, nonbondedMethod=ff.NoCutoff, nonbondedCutoff=999.,
                                                       constraints=ff.HBonds, implicitSolvent=OBC2, hydrogenMass=None)

    def test_cutoff_sets_correct_method(self):
        _create_openmm_system(self.mock_parm, cutoff=1.5, use_big_timestep=False, implicit_solvent='obc', remove_com=False)
        self.mock_parm.createSystem.assert_called_with(removeCMMotion=False, nonbondedMethod=ff.CutoffNonPeriodic, nonbondedCutoff=1.5,
                                                       constraints=ff.HBonds, implicitSolvent=OBC2, hydrogenMass=None)

    def test_big_timestep_sets_allbonds_and_hydrogen_masses(self):
        _create_openmm_system(self.mock_parm, cutoff=None, use_big_timestep=True, implicit_solvent='obc', remove_com=False)
        self.mock_parm.createSystem.assert_called_with(removeCMMotion=False, nonbondedMethod=ff.NoCutoff, nonbondedCutoff=999.,
                                                       constraints=ff.AllBonds, implicitSolvent=OBC2, hydrogenMass=3.0 * (gram / mole))

    def test_gbneck_sets_correct_solvent_model(self):
        _create_openmm_system(self.mock_parm, cutoff=None, use_big_timestep=False, implicit_solvent='gbNeck', remove_com=False)
        self.mock_parm.createSystem.assert_called_with(removeCMMotion=False, nonbondedMethod=ff.NoCutoff, nonbondedCutoff=999.,
                                                       constraints=ff.HBonds, implicitSolvent=GBn, hydrogenMass=None)

    def test_gbneck2_sets_correct_solvent_model(self):
        _create_openmm_system(self.mock_parm, cutoff=None, use_big_timestep=False, implicit_solvent='gbNeck2', remove_com=False)
        self.mock_parm.createSystem.assert_called_with(removeCMMotion=False, nonbondedMethod=ff.NoCutoff, nonbondedCutoff=999.,
                                                       constraints=ff.HBonds, implicitSolvent=GBn2, hydrogenMass=None)


class TestCreateIntegrator(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch('meld.system.openmm_runner.runner.LangevinIntegrator', spec=LangevinIntegrator)
        self.MockIntegrator = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_sets_correct_temperature(self):
        _create_integrator(temperature=300., use_big_timestep=False)
        self.MockIntegrator.assert_called_with(300. * kelvin, 1.0 / picosecond, 2 * femtosecond)

    def test_big_timestep_should_set_correct_timestep(self):
        _create_integrator(temperature=300., use_big_timestep=True)
        self.MockIntegrator.assert_called_with(300. * kelvin, 1.0 / picosecond, 3.5 * femtosecond)


class TestAddAlwaysActiveRestraints(unittest.TestCase):
    def test_returns_empty_list_when_called_with_empty_manager(self):
        force_dict = {}
        results = _add_always_active_restraints(mock.Mock(), [], alpha=1.0, timestep=1, force_dict=force_dict)
        self.assertEqual(results, [])

    def test_returns_list_of_selectable_restraints_when_called_with_list(self):
        force_dict = {}
        r1 = SelectableRestraint()
        r2 = SelectableRestraint()
        r3 = SelectableRestraint()

        results = _add_always_active_restraints(mock.Mock(), [r1, r2, r3], alpha=1.0, timestep=1, force_dict=force_dict)

        self.assertEqual(results, [r1, r2, r3])

    def test_raises_nonimplemented_if_nonselectable_restraints_are_included(self):
        force_dict = {}
        r1 = SelectableRestraint()
        r2 = SelectableRestraint()
        r3 = NonSelectableRestraint()

        with self.assertRaises(NotImplementedError):
            _add_always_active_restraints(mock.Mock(), [r1, r2, r3], alpha=1.0, timestep=1, force_dict=force_dict)


class TestAddSelectivelyActiveRestraints(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch('meld.system.openmm_runner.runner.MeldForce')
        self.MockMeldForce = self.patcher.start()
        self.mock_meld_force = mock.Mock(spec=MeldForce)
        self.MockMeldForce.return_value = self.mock_meld_force
        self.mock_system = mock.Mock()
        self.mock_system.index_of_atom.side_effect = range(1, 100)
        self.scaler = LinearScaler(0, 1)
        self.ramp = ConstantRamp()
        self.mock_openmm_system = mock.Mock()

    def tearDown(self):
        self.patcher.stop()

    def test_nothing_should_happen_with_empty_input(self):
        force_dict = {}
        _add_selectively_active_restraints(self.mock_openmm_system, [], [], 0.5, 1, force_dict)
        self.assertEqual(self.MockMeldForce.call_count, 0)

    def test_meld_force_should_be_created_with_non_empty_input(self):
        force_dict = {}
        dist_rest = DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 0, 0, 0.3, 999., 1.0)
        _add_selectively_active_restraints(self.mock_openmm_system, [], [dist_rest], alpha=1.0, timestep=1, force_dict=force_dict)
        self.assertEqual(self.MockMeldForce.call_count, 1)
        self.mock_openmm_system.addForce.assert_called_once_with(self.mock_meld_force)
        self.assertIn('meld', force_dict)

    def test_adding_distance_restraint_should_call_openmm_correctly(self):
        force_dict = {}
        self.mock_meld_force.addDistanceRestraint.return_value = 0
        self.mock_meld_force.addGroup.return_value = 0
        dist_rest = DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 0, 0, 0.3, 999., 10.0)
        _add_selectively_active_restraints(self.mock_openmm_system, [], [dist_rest], alpha=0.25, timestep=1, force_dict=force_dict)
        self.mock_meld_force.addDistanceRestraint.assert_called_once_with(0, 1, 0, 0, 0.3, 999., 7.5)
        self.mock_meld_force.addGroup.assert_called_once_with([0], 1)
        self.mock_meld_force.addCollection.assert_called_once_with([0], 1)

    def test_adding_torsion_restraint_should_call_openmm_correctly(self):
        force_dict = {}
        self.mock_meld_force.addTorsionRestraint.return_value = 0
        self.mock_meld_force.addGroup.return_value = 0
        tors_rest = TorsionRestraint(self.mock_system, self.scaler, self.ramp,
                                     1, 'CA', 2, 'CA',
                                     3, 'CA', 4, 'CA',
                                     0, 10, 10.0)
        _add_selectively_active_restraints(self.mock_openmm_system, [], [tors_rest], alpha=0.25, timestep=1, force_dict=force_dict)
        self.mock_meld_force.addTorsionRestraint.assert_called_once_with(0, 1, 2, 3, 0, 10, 7.5)
        self.mock_meld_force.addGroup.assert_called_once_with([0], 1)
        self.mock_meld_force.addCollection.assert_called_once_with([0], 1)

    def test_adding_collections_should_call_openmm_correctly(self):
        force_dict = {}
        self.mock_meld_force.addDistanceRestraint.side_effect = range(100)
        self.mock_meld_force.addGroup.side_effect = range(100)

        r1 = DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 0, 0, 0.3, 999., 10.0)
        r2 = DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 0, 0, 0.3, 999., 10.0)
        r3 = DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 0, 0, 0.3, 999., 10.0)
        r4 = DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 0, 0, 0.3, 999., 10.0)

        g1 = RestraintGroup([r1, r2], 1)
        g2 = RestraintGroup([r3], 1)
        g3 = RestraintGroup([r4], 1)

        c1 = SelectivelyActiveCollection([g1], 1)
        c2 = SelectivelyActiveCollection([g2, g3], 1)

        _add_selectively_active_restraints(self.mock_openmm_system, [c1, c2], [], alpha=0.5, timestep=1, force_dict=force_dict)

        self.assertEqual(self.mock_meld_force.addDistanceRestraint.call_count, 4)

        group_calls = [
            mock.call([0, 1], 1),
            mock.call([2], 1),
            mock.call([3], 1)]
        self.mock_meld_force.addGroup.assert_has_calls(group_calls)

        coll_calls = [
            mock.call([0], 1),
            mock.call([1, 2], 1)]
        self.mock_meld_force.addCollection.assert_has_calls(coll_calls)
