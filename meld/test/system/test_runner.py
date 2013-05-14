import unittest
import mock
from meld.system.runner import OpenMMRunner, OpenMMOptions, _parm_top_from_string, _create_openmm_system
from meld.system.runner import _create_integrator
from meld.system import protein, builder, ConstantTemperatureScaler
from traits.api import TraitError
from simtk.openmm.app import AmberPrmtopFile, OBC2, GBn, GBn2
from simtk.openmm.app import forcefield as ff
from simtk.openmm import LangevinIntegrator
from simtk.unit import kelvin, picosecond, femtosecond


class TestOpenMMRunner(unittest.TestCase):
    def setUp(self):
        p = protein.ProteinMoleculeFromSequence('NALA ALA CALA')
        b = builder.SystemBuilder()
        self.system = b.build_system_from_molecules([p])
        self.system.temperature_scaler = ConstantTemperatureScaler(300.)

    def test_raises_when_system_has_no_temperature_scaler(self):
        self.system.temperature_scaler = None
        with self.assertRaises(RuntimeError):
            OpenMMRunner(self.system)


class TestPrmTopFromString(unittest.TestCase):
    def test_should_call_openmm(self):
        with mock.patch('meld.system.runner.AmberPrmtopFile') as mock_parm:
            _parm_top_from_string('ABCD')

            mock_parm.assert_called_once_with(parm_string='ABCD')


class TestCreateOpenMMSystem(unittest.TestCase):
    def setUp(self):
        self.mock_parm = mock.Mock(spec=AmberPrmtopFile)

    def test_no_cutoff_should_set_correct_method(self):
        _create_openmm_system(self.mock_parm, cutoff=None, use_big_timestep=False, implicit_solvent='obc')
        self.mock_parm.createSystem.assert_called_with(nonbondedMethod=ff.NoCutoff, nonbondedCutoff=999.,
                                                       constraints=ff.HBonds, implicitSolvent=OBC2)

    def test_cutoff_sets_correct_method(self):
        _create_openmm_system(self.mock_parm, cutoff=1.5, use_big_timestep=False, implicit_solvent='obc')
        self.mock_parm.createSystem.assert_called_with(nonbondedMethod=ff.CutoffNonPeriodic, nonbondedCutoff=1.5,
                                                       constraints=ff.HBonds, implicitSolvent=OBC2)

    def test_big_timestep_sets_hangles(self):
        _create_openmm_system(self.mock_parm, cutoff=None, use_big_timestep=True, implicit_solvent='obc')
        self.mock_parm.createSystem.assert_called_with(nonbondedMethod=ff.NoCutoff, nonbondedCutoff=999.,
                                                       constraints=ff.HAngles, implicitSolvent=OBC2)

    def test_gbneck_sets_correct_solvent_model(self):
        _create_openmm_system(self.mock_parm, cutoff=None, use_big_timestep=False, implicit_solvent='gbNeck')
        self.mock_parm.createSystem.assert_called_with(nonbondedMethod=ff.NoCutoff, nonbondedCutoff=999.,
                                                       constraints=ff.HBonds, implicitSolvent=GBn)

    def test_gbneck2_sets_correct_solvent_model(self):
        _create_openmm_system(self.mock_parm, cutoff=None, use_big_timestep=False, implicit_solvent='gbNeck2')
        self.mock_parm.createSystem.assert_called_with(nonbondedMethod=ff.NoCutoff, nonbondedCutoff=999.,
                                                       constraints=ff.HBonds, implicitSolvent=GBn2)


class TestCreateIntegrator(unittest.TestCase):
    def setUp(self):
        self.patcher = mock.patch('meld.system.runner.LangevinIntegrator', spec=LangevinIntegrator)
        self.MockIntegrator = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_sets_correct_temperature(self):
        _create_integrator(temperature=300., use_big_timestep=False)
        self.MockIntegrator.assert_called_with(300. * kelvin, 1.0 / picosecond, 2 * femtosecond)

    def test_big_timestep_should_set_correct_timestep(self):
        _create_integrator(temperature=300., use_big_timestep=True)
        self.MockIntegrator.assert_called_with(300. * kelvin, 1.0 / picosecond, 3.5 * femtosecond)


class TestOptions(unittest.TestCase):
    def setUp(self):
        self.options = OpenMMOptions()

    def test_timesteps_initializes_to_5000(self):
        self.assertEqual(self.options.timesteps, 5000)

    def test_timesteps_below_1_raises(self):
        with self.assertRaises(TraitError):
            self.options.timesteps = 0

    def test_minimization_steps_initializes_to_1000(self):
        self.assertEqual(self.options.minimize_steps, 1000)

    def test_minimizesteps_below_1_raises(self):
        with self.assertRaises(TraitError):
            self.options.minimize_steps = 0

    def test_implicit_solvent_model_defaults_to_gbneck2(self):
        self.assertEqual(self.options.implicit_solvent_model, 'gbNeck2')

    def test_implicit_solvent_model_accepts_gbneck(self):
        self.options.implicit_solvent_model = 'gbNeck'
        self.assertEqual(self.options.implicit_solvent_model, 'gbNeck')

    def test_implicit_solvent_model_accepts_obc(self):
        self.options.implicit_solvent_model = 'obc'
        self.assertEqual(self.options.implicit_solvent_model, 'obc')

    def test_bad_implicit_model_raises(self):
        with self.assertRaises(TraitError):
            self.options.implicit_solvent_model = 'bad'

    def test_cutoff_defaults_to_none(self):
        self.assertIs(self.options.cutoff, None)

    def test_cutoff_accepts_positive_float(self):
        self.options.cutoff = 1.0
        self.assertEqual(self.options.cutoff, 1.0)

    def test_cutoff_should_raise_with_non_positive_value(self):
        with self.assertRaises(TraitError):
            self.options.cutoff = 0.

    def test_use_big_timestep_defaults_to_false(self):
        self.assertEqual(self.options.use_big_timestep, False)

    def test_use_amap_defaults_to_false(self):
        self.assertEqual(self.options.use_amap, False)
