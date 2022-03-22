import unittest
from unittest import mock  # type: ignore
from meld import AmberSubSystemFromSequence, AmberSystemBuilder, AmberOptions
from meld.system import scalers
from meld.system import restraints
from openmm import unit as u  # type: ignore


class TestConstantScaler(unittest.TestCase):
    def test_should_return_1_when_alpha_is_0(self):
        scaler = scalers.ConstantScaler()
        self.assertAlmostEqual(scaler(0.0), 1.0)

    def test_should_return_1_when_alpha_is_1(self):
        scaler = scalers.ConstantScaler()
        self.assertAlmostEqual(scaler(1.0), 1.0)

    def test_should_raise_if_alpha_is_less_than_zero(self):
        scaler = scalers.ConstantScaler()
        with self.assertRaises(RuntimeError):
            scaler(-1.0)

    def test_should_raise_if_alpha_is_greater_than_one(self):
        scaler = scalers.ConstantScaler()
        with self.assertRaises(RuntimeError):
            scaler(2.0)


class TestLinearScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.LinearScaler(-1, 1)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.LinearScaler(2, 1)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.LinearScaler(1, -1)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.LinearScaler(1, 2)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            scalers.LinearScaler(0.7, 0.6)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = scalers.LinearScaler(0.2, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = scalers.LinearScaler(0.2, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_1_below_alpha_min(self):
        scaler = scalers.LinearScaler(0.2, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.1), 1.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = scalers.LinearScaler(0.2, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_correct_value_in_middle(self):
        scaler = scalers.LinearScaler(0.0, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.3), 0.7)


class TestPlateauLinearScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauLinearScaler(-1, 0, 0.5, 1)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauLinearScaler(2, 0, 0.5, 1)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauLinearScaler(1, 0, 0.5, -1)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauLinearScaler(1, 0, 0.5, 2)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauLinearScaler(0.7, 0, 0.5, 0.6)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = scalers.PlateauLinearScaler(0.2, 0.5, 0.7, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = scalers.PlateauLinearScaler(0.2, 0.6, 0.7, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_0_below_alpha_min(self):
        scaler = scalers.PlateauLinearScaler(0.2, 0.4, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.1), 0.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = scalers.PlateauLinearScaler(0.2, 0.4, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_correct_value_between_alpha_one_alpha_two_down(self):
        scaler = scalers.PlateauLinearScaler(0.0, 0.4, 0.6, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.3), 0.75)

    def test_should_return_correct_value_between_alpha_one_alpha_two_down2(self):
        scaler = scalers.PlateauLinearScaler(0.0, 0.4, 0.6, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.1), 0.25)

    def test_should_return_correct_value_between_alpha_two_alpha_max_up(self):
        scaler = scalers.PlateauLinearScaler(0.0, 0.4, 0.6, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.7), 0.75)

    def test_should_return_correct_value_between_alpha_two_alpha_max_up2(self):
        scaler = scalers.PlateauLinearScaler(0.0, 0.4, 0.6, 1.0, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.9), 0.25)


class TestNonLinearScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.NonLinearScaler(-1, 1, 4)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.NonLinearScaler(2, 1, 4)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.NonLinearScaler(1, -1, 4)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.NonLinearScaler(1, 2, 4)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            scalers.NonLinearScaler(0.7, 0.6, 4)

    def test_should_raise_if_factor_below_one(self):
        with self.assertRaises(RuntimeError):
            scalers.NonLinearScaler(0.0, 1.0, 0.2)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = scalers.NonLinearScaler(0.2, 0.8, 4)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = scalers.NonLinearScaler(0.2, 0.8, 4)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_1_below_alpha_min(self):
        scaler = scalers.NonLinearScaler(0.2, 0.8, 4, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.1), 1.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = scalers.NonLinearScaler(0.2, 0.8, 4, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_midpoint_should_return_correct_value(self):
        scaler = scalers.NonLinearScaler(0.2, 0.8, 4, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.5), 0.119202922)


class TestPlateauNonLinearScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauNonLinearScaler(-1, 0.5, 0.6, 1, 4)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauNonLinearScaler(2, 0.5, 0.6, 1, 4)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauNonLinearScaler(1, 0.5, 0.6, -1, 4)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauNonLinearScaler(1, 0.5, 0.6, 2, 4)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauNonLinearScaler(0.7, 0.65, 0.63, 0.6, 4)

    def test_should_raise_if_factor_below_one(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauNonLinearScaler(0.0, 0.5, 0.7, 1.0, 0.2)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = scalers.PlateauNonLinearScaler(0.2, 0.4, 0.6, 0.8, 4)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = scalers.PlateauNonLinearScaler(0.2, 0.4, 0.6, 0.8, 4)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_0_below_alpha_min(self):
        scaler = scalers.PlateauNonLinearScaler(0.2, 0.4, 0.6, 0.8, 4, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.1), 0.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = scalers.PlateauNonLinearScaler(0.2, 0.4, 0.6, 0.8, 4, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_1_between_alpha_one_alpha_two(self):
        scaler = scalers.PlateauNonLinearScaler(0.2, 0.4, 0.6, 0.8, 4, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.5), 1.0)

    def test_midpoint_should_return_correct_value_scaling_up(self):
        scaler = scalers.PlateauNonLinearScaler(0.7, 0.8, 0.9, 1.0, 4, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.95), 0.119202922)

    def test_midpoint_should_return_correct_value_scaling_down(self):
        scaler = scalers.PlateauNonLinearScaler(0.7, 0.8, 0.9, 1.0, 4, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.75), 0.88079708)


class TestPlateauSmoothScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauSmoothScaler(-1, 0.5, 0.6, 1)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauSmoothScaler(2, 0.5, 0.6, 1)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauSmoothScaler(1, 0.5, 0.6, -1)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauSmoothScaler(1, 0.5, 0.6, 2)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            scalers.PlateauSmoothScaler(0.7, 0.5, 0.6, 0.6)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.5, 0.6, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.5, 0.6, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_0_below_alpha_min(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.5, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.1), 0.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.5, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_1_between_alpha_one_alpha_two(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.4, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.5), 1.0)

    def test_should_return_correct_value_middle_up(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.4, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.75), 0.15625)

    def test_should_return_correct_value_middle_up2(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.4, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.65), 0.84375)

    def test_should_return_correct_value_middle_down(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.4, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.35), 0.84375)

    def test_should_return_correct_value_middle_down2(self):
        scaler = scalers.PlateauSmoothScaler(0.2, 0.4, 0.6, 0.8, 1.0, 0.0)
        self.assertAlmostEqual(scaler(0.25), 0.15625)


class TestCreatescalersAndScalers(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("GLY GLY GLY GLY")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p])
        self.manager = restraints.RestraintManager(self.system)

    def test_can_create_constant_scaler(self):
        scaler = self.manager.create_scaler("constant")
        self.assertTrue(isinstance(scaler, scalers.ConstantScaler))

    def test_can_create_linear_scaler(self):
        scaler = self.manager.create_scaler("linear", alpha_min=0.2, alpha_max=0.8)
        self.assertTrue(isinstance(scaler, scalers.LinearScaler))

    def test_can_create_non_linear_scaler(self):
        scaler = self.manager.create_scaler(
            "nonlinear", alpha_min=0.2, alpha_max=0.8, factor=4
        )
        self.assertTrue(isinstance(scaler, scalers.NonLinearScaler))

    def test_creating_restraint_without_specifying_scaler_uses_constant(self):
        rest = self.manager.create_restraint(
            "distance",
            atom1=self.system.index.atom(0, "CA"),
            atom2=self.system.index.atom(1, "CA"),
            r1=0 * u.nanometer,
            r2=1 * u.nanometer,
            r3=3 * u.nanometer,
            r4=4 * u.nanometer,
            k=1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.assertTrue(isinstance(rest.scaler, scalers.ConstantScaler))

    def test_creating_restraint_with_scaler_should_use_it(self):
        scaler = scalers.LinearScaler(0, 1)
        rest = self.manager.create_restraint(
            "distance",
            scaler,
            atom1=self.system.index.atom(0, "CA"),
            atom2=self.system.index.atom(1, "CA"),
            r1=0 * u.nanometer,
            r2=1 * u.nanometer,
            r3=3 * u.nanometer,
            r4=4 * u.nanometer,
            k=1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.assertTrue(isinstance(rest.scaler, scalers.LinearScaler))

    def test_creating_restraint_should_raise_if_scaler_is_wrong_type(self):
        scaler = scalers.TimeRamp()
        with self.assertRaises(ValueError):
            _rest = self.manager.create_restraint(
                "distance",
                scaler,
                atom1=self.system.index.atom(0, "CA"),
                atom2=self.system.index.atom(1, "CA"),
                r1=0.0,
                r2=1.0,
                r3=3.0,
                r4=4.0,
                k=1.0,
            )

    def test_creating_restraint_should_raise_if_ramp_is_wrong_type(self):
        scaler = scalers.ConstantScaler()
        ramp = scalers.ConstantScaler()
        with self.assertRaises(ValueError):
            _rest = self.manager.create_restraint(
                "distance",
                scaler,
                ramp=ramp,
                atom1=self.system.index.atom(0, "CA"),
                atom2=self.system.index.atom(1, "CA"),
                r1=0.0,
                r2=1.0,
                r3=3.0,
                r4=4.0,
                k=1.0,
            )

    def test_create_restraint_without_specifying_ramp_should_use_constant_ramp(self):
        scaler = scalers.ConstantScaler()
        rest = self.manager.create_restraint(
            "distance",
            scaler,
            atom1=self.system.index.atom(0, "CA"),
            atom2=self.system.index.atom(1, "CA"),
            r1=0.0 * u.nanometer,
            r2=1.0 * u.nanometer,
            r3=3.0 * u.nanometer,
            r4=4.0 * u.nanometer,
            k=1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.assertTrue(isinstance(rest.ramp, scalers.ConstantRamp))


class TestConstantRamp(unittest.TestCase):
    def setUp(self):
        self.ramp = scalers.ConstantRamp()

    def test_should_raise_with_negative_time(self):
        with self.assertRaises(ValueError):
            self.ramp(-1)

    def test_should_always_return_one(self):
        self.assertEqual(self.ramp(0), 1.0)
        self.assertEqual(self.ramp(1000), 1.0)
        self.assertEqual(self.ramp(1000000000), 1.0)


class TestLinearRamp(unittest.TestCase):
    def setUp(self):
        self.ramp = scalers.LinearRamp(100, 200, 0.1, 0.9)

    def test_should_raise_with_negative_time(self):
        with self.assertRaises(ValueError):
            self.ramp(-1)

    def test_should_return_start_weight_before_start_time(self):
        self.assertEqual(self.ramp(0), 0.1)

    def test_return_end_weight_after_end_time(self):
        self.assertEqual(self.ramp(500), 0.9)

    def test_should_return_midpoint_half_way_between_start_and_end(self):
        self.assertAlmostEqual(self.ramp(150), 0.5)


class TestNonLinearRampUpWard(unittest.TestCase):
    def setUp(self):
        self.ramp = scalers.NonLinearRamp(100, 200, 0.1, 0.9, 4)

    def test_should_raise_with_negative_time(self):
        with self.assertRaises(ValueError):
            self.ramp(-1)

    def test_should_return_start_weight_before_start_time(self):
        self.assertEqual(self.ramp(0), 0.1)

    def test_should_return_end_weight_after_end_time(self):
        self.assertEqual(self.ramp(500), 0.9)

    def test_should_return_correct_value_at_midpoint(self):
        self.assertAlmostEqual(self.ramp(150), 0.195362337617)


class TestNonLinearRampDownWard(unittest.TestCase):
    def setUp(self):
        self.ramp = scalers.NonLinearRamp(100, 200, 0.9, 0.1, 4)

    def test_should_raise_with_negative_time(self):
        with self.assertRaises(ValueError):
            self.ramp(-1)

    def test_should_return_start_weight_before_start_time(self):
        self.assertEqual(self.ramp(0), 0.9)

    def test_should_return_end_weight_after_end_time(self):
        self.assertEqual(self.ramp(500), 0.1)

    def test_should_return_correct_value_at_midpoint(self):
        self.assertAlmostEqual(self.ramp(150), 0.195362337617)


class TestTimeRampSwitcher(unittest.TestCase):
    def setUp(self):
        self.first_ramp = mock.Mock()
        self.second_ramp = mock.Mock()
        self.ramp_switch = scalers.TimeRampSwitcher(
            self.first_ramp, self.second_ramp, 500
        )

    def test_should_call_first_ramp_before_switching_time(self):
        self.ramp_switch(0)
        self.first_ramp.assert_called_once_with(0)
        self.assertEqual(self.second_ramp.call_count, 0)

    def test_should_call_second_ramp_on_switching_time(self):
        self.ramp_switch(500)
        self.second_ramp.assert_called_once_with(500)
        self.assertEqual(self.first_ramp.call_count, 0)


class TestConstantPositioner(unittest.TestCase):
    def setUp(self):
        self.positioner = scalers.ConstantPositioner(42.0 * u.nanometer)

    def test_should_raise_when_alpha_below_zero(self):
        with self.assertRaises(ValueError):
            self.positioner(-1)

    def test_should_raise_when_alpha_above_one(self):
        with self.assertRaises(ValueError):
            self.positioner(2)

    def test_always_returns_same_value(self):
        self.assertEqual(self.positioner(0.0), 42.0)
        self.assertEqual(self.positioner(0.5), 42.0)
        self.assertEqual(self.positioner(1.0), 42.0)


class TestLinearPositioner(unittest.TestCase):
    def setUp(self):
        self.positioner = scalers.LinearPositioner(
            0.1, 0.9, 0 * u.nanometer, 100 * u.nanometer
        )

    def test_should_raise_when_alpha_below_zero(self):
        with self.assertRaises(ValueError):
            self.positioner(-1)

    def test_should_raise_when_alpha_above_one(self):
        with self.assertRaises(ValueError):
            self.positioner(2)

    def test_returns_min_value_below_alpha_min(self):
        self.assertAlmostEqual(self.positioner(0), 0.0)

    def test_returns_max_value_above_alpha_max(self):
        self.assertAlmostEqual(self.positioner(1.0), 100.0)

    def test_returns_mid_value_at_half_way(self):
        self.assertAlmostEqual(self.positioner(0.5), 50.0)
