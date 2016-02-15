#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import unittest
import mock


from meld import system
from meld.system import restraints


class TestAlwaysActiveCollection(unittest.TestCase):
    def setUp(self):
        self.coll = restraints.AlwaysActiveCollection()

    def test_adding_non_restraint_raises(self):
        with self.assertRaises(RuntimeError):
            self.coll.add_restraint(object)

    def test_should_be_able_to_add_restraint(self):
        rest = restraints.Restraint()
        self.coll.add_restraint(rest)
        self.assertIn(rest, self.coll.restraints)


class TestSelectivelyActiveCollection(unittest.TestCase):
    def test_restraint_should_be_present_after_adding(self):
        rest = [restraints.SelectableRestraint()]
        coll = restraints.SelectivelyActiveCollection(rest, 1)
        self.assertEqual(len(coll.groups), 1)

    def test_can_add_two_restraints(self):
        rest = [restraints.SelectableRestraint(), restraints.SelectableRestraint()]
        coll = restraints.SelectivelyActiveCollection(rest, 1)
        self.assertEqual(len(coll.groups), 2)

    def test_adding_non_selectable_restraint_should_raise(self):
        rest = [restraints.NonSelectableRestraint()]
        with self.assertRaises(RuntimeError):
            restraints.SelectivelyActiveCollection(rest, 1)

    def test_empty_restraint_list_should_raise(self):
        with self.assertRaises(RuntimeError):
            restraints.SelectivelyActiveCollection([], 0)

    def test_negative_num_active_should_raise(self):
        rest = [restraints.SelectableRestraint()]
        with self.assertRaises(RuntimeError):
            restraints.SelectivelyActiveCollection(rest, -1)

    def test_num_active_greater_than_num_restraints_should_raise(self):
        rest = [restraints.SelectableRestraint()]
        with self.assertRaises(RuntimeError):
            restraints.SelectivelyActiveCollection(rest, 2)

    def test_num_active_should_be_set(self):
        rest = [restraints.SelectableRestraint()]
        coll = restraints.SelectivelyActiveCollection(rest, 1)
        self.assertEqual(coll.num_active, 1)

    def test_should_wrap_bare_restraint_in_group(self):
        rest = [restraints.SelectableRestraint()]
        with mock.patch('meld.system.restraints.RestraintGroup.__init__', spec=True) as group_init:
            group_init.return_value = None
            restraints.SelectivelyActiveCollection(rest, 1)
            self.assertEqual(group_init.call_count, 1)

    def test_should_not_wrap_a_group_in_a_group(self):
        rest = [restraints.SelectableRestraint()]
        grps = [restraints.RestraintGroup(rest, 1)]
        with mock.patch('meld.system.restraints.RestraintGroup.__init__', spec=True) as group_init:
            restraints.SelectivelyActiveCollection(grps, 1)
            self.assertEqual(group_init.call_count, 0)


class TestRestraintGroup(unittest.TestCase):
    def test_should_accept_selectable_restraint(self):
        rest = [restraints.SelectableRestraint()]
        grp = restraints.RestraintGroup(rest, 1)
        self.assertEqual(len(grp.restraints), 1)

    def test_should_not_accept_non_selectable_restraint(self):
        rest = [restraints.NonSelectableRestraint()]
        with self.assertRaises(RuntimeError):
            restraints.RestraintGroup(rest, 1)

    def test_should_raise_on_empy_restraint_list(self):
        with self.assertRaises(RuntimeError):
            restraints.RestraintGroup([], 0)

    def test_num_active_below_zero_should_raise(self):
        rest = [restraints.SelectableRestraint()]
        with self.assertRaises(RuntimeError):
            restraints.RestraintGroup(rest, -1)

    def test_num_active_above_n_rest_should_raise(self):
        rest = [restraints.SelectableRestraint()]
        with self.assertRaises(RuntimeError):
            restraints.RestraintGroup(rest, 2)

    def test_num_active_should_be_set(self):
        rest = [restraints.SelectableRestraint()]
        grp = restraints.RestraintGroup(rest, 1)
        self.assertEqual(grp.num_active, 1)


class TestRestraintManager(unittest.TestCase):
    def setUp(self):
        self.mock_system = mock.Mock(spec=system.System)
        self.mock_system.index_of_atom.return_value = 0
        self.rest_manager = restraints.RestraintManager(self.mock_system)

    def test_can_add_as_always_active_non_selectable_restraint(self):
        rest = restraints.NonSelectableRestraint()
        self.rest_manager.add_as_always_active(rest)
        self.assertIn(rest, self.rest_manager.always_active)

    def test_can_add_as_always_active_selectable_restraint(self):
        rest = restraints.SelectableRestraint()
        self.rest_manager.add_as_always_active(rest)
        self.assertIn(rest, self.rest_manager.always_active)

    def test_can_add_list_of_always_active_restraints(self):
        rests = [restraints.SelectableRestraint(), restraints.NonSelectableRestraint()]
        self.rest_manager.add_as_always_active_list(rests)
        self.assertEqual(len(self.rest_manager.always_active), 2)

    def test_creating_bad_restraint_raises_error(self):
        with self.assertRaises(RuntimeError):
            self.rest_manager.create_restraint('blarg', x=42, y=99, z=-403)

    def test_can_create_distance_restraint(self):
        rest = self.rest_manager.create_restraint(
            'distance', atom_1_res_index=1, atom_1_name='CA',
            atom_2_res_index=2, atom_2_name='CA',
            r1=0, r2=0, r3=0.3, r4=999., k=2500)
        self.assertTrue(isinstance(rest, restraints.DistanceRestraint))

    def test_can_add_seletively_active_collection(self):
        rest_list = [restraints.SelectableRestraint(), restraints.SelectableRestraint()]
        self.rest_manager.add_selectively_active_collection(rest_list, 2)
        self.assertEqual(len(self.rest_manager.selectively_active_collections), 1)

    def test_can_create_restraint_group(self):
        rest_list = [restraints.SelectableRestraint(), restraints.SelectableRestraint()]
        grp = self.rest_manager.create_restraint_group(rest_list, 2)
        self.assertEqual(len(grp.restraints), 2)


class TestDistanceRestraint(unittest.TestCase):
    def setUp(self):
        self.mock_system = mock.Mock()
        self.scaler = restraints.ConstantScaler()
        self.ramp = restraints.ConstantRamp()

    def test_should_find_two_indices(self):
        restraints.DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 0, 0, 0.3, 999., 1.0)
        calls = [
            mock.call(1, 'CA'),
            mock.call(2, 'CA')]
        self.mock_system.index_of_atom.assert_has_calls(calls)

    def test_should_raise_on_bad_index(self):
        self.mock_system.index_of_atom.side_effect = KeyError()

        with self.assertRaises(KeyError):
            restraints.DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'BAD', 2, 'CA', 0, 0, 0.3, 999., 1.0)

    def test_should_raise_if_r2_less_than_r1(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 10., 0., 10., 10., 1.0)

    def test_should_raise_if_r3_less_than_r2(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 10., 10., 0., 10., 1.0)

    def test_should_raise_if_r4_less_than_r3(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 10., 10., 10., 0., 1.0)

    def test_should_raise_with_negative_r(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', -1., 10., 10., 10., 1.0)

    def test_should_raise_with_negative_k(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 10., 10., 10., 10., -1.0)


class TestHyperbolicDistanceRestraint(unittest.TestCase):
    def setUp(self):
        self.mock_system = mock.Mock()
        self.scaler = restraints.ConstantScaler()
        self.ramp = restraints.ConstantRamp()

    def test_should_find_two_indices(self):
        restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp, 1, 'CA', 2, 'CA', 0.0, 0.0, 0.6, 0.7, 1.0, 1.0)
        calls = [
            mock.call(1, 'CA'),
            mock.call(2, 'CA')]
        self.mock_system.index_of_atom.assert_has_calls(calls)

    def test_should_raise_on_bad_index(self):
        self.mock_system.index_of_atom.side_effect = KeyError()

        with self.assertRaises(KeyError):
            restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp,
                                                   1, 'BAD', 2, 'CA', 0.0, 0.1, 0.2, 0.3, 1.0, 1.0)

    def test_should_raise_with_negative_r(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp,
                                                   1, 'CA', 2, 'CA', -1., 1.0, 2.0, 3.0, 1.0, 1.0)

    def test_should_raise_if_r2_less_than_r1(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp,
                                                   1, 'CA', 2, 'CA', 10.0, 0.0, 1.0, 2.0, 1.0, 1.0)

    def test_should_raise_if_r3_less_than_r2(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp,
                                                   1, 'CA', 2, 'CA', 0.0, 1.0, 0.0, 2.0, 1.0, 1.0)

    def test_should_raise_if_r4_less_than_r3(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp,
                                                   1, 'CA', 2, 'CA', 0.0, 1.0, 2.0, 0.0, 1.0, 1.0)

    def test_should_raise_if_r4_equals_r3(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp,
                                                   1, 'CA', 2, 'CA', 0.0, 1.0, 2.0, 2.0, 1.0, 1.0)

    def test_should_raise_with_negative_k(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp,
                                                   1, 'CA', 2, 'CA', 1., 2., 3., 4., -1.0, 1.0)

    def test_should_raise_with_negative_asymptote(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(self.mock_system, self.scaler, self.ramp,
                                                   1, 'CA', 2, 'CA', 1., 2., 3., 4., 1.0, -1.0)


class TestTorsionRestraint(unittest.TestCase):
    def setUp(self):
        self.mock_system = mock.Mock()
        self.mock_system.index_of_atom.side_effect = [0, 1, 2, 3]
        self.scaler = mock.Mock()
        self.ramp = mock.Mock()

    def test_should_find_four_indices(self):
        restraints.TorsionRestraint(
            self.mock_system,
            self.scaler,
            self.ramp,
            1, 'CA',
            2, 'CA',
            3, 'CA',
            4, 'CA',
            180., 0., 1.0)

        calls = [
            mock.call(1, 'CA'),
            mock.call(2, 'CA'),
            mock.call(3, 'CA'),
            mock.call(4, 'CA')]
        self.mock_system.index_of_atom.assert_has_calls(calls)

    def test_should_raise_with_non_unique_indices(self):
        self.mock_system.index_of_atom.side_effect = [0, 0, 1, 2]  # repeated index
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'CA',
                1, 'CA',
                3, 'CA',
                4, 'CA',
                180., 0., 1.0)

    def test_should_fail_with_phi_below_minus_180(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'CA',
                2, 'CA',
                3, 'CA',
                4, 'CA',
                -270., 0., 1.0)

    def test_should_fail_with_phi_above_180(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'CA',
                2, 'CA',
                3, 'CA',
                4, 'CA',
                270., 0., 1.0)

    def test_should_fail_with_delta_phi_above_180(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'CA',
                2, 'CA',
                3, 'CA',
                4, 'CA',
                0., 200., 1.0)

    def test_should_fail_with_delta_phi_below_0(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'CA',
                2, 'CA',
                3, 'CA',
                4, 'CA',
                0., -90., 1.0)

    def test_should_fail_with_negative_k(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'CA',
                2, 'CA',
                3, 'CA',
                4, 'CA',
                0., 90., -1.0)


class TestRdcRestraint(unittest.TestCase):
    def setUp(self):
        self.mock_system = mock.Mock()
        self.mock_system.index_of_atom.side_effect = [0, 1]
        self.scaler = mock.Mock()
        self.ramp = mock.Mock()

    def test_should_find_four_indices(self):
        restraints.RdcRestraint(
            self.mock_system,
            self.scaler,
            self.ramp,
            1, 'N',
            1, 'H',
            100., 10.,
            10., 1.0,
            1.0, 0)

        calls = [
            mock.call(1, 'N'),
            mock.call(1, 'H')]
        self.mock_system.index_of_atom.assert_has_calls(calls)

    def test_should_raise_with_non_unique_indices(self):
        self.mock_system.index_of_atom.side_effect = [0, 0]  # repeated index
        with self.assertRaises(ValueError):
            restraints.RdcRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'N',
                1, 'N', # repeated index
                100., 10.,
                10., 1.0,
                1.0, 0)

    def test_should_raise_with_negative_tolerance(self):
        with self.assertRaises(ValueError):
            restraints.RdcRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'N',
                1, 'H',
                100., 10.,
                -10., 1.0,
                1.0, 0)

    def test_should_raise_with_negative_force_const(self):
        with self.assertRaises(ValueError):
            restraints.RdcRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'N',
                1, 'H',
                100., 10.,
                10., -1.0,
                1.0, 0)

    def test_should_raise_with_negative_weight(self):
        with self.assertRaises(ValueError):
            restraints.RdcRestraint(
                self.mock_system,
                self.scaler,
                self.ramp,
                1, 'N',
                1, 'H',
                100., 10.,
                10., 1.0,
                -1.0, 0)


class TestConstantScaler(unittest.TestCase):
    def test_should_return_1_when_alpha_is_0(self):
        scaler = restraints.ConstantScaler()
        self.assertAlmostEqual(scaler(0.0), 1.0)

    def test_should_return_1_when_alpha_is_1(self):
        scaler = restraints.ConstantScaler()
        self.assertAlmostEqual(scaler(1.0), 1.0)

    def test_should_raise_if_alpha_is_less_than_zero(self):
        scaler = restraints.ConstantScaler()
        with self.assertRaises(RuntimeError):
            scaler(-1.0)

    def test_should_raise_if_alpha_is_greater_than_one(self):
        scaler = restraints.ConstantScaler()
        with self.assertRaises(RuntimeError):
            scaler(2.0)

class TestLinearScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.LinearScaler(-1, 1)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.LinearScaler(2, 1)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.LinearScaler(1, -1)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.LinearScaler(1, 2)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            restraints.LinearScaler(0.7, 0.6)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = restraints.LinearScaler(0.2, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = restraints.LinearScaler(0.2, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_1_below_alpha_min(self):
        scaler = restraints.LinearScaler(0.2, 0.8)
        self.assertAlmostEqual(scaler(0.1), 1.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = restraints.LinearScaler(0.2, 0.8)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_correct_value_in_middle(self):
        scaler = restraints.LinearScaler(0.0, 1.0)
        self.assertAlmostEqual(scaler(0.3), 0.7)

class TestPlateauLinearScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauLinearScaler(-1,0,0.5, 1)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauLinearScaler(2,0,0.5, 1)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauLinearScaler(1,0,0.5, -1)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauLinearScaler(1,0,0.5, 2)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauLinearScaler(0.7,0,0.5, 0.6)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = restraints.PlateauLinearScaler(0.2,0.5,0.7,0.8)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = restraints.PlateauLinearScaler(0.2,0.6,0.7, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_0_below_alpha_min(self):
        scaler = restraints.PlateauLinearScaler(0.2,0.4,0.6, 0.8)
        self.assertAlmostEqual(scaler(0.1), 0.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = restraints.PlateauLinearScaler(0.2, 0.4,0.6,0.8)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_correct_value_between_alpha_one_alpha_two_down(self):
        scaler = restraints.PlateauLinearScaler(0.0, 0.4,0.6,1.0)
        self.assertAlmostEqual(scaler(0.3), 0.75)

    def test_should_return_correct_value_between_alpha_one_alpha_two_down2(self):
        scaler = restraints.PlateauLinearScaler(0.0, 0.4,0.6,1.0)
        self.assertAlmostEqual(scaler(0.1), 0.25)

    def test_should_return_correct_value_between_alpha_two_alpha_max_up(self):
        scaler = restraints.PlateauLinearScaler(0.0, 0.4,0.6,1.0)
        self.assertAlmostEqual(scaler(0.7), 0.75)

    def test_should_return_correct_value_between_alpha_two_alpha_max_up2(self):
        scaler = restraints.PlateauLinearScaler(0.0, 0.4,0.6,1.0)
        self.assertAlmostEqual(scaler(0.9), 0.25)

class TestNonLinearScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.NonLinearScaler(-1, 1, 4)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.NonLinearScaler(2, 1, 4)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.NonLinearScaler(1, -1, 4)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.NonLinearScaler(1, 2, 4)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            restraints.NonLinearScaler(0.7, 0.6, 4)

    def test_should_raise_if_factor_below_one(self):
        with self.assertRaises(RuntimeError):
            restraints.NonLinearScaler(0.0, 1.0, 0.2)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = restraints.NonLinearScaler(0.2, 0.8, 4)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = restraints.NonLinearScaler(0.2, 0.8, 4)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_1_below_alpha_min(self):
        scaler = restraints.NonLinearScaler(0.2, 0.8, 4)
        self.assertAlmostEqual(scaler(0.1), 1.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = restraints.NonLinearScaler(0.2, 0.8, 4)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_midpoint_should_return_correct_value(self):
        scaler = restraints.NonLinearScaler(0.2, 0.8, 4)
        self.assertAlmostEqual(scaler(0.5), 0.119202922)

class TestPlateauNonLinearScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauNonLinearScaler(-1,0.5,0.6,1, 4)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauNonLinearScaler(2,0.5,0.6,1, 4)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauNonLinearScaler(1,0.5,0.6,-1, 4)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauNonLinearScaler(1,0.5,0.6,2, 4)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauNonLinearScaler(0.7,0.65,0.63, 0.6, 4)

    def test_should_raise_if_factor_below_one(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauNonLinearScaler(0.0,0.5,0.7, 1.0, 0.2)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = restraints.PlateauNonLinearScaler(0.2, 0.4,0.6,0.8, 4)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = restraints.PlateauNonLinearScaler(0.2,0.4,0.6, 0.8, 4)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_0_below_alpha_min(self):
        scaler = restraints.PlateauNonLinearScaler(0.2,0.4,0.6, 0.8, 4)
        self.assertAlmostEqual(scaler(0.1), 0.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = restraints.PlateauNonLinearScaler(0.2, 0.4,0.6,0.8, 4)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_1_between_alpha_one_alpha_two(self):
        scaler = restraints.PlateauNonLinearScaler(0.2, 0.4,0.6,0.8, 4)
        self.assertAlmostEqual(scaler(0.5), 1.0)

    def test_midpoint_should_return_correct_value_scaling_up(self):
        scaler = restraints.PlateauNonLinearScaler(0.7, 0.8,0.9,1.0, 4)
        self.assertAlmostEqual(scaler(0.95), 0.119202922)

    def test_midpoint_should_return_correct_value_scaling_down(self):
        scaler = restraints.PlateauNonLinearScaler(0.7, 0.8,0.9,1.0, 4)
        self.assertAlmostEqual(scaler(0.75), 0.88079708)

class TestSmoothScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.SmoothScaler(-1, 1)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.SmoothScaler(2, 1)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.SmoothScaler(1,-1)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.SmoothScaler(1, 2)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            restraints.SmoothScaler(0.7, 0.6)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = restraints.SmoothScaler(0.2,0.8)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = restraints.SmoothScaler(0.2, 0.8)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_1_below_alpha_min(self):
        scaler = restraints.SmoothScaler(0.2, 0.8)
        self.assertAlmostEqual(scaler(0.1), 1.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = restraints.SmoothScaler(0.2,0.8)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_correct_value_middle(self):
        scaler = restraints.SmoothScaler(0.5,1.0)
        self.assertAlmostEqual(scaler(0.75), 0.5)

    def test_should_return_correct_value_middle2(self):
        scaler = restraints.SmoothScaler(0.5,1.0)
        self.assertAlmostEqual(scaler(0.90), 0.104)

class TestPlateauSmoothScaler(unittest.TestCase):
    def test_should_raise_when_alpha_min_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauSmoothScaler(-1,0.5,0.6, 1)

    def test_should_raise_when_alpha_min_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauSmoothScaler(2,0.5,0.6, 1)

    def test_should_raise_when_alpha_max_below_zero(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauSmoothScaler(1,0.5,0.6,-1)

    def test_should_raise_when_alpha_max_above_one(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauSmoothScaler(1,0.5,0.6, 2)

    def test_should_raise_if_alpha_max_less_than_alpha_min(self):
        with self.assertRaises(RuntimeError):
            restraints.PlateauSmoothScaler(0.7,0.5,0.6, 0.6)

    def test_should_raise_if_alpha_is_below_zero(self):
        scaler = restraints.PlateauSmoothScaler(0.2,0.5,0.6,0.8)
        with self.assertRaises(RuntimeError):
            scaler(-1)

    def test_should_raise_if_alpha_is_above_one(self):
        scaler = restraints.PlateauSmoothScaler(0.2, 0.5,0.6,0.8)
        with self.assertRaises(RuntimeError):
            scaler(2)

    def test_should_return_0_below_alpha_min(self):
        scaler = restraints.PlateauSmoothScaler(0.2,0.5,0.6, 0.8)
        self.assertAlmostEqual(scaler(0.1), 0.0)

    def test_should_return_0_above_alpha_max(self):
        scaler = restraints.PlateauSmoothScaler(0.2,0.5,0.6,0.8)
        self.assertAlmostEqual(scaler(0.9), 0.0)

    def test_should_return_1_between_alpha_one_alpha_two(self):
        scaler = restraints.PlateauSmoothScaler(0.2,0.4,0.6,0.8)
        self.assertAlmostEqual(scaler(0.5), 1.0)

    def test_should_return_correct_value_middle_up(self):
        scaler = restraints.PlateauSmoothScaler(0.2,0.4,0.6,0.8)
        self.assertAlmostEqual(scaler(0.75), 0.15625)

    def test_should_return_correct_value_middle_up2(self):
        scaler = restraints.PlateauSmoothScaler(0.2,0.4,0.6,0.8)
        self.assertAlmostEqual(scaler(0.65), 0.84375)

    def test_should_return_correct_value_middle_down(self):
        scaler = restraints.PlateauSmoothScaler(0.2,0.4,0.6,0.8)
        self.assertAlmostEqual(scaler(0.35), 0.84375)

    def test_should_return_correct_value_middle_down2(self):
        scaler = restraints.PlateauSmoothScaler(0.2,0.4,0.6,0.8)
        self.assertAlmostEqual(scaler(0.25), 0.15625)

class TestCreateRestraintsAndScalers(unittest.TestCase):
    def setUp(self):
        self.mock_system = mock.Mock()
        self.manager = restraints.RestraintManager(self.mock_system)

    def test_can_create_constant_scaler(self):
        scaler = self.manager.create_scaler('constant')
        self.assertTrue(isinstance(scaler, restraints.ConstantScaler))

    def test_can_create_linear_scaler(self):
        scaler = self.manager.create_scaler('linear', alpha_min=0.2, alpha_max=0.8)
        self.assertTrue(isinstance(scaler, restraints.LinearScaler))

    def test_can_create_non_linear_scaler(self):
        scaler = self.manager.create_scaler('nonlinear', alpha_min=0.2, alpha_max=0.8, factor=4)
        self.assertTrue(isinstance(scaler, restraints.NonLinearScaler))

    def test_creating_restraint_without_specifying_scaler_uses_constant(self):
        self.mock_system.index_of_atom.side_effect = [0, 1]
        rest = self.manager.create_restraint(
            'distance',
            atom_1_res_index=1,
            atom_1_name='CA',
            atom_2_res_index=2,
            atom_2_name='CA',
            r1=0, r2=1, r3=3, r4=4, k=1.0)
        self.assertTrue(isinstance(rest.scaler, restraints.ConstantScaler))

    def test_creating_restraint_with_scaler_should_use_it(self):
        self.mock_system.index_of_atom.side_effect = [0, 1]
        scaler = restraints.LinearScaler(0, 1)
        rest = self.manager.create_restraint(
            'distance',
            scaler,
            atom_1_res_index=1,
            atom_1_name='CA',
            atom_2_res_index=2,
            atom_2_name='CA',
            r1=0, r2=1, r3=3, r4=4, k=1.0)
        self.assertTrue(isinstance(rest.scaler, restraints.LinearScaler))

    def test_creating_restraint_should_raise_if_scaler_is_wrong_type(self):
        scaler = restraints.TimeRamp()
        self.mock_system.index_of_atom.side_effect = [0, 1]
        with self.assertRaises(ValueError):
            rest = self.manager.create_restraint(
                'distance',
                scaler,
                atom_1_res_index=1,
                atom_1_name='CA',
                atom_2_res_index=2,
                atom_2_name='CA',
                r1=0., r2=1., r3=3., r4=4., k=1.0)

    def test_creating_restraint_should_raise_if_ramp_is_wrong_type(self):
        scaler = restraints.ConstantScaler()
        ramp = restraints.ConstantScaler()
        self.mock_system.index_of_atom.side_effect = [0, 1]
        with self.assertRaises(ValueError):
            rest = self.manager.create_restraint(
                'distance',
                scaler,
                ramp=ramp,
                atom_1_res_index=1,
                atom_1_name='CA',
                atom_2_res_index=2,
                atom_2_name='CA',
                r1=0., r2=1., r3=3., r4=4., k=1.0)

    def test_create_restraint_without_specifying_ramp_should_use_constant_ramp(self):
        scaler = restraints.ConstantScaler()
        self.mock_system.index_of_atom.side_effect = [0, 1]
        rest = self.manager.create_restraint(
            'distance',
            scaler,
            atom_1_res_index=1,
            atom_1_name='CA',
            atom_2_res_index=2,
            atom_2_name='CA',
            r1=0., r2=1., r3=3., r4=4., k=1.0)
        self.assertTrue(isinstance(rest.ramp, restraints.ConstantRamp))

class TestConstantRamp(unittest.TestCase):
    def setUp(self):
        self.ramp = restraints.ConstantRamp()

    def test_should_raise_with_negative_time(self):
        with self.assertRaises(ValueError):
            self.ramp(-1)

    def test_should_always_return_one(self):
        self.assertEqual(self.ramp(0), 1.0)
        self.assertEqual(self.ramp(1000), 1.0)
        self.assertEqual(self.ramp(1000000000), 1.0)


class TestLinearRamp(unittest.TestCase):
    def setUp(self):
        self.ramp = restraints.LinearRamp(100, 200, 0.1, 0.9)

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
        self.ramp = restraints.NonLinearRamp(100, 200, 0.1, 0.9, 4)

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
        self.ramp = restraints.NonLinearRamp(100, 200, 0.9, 0.1, 4)

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
        self.ramp_switch = restraints.TimeRampSwitcher(self.first_ramp, self.second_ramp, 500)

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
        self.positioner = restraints.ConstantPositioner(42.0)

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
        self.positioner = restraints.LinearPositioner(0.1, 0.9, 0, 100)

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
