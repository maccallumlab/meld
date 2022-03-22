#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

import numpy as np  # type: ignore
import unittest
from unittest import mock  # type: ignore


from meld.system import restraints
from meld.system import scalers
from meld import AmberSubSystemFromSequence, AmberSystemBuilder, AmberOptions
from openmm import unit as u  # type: ignore


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
        with mock.patch(
            "meld.system.restraints.RestraintGroup.__init__", spec=True
        ) as group_init:
            group_init.return_value = None
            restraints.SelectivelyActiveCollection(rest, 1)
            self.assertEqual(group_init.call_count, 1)

    def test_should_not_wrap_a_group_in_a_group(self):
        rest = [restraints.SelectableRestraint()]
        grps = [restraints.RestraintGroup(rest, 1)]
        with mock.patch(
            "meld.system.restraints.RestraintGroup.__init__", spec=True
        ) as group_init:
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
        p = AmberSubSystemFromSequence("GLY GLY GLY GLY")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p])
        self.rest_manager = restraints.RestraintManager(self.system)

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
            self.rest_manager.create_restraint("blarg", x=42, y=99, z=-403)

    def test_can_create_distance_restraint(self):
        rest = self.rest_manager.create_restraint(
            "distance",
            atom1=self.system.index.atom(0, "CA"),
            atom2=self.system.index.atom(1, "CA"),
            r1=0 * u.nanometer,
            r2=0 * u.nanometer,
            r3=0.3 * u.nanometer,
            r4=999.0 * u.nanometer,
            k=2500 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
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
        p = AmberSubSystemFromSequence("GLY GLY GLY GLY")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p])
        self.scaler = restraints.ConstantScaler()
        self.ramp = restraints.ConstantRamp()

    def test_should_raise_if_r2_less_than_r1(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                10.0 * u.nanometer,
                0.0 * u.nanometer,
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
            )

    def test_should_raise_if_r3_less_than_r2(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                0.0 * u.nanometer,
                10.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
            )

    def test_should_raise_if_r4_less_than_r3(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                0.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
            )

    def test_should_raise_with_negative_r(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                -1.0 * u.nanometer,
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
            )

    def test_should_raise_with_negative_k(self):
        with self.assertRaises(RuntimeError):
            restraints.DistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                10.0 * u.nanometer,
                -1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
            )


class TestHyperbolicDistanceRestraint(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("GLY GLY GLY GLY")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p])
        self.scaler = restraints.ConstantScaler()
        self.ramp = restraints.ConstantRamp()

    def test_should_raise_with_negative_r(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                -1.0 * u.nanometer,
                1.0 * u.nanometer,
                2.0 * u.nanometer,
                3.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                1.0 * u.kilojoule_per_mole,
            )

    def test_should_raise_if_r2_less_than_r1(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                10.0 * u.nanometer,
                0.0 * u.nanometer,
                1.0 * u.nanometer,
                2.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                1.0 * u.kilojoule_per_mole,
            )

    def test_should_raise_if_r3_less_than_r2(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                0.0 * u.nanometer,
                1.0 * u.nanometer,
                0.0 * u.nanometer,
                2.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                1.0 * u.kilojoule_per_mole,
            )

    def test_should_raise_if_r4_less_than_r3(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                0.0 * u.nanometer,
                1.0 * u.nanometer,
                2.0 * u.nanometer,
                0.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                1.0 * u.kilojoule_per_mole,
            )

    def test_should_raise_if_r4_equals_r3(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                0.0 * u.nanometer,
                1.0 * u.nanometer,
                2.0 * u.nanometer,
                2.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                1.0 * u.kilojoule_per_mole,
            )

    def test_should_raise_with_negative_k(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                1.0 * u.nanometer,
                2.0 * u.nanometer,
                3.0 * u.nanometer,
                4.0 * u.nanometer,
                -1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                1.0 * u.kilojoule_per_mole,
            )

    def test_should_raise_with_negative_asymptote(self):
        with self.assertRaises(RuntimeError):
            restraints.HyperbolicDistanceRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "CA"),
                self.system.index.atom(1, "CA"),
                1.0 * u.nanometer,
                2.0 * u.nanometer,
                3.0 * u.nanometer,
                4.0 * u.nanometer,
                1.0 * u.kilojoule_per_mole / u.nanometer ** 2,
                -1.0 * u.kilojoule_per_mole,
            )


class TestTorsionRestraint(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("GLY GLY GLY GLY")
        options = AmberOptions()
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p])
        self.scaler = mock.Mock()
        self.ramp = mock.Mock()

    def test_should_raise_with_non_unique_indices(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "C"),
                self.system.index.atom(1, "C"),
                self.system.index.atom(1, "CA"),
                self.system.index.atom(1, "C"),
                180.0 * u.degree,
                0.0 * u.degree,
                1.0 * u.kilojoule_per_mole / u.degree ** 2,
            )

    def test_should_fail_with_phi_below_minus_180(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "C"),
                self.system.index.atom(1, "N"),
                self.system.index.atom(1, "CA"),
                self.system.index.atom(1, "C"),
                -270.0 * u.degree,
                0.0 * u.degree,
                1.0 * u.kilojoule_per_mole / u.degree ** 2,
            )

    def test_should_fail_with_phi_above_180(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "C"),
                self.system.index.atom(1, "N"),
                self.system.index.atom(1, "CA"),
                self.system.index.atom(1, "C"),
                270.0 * u.degree,
                0.0 * u.degree,
                1.0 * u.kilojoule_per_mole / u.degree ** 2,
            )

    def test_should_fail_with_delta_phi_above_180(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "C"),
                self.system.index.atom(1, "N"),
                self.system.index.atom(1, "CA"),
                self.system.index.atom(1, "C"),
                0.0 * u.degree,
                200.0 * u.degree,
                1.0 * u.kilojoule_per_mole / u.degree ** 2,
            )

    def test_should_fail_with_delta_phi_below_0(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "C"),
                self.system.index.atom(1, "N"),
                self.system.index.atom(1, "CA"),
                self.system.index.atom(1, "C"),
                0.0 * u.degree,
                -90.0 * u.degree,
                1.0 * u.kilojoule_per_mole / u.degree ** 2,
            )

    def test_should_fail_with_negative_k(self):
        with self.assertRaises(RuntimeError):
            restraints.TorsionRestraint(
                self.system,
                self.scaler,
                self.ramp,
                self.system.index.atom(0, "C"),
                self.system.index.atom(1, "N"),
                self.system.index.atom(1, "CA"),
                self.system.index.atom(1, "C"),
                0.0 * u.degree,
                90.0 * u.degree,
                -1.0 * u.kilojoule_per_mole / u.degree ** 2,
            )


class TestCOMRestraint(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("GLY GLY GLY GLY")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p])
        self.scaler = scalers.ConstantScaler()
        self.ramp = scalers.ConstantRamp()

    def test_should_raise_when_dims_has_non_xyz(self):
        group1 = [self.system.index.atom(0, "CA")]
        group2 = [self.system.index.atom(1, "CA")]
        with self.assertRaises(ValueError):
            restraints.COMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group1=group1,
                group2=group2,
                weights1=None,
                weights2=None,
                dims="a",
                force_const=1 * u.kilojoule_per_mole / u.nanometer ** 2,
                distance=1 * u.nanometer,
            )

    def test_should_raise_on_repeated_dim(self):
        group1 = [self.system.index.atom(0, "CA")]
        group2 = [self.system.index.atom(1, "CA")]
        with self.assertRaises(ValueError):
            restraints.COMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group1=group1,
                group2=group2,
                weights1=None,
                weights2=None,
                dims="xx",
                force_const=1 * u.kilojoule_per_mole / u.nanometer ** 2,
                distance=1 * u.nanometer,
            )

    def test_should_raise_on_negative_k(self):
        group1 = [self.system.index.atom(0, "CA")]
        group2 = [self.system.index.atom(1, "CA")]
        with self.assertRaises(ValueError):
            restraints.COMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group1=group1,
                group2=group2,
                weights1=None,
                weights2=None,
                dims="x",
                force_const=-1 * u.kilojoule_per_mole / u.nanometer ** 2,
                distance=1 * u.nanometer,
            )

    def test_should_raise_on_negative_distance(self):
        group1 = [self.system.index.atom(0, "CA")]
        group2 = [self.system.index.atom(1, "CA")]
        with self.assertRaises(ValueError):
            restraints.COMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group1=group1,
                group2=group2,
                weights1=None,
                weights2=None,
                dims="x",
                force_const=1 * u.kilojoule_per_mole / u.nanometer ** 2,
                distance=-1 * u.nanometer,
            )

    def test_should_raise_on_group1_size_mismatch(self):
        group1 = [self.system.index.atom(0, "CA")]
        group2 = [self.system.index.atom(1, "CA")]
        with self.assertRaises(ValueError):
            restraints.COMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group1=group1,
                group2=group2,
                weights1=np.array([1.0, 1.0]),  # wrong length
                weights2=None,
                dims="x",
                force_const=1,
                distance=1,
            )

    def test_should_raise_on_group2_size_mismatch(self):
        group1 = [self.system.index.atom(0, "CA")]
        group2 = [self.system.index.atom(1, "CA")]
        with self.assertRaises(ValueError):
            restraints.COMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group1=group1,
                group2=group2,
                weights1=None,
                weights2=np.array([1.0, 1.0]),  # wrong length
                dims="x",
                force_const=1,
                distance=1,
            )

    def test_should_raise_on_negative_weights1(self):
        group1 = [self.system.index.atom(0, "CA"), self.system.index.atom(1, "CA")]
        group2 = [self.system.index.atom(2, "CA"), self.system.index.atom(3, "CA")]
        with self.assertRaises(ValueError):
            restraints.COMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group1=group1,
                group2=group2,
                weights1=[1.0, -1.0],
                weights2=None,
                dims="x",
                force_const=1,
                distance=1,
            )

    def test_should_raise_on_negative_weights2(self):
        group1 = [self.system.index.atom(0, "CA"), self.system.index.atom(1, "CA")]
        group2 = [self.system.index.atom(2, "CA"), self.system.index.atom(3, "CA")]
        with self.assertRaises(ValueError):
            restraints.COMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group1=group1,
                group2=group2,
                weights1=None,
                weights2=[1.0, -1.0],
                dims="x",
                force_const=1,
                distance=1,
            )


class TestAbsoluteCOMRestraint(unittest.TestCase):
    def setUp(self):
        p = AmberSubSystemFromSequence("GLY GLY GLY GLY")
        options = AmberOptions()
        b = AmberSystemBuilder(options)
        self.system = b.build_system([p])
        self.scaler = scalers.ConstantScaler()
        self.ramp = scalers.ConstantRamp()

    def test_should_raise_when_dims_has_non_xyz(self):
        with self.assertRaises(ValueError):
            restraints.AbsoluteCOMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group=[(1, "CA")],
                weights=None,
                dims="a",  # not xyz
                force_const=1 * u.kilojoule_per_mole / u.nanometer ** 2,
                position=np.array([0.0, 0.0, 0.0]) * u.nanometer,
            )

    def test_should_raise_on_repeated_dim(self):
        with self.assertRaises(ValueError):
            restraints.AbsoluteCOMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group=[(1, "CA")],
                weights=None,
                dims="xx",  # repeated
                force_const=1 * u.kilojoule_per_mole / u.nanometer ** 2,
                position=np.array([0.0, 0.0, 0.0]) * u.nanometer,
            )

    def test_should_raise_on_negative_k(self):
        with self.assertRaises(ValueError):
            restraints.AbsoluteCOMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group=[(1, "CA")],
                weights=None,
                dims="xyz",
                force_const=-1
                * u.kilojoule_per_mole
                / u.nanometer ** 2,  # negative force const
                position=np.array([0.0, 0.0, 0.0]) * u.nanometer,
            )

    def test_should_raise_on_position_wrong_shape(self):
        with self.assertRaises(ValueError):
            restraints.AbsoluteCOMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group=[(1, "CA")],
                weights=None,
                dims="xyz",
                force_const=1 * u.kilojoule_per_mole / u.nanometer ** 2,
                position=np.array([0.0, 0.0, 0.0, 0.0]) * u.nanometer,
            )  # too many positions

    def test_should_raise_on_size_mismatch(self):
        group = [self.system.index.atom(0, "CA")]
        with self.assertRaises(ValueError):
            restraints.AbsoluteCOMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group=group,
                weights=[1.0, 1.0],  # too many weights
                dims="xyz",
                force_const=1 * u.kilojoule_per_mole / u.nanometer ** 2,
                position=np.array([0.0, 0.0, 0.0]) * u.nanometer,
            )

    def test_should_raise_on_negative_weight(self):
        group = [self.system.index.atom(0, "CA"), self.system.index.atom(1, "CA")]
        with self.assertRaises(ValueError):
            restraints.AbsoluteCOMRestraint(
                system=self.system,
                scaler=self.scaler,
                ramp=self.ramp,
                group=group,
                weights=[1.0, -1.0],
                dims="xyz",
                force_const=1 * u.kilojoule_per_mole / u.nanometer ** 2,
                position=np.array([0.0, 0.0, 0.0]) * u.nanometer,
            )
