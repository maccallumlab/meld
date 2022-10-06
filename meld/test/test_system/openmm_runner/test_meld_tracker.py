import unittest
from meld.runner.transform.restraints.meld import tracker
from meld.system import restraints
from meld.system import scalers
import numpy as np
import meld
from copy import deepcopy

from openmm import unit as u  # type: ignore

from meld.system.builders.amber.builder import AmberOptions  # type: ignore


class TestTrackerWithNoDependencies(unittest.TestCase):
    def setUp(self):
        p = meld.AmberSubSystemFromSequence(sequence="NALA ALA CALA")

        # build the system
        options = meld.AmberOptions()
        b = meld.AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()
        self.state = self.system.get_state_template()
        self.tracker = tracker.RestraintTracker(
            self.system.param_sampler, self.system.mapper
        )

        r1 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r1, 0.0, 0.0, self.state)
        r2 = restraints.TorsionRestraint(
            self.system,
            None,
            None,
            self.system.index.atom(1, "N"),
            self.system.index.atom(1, "CA"),
            self.system.index.atom(1, "C"),
            self.system.index.atom(2, "N"),
            90 * u.degrees,
            10 * u.degrees,
            0 * u.kilojoule_per_mole / u.degrees ** 2,
        )
        self.tracker.add_torsion_restraint(r2, 0.0, 0.0, self.state)

    def test_new_retraints_should_be_updated(self):
        update = self.tracker.get_and_reset_need_update()
        self.assertEqual(len(update), 2)
        self.assertIn(("distance", 0), update)
        self.assertIn(("torsion", 0), update)

    def test_no_restraints_should_be_updated_after_reset(self):
        _ = self.tracker.get_and_reset_need_update()
        update = self.tracker.get_and_reset_need_update()
        self.assertEqual(len(update), 0)


class TestTrackerWithScalerDependency(unittest.TestCase):
    def setUp(self):
        p = meld.AmberSubSystemFromSequence(sequence="NALA ALA CALA")

        # build the system
        options = meld.AmberOptions()
        b = meld.AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()
        self.state = self.system.get_state_template()
        self.tracker = tracker.RestraintTracker(
            self.system.param_sampler, self.system.mapper
        )

        self.scaler1 = scalers.LinearScaler(0.0, 0.5)
        self.scaler2 = scalers.LinearScaler(0.5, 1.0)
        r1 = restraints.DistanceRestraint(
            self.system,
            self.scaler1,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r1, 0.0, 0.0, self.state)
        r2 = restraints.DistanceRestraint(
            self.system,
            self.scaler2,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r2, 0.0, 0.0, self.state)
        r3 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r3, 0.0, 0.0, self.state)

    def test_should_update_only_scaler1(self):
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.2, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 1)
        self.assertIn(("distance", 0), update)

    def test_should_update_both_scalers(self):
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.8, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 2)
        self.assertIn(("distance", 0), update)
        self.assertIn(("distance", 1), update)

    def test_should_update_only_scaler2(self):
        self.tracker.update(0.7, 0, self.state)
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.8, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 1)
        self.assertIn(("distance", 1), update)

    def test_should_not_update_with_unchanged_alpha(self):
        self.tracker.update(0.5, 0, self.state)
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.5, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 0)


class TestTrackerWithRampDependency(unittest.TestCase):
    def setUp(self):
        p = meld.AmberSubSystemFromSequence(sequence="NALA ALA CALA")

        # build the system
        options = meld.AmberOptions()
        b = meld.AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()
        self.state = self.system.get_state_template()
        self.tracker = tracker.RestraintTracker(
            self.system.param_sampler, self.system.mapper
        )

        self.ramp1 = scalers.LinearRamp(0, 100, 0.0, 1.0)
        self.ramp2 = scalers.LinearRamp(0, 200, 0.0, 1.0)
        r1 = restraints.DistanceRestraint(
            self.system,
            None,
            self.ramp1,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r1, 0.0, 0.0, self.state)
        r2 = restraints.DistanceRestraint(
            self.system,
            None,
            self.ramp2,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r2, 0.0, 0.0, self.state)
        r3 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r3, 0.0, 0.0, self.state)

    def test_should_update_both_ramps(self):
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.0, 1, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 2)
        self.assertIn(("distance", 0), update)
        self.assertIn(("distance", 1), update)

    def test_should_update_only_ramp2(self):
        self.tracker.update(0.0, 150, self.state)
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.0, 151, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 1)
        self.assertIn(("distance", 1), update)

    def test_should_not_update_with_unchanged_step(self):
        self.tracker.update(0, 50, self.state)
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0, 50, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 0)

    def test_should_not_update_outside_ramp_range(self):
        self.tracker.update(0, 500, self.state)
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0, 501, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 0)


class TestTrackerWithPositionerDependency(unittest.TestCase):
    def setUp(self):
        p = meld.AmberSubSystemFromSequence(sequence="NALA ALA CALA")

        # build the system
        options = meld.AmberOptions()
        b = meld.AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()
        self.state = self.system.get_state_template()
        self.tracker = tracker.RestraintTracker(
            self.system.param_sampler, self.system.mapper
        )

        self.positioner1 = scalers.LinearPositioner(
            0.0, 0.5, 1.0 * u.nanometer, 2.0 * u.nanometer
        )
        self.positioner2 = scalers.LinearPositioner(
            0.5, 1.0, 1.0 * u.nanometer, 2.0 * u.nanometer
        )
        r1 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            self.positioner1,
            self.positioner1,
            999 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r1, 0.0, 0.0, self.state)
        r2 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            self.positioner2,
            self.positioner2,
            999 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r2, 0.0, 0.0, self.state)
        r3 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r3, 0.0, 0.0, self.state)

    def test_should_update_both_positioners(self):
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.6, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 2)
        self.assertIn(("distance", 0), update)
        self.assertIn(("distance", 1), update)

    def test_should_update_only_positioner1(self):
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.1, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 1)
        self.assertIn(("distance", 0), update)

    def test_should_update_only_positioner2(self):
        self.tracker.update(0.7, 0, self.state)
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.8, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 1)
        self.assertIn(("distance", 1), update)

    def test_should_not_update_on_unchanged_alpha(self):
        self.tracker.update(0.2, 0, self.state)
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.2, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 0)


class TestTrackerWithPeakMapperDependency(unittest.TestCase):
    def setUp(self):
        p = meld.AmberSubSystemFromSequence(sequence="NALA ALA CALA")

        # build the system
        options = meld.AmberOptions()
        b = meld.AmberSystemBuilder(options)
        spec = b.build_system([p])
        self.system = spec.finalize()
        self.tracker = tracker.RestraintTracker(
            self.system.param_sampler, self.system.mapper
        )

        self.map = self.system.mapper.add_map("map", 2, 2, atom_names=["CA"])
        self.map.add_atom_group(CA=self.system.index.atom(0, "CA"))
        self.map.add_atom_group(CA=self.system.index.atom(1, "CA"))
        self.map.add_atom_group(CA=self.system.index.atom(2, "CA"))
        self.state = self.system.get_state_template()
        r1 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.map.get_mapping(0, "CA"),
            self.system.index.atom(1, "N"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r1, 0.0, 0.0, self.state)
        r2 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.map.get_mapping(1, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r2, 0.0, 0.0, self.state)
        r3 = restraints.DistanceRestraint(
            self.system,
            None,
            None,
            self.system.index.atom(0, "CA"),
            self.system.index.atom(2, "CA"),
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.nanometer,
            0 * u.kilojoule_per_mole / u.nanometer ** 2,
        )
        self.tracker.add_distance_restraint(r3, 0.0, 0.0, self.state)

    def test_should_not_update_on_unchanged_mapping(self):
        self.tracker.update(0.0, 0, self.state)
        _ = self.tracker.get_and_reset_need_update()
        self.tracker.update(0.0, 0, self.state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 0)

    def test_should_update_mapping1(self):
        self.tracker.update(0.0, 0, self.state)
        _ = self.tracker.get_and_reset_need_update()
        pert_mappings = np.array([2, 1])
        pert_state = deepcopy(self.state)
        print("state size before", self.state.mappings.shape)
        pert_state.mappings = pert_mappings
        print("state size after", pert_state.mappings.shape)

        self.tracker.update(0.0, 0, pert_state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 1)
        self.assertIn(("distance", 0), update)

    def test_should_update_mapping2(self):
        self.tracker.update(0.0, 0, self.state)
        _ = self.tracker.get_and_reset_need_update()
        pert_mappings = np.array([0, 2])
        pert_state = deepcopy(self.state)
        pert_state.mappings = pert_mappings

        self.tracker.update(0.0, 0, pert_state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 1)
        self.assertIn(("distance", 1), update)

    def test_should_update_both(self):
        self.tracker.update(0.0, 0, self.state)
        _ = self.tracker.get_and_reset_need_update()
        pert_mappings = np.array([1, 0])
        pert_state = deepcopy(self.state)
        pert_state.mappings = pert_mappings

        self.tracker.update(0.0, 0, pert_state)
        update = self.tracker.get_and_reset_need_update()

        self.assertEqual(len(update), 2)
        self.assertIn(("distance", 0), update)
        self.assertIn(("distance", 1), update)
