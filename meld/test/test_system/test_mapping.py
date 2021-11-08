import unittest
from meld.system.indexing import AtomIndex
from meld.system import mapping

import numpy as np  # type: ignore


class TestPeakMapper(unittest.TestCase):
    def test_should_raise_with_negative_n_peeaks(self):
        with self.assertRaises(ValueError):
            mapping.PeakMapper("test", n_peaks=-1, atom_names=["N", "H"])

    def test_add_atom_group_should_fail_with_too_few(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        with self.assertRaises(KeyError):
            mapper.add_atom_group(N=AtomIndex(0))

    def test_add_atom_group_should_fail_with_extra(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        with self.assertRaises(KeyError):
            mapper.add_atom_group(N=AtomIndex(0), H=AtomIndex(1), Q=AtomIndex(2))

    def test_add_atom_group_should_fail_with_mismatch(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        with self.assertRaises(KeyError):
            # no Q
            mapper.add_atom_group(N=AtomIndex(0), Q=AtomIndex(1))

    def test_should_raise_with_non_atom_index(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        with self.assertRaises(ValueError):
            mapper.add_atom_group(N=0, H=AtomIndex(1))

    def test_atom_groups_length_should_match(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        mapper.add_atom_group(N=AtomIndex(0), H=AtomIndex(1))
        mapper.add_atom_group(N=AtomIndex(2), H=AtomIndex(3))
        mapper.add_atom_group(N=AtomIndex(4), H=AtomIndex(5))

        self.assertEqual(mapper.n_atom_groups, 3)

    def test_get_atom_mapping_should_raise_with_bad_peak_id(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])

        with self.assertRaises(KeyError):
            # no peak_id 10
            mapper.get_mapping(peak_id=10, atom_name="H")

        with self.assertRaises(KeyError):
            # peak_id -1 is bad
            mapper.get_mapping(peak_id=-1, atom_name="H")

    def test_get_atom_mapping_should_raise_with_name(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])

        with self.assertRaises(KeyError):
            # no name Q
            mapper.get_mapping(peak_id=0, atom_name="Q")

    def test_get_atom_mapping_should_have_correct_name(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        m = mapper.get_mapping(peak_id=0, atom_name="N")

        self.assertEqual(m.map_name, "test")

    def test_get_atom_mapping_should_have_correct_peak_id(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        m = mapper.get_mapping(peak_id=0, atom_name="N")

        self.assertEqual(m.peak_id, 0)

    def test_get_atom_mapping_should_have_correct_atom_name(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        m = mapper.get_mapping(peak_id=0, atom_name="N")

        self.assertEqual(m.atom_name, "N")

    def test_initial_state_should_have_correct_size(self):
        mapper1 = mapping.PeakMapper("test", n_peaks=3, atom_names=["N", "H"])
        mapper1.add_atom_group(N=AtomIndex(0), H=AtomIndex(1))

        # mapper1 has 3 peaks, but 1 atom_group
        # Initial state should have the size of max of these two,
        # so it should be 3
        self.assertEqual(mapper1.get_initial_state().shape[0], 3)

        mapper2 = mapping.PeakMapper("test", n_peaks=1, atom_names=["N", "H"])
        mapper2.add_atom_group(N=AtomIndex(0), H=AtomIndex(1))
        mapper2.add_atom_group(N=AtomIndex(2), H=AtomIndex(3))
        mapper2.add_atom_group(N=AtomIndex(4), H=AtomIndex(5))

        # mapper2 has 1 peak, but 3 atom_group
        # Initial state should have the size of max of these two,
        # so it should be 3
        self.assertEqual(mapper2.get_initial_state().shape[0], 3)

    def test_extract_value_should_fail_with_map_name_mismatch(self):
        mapper = mapping.PeakMapper("test", n_peaks=10, atom_names=["N", "H"])
        m = mapping.PeakMapping(map_name="fail", peak_id=0, atom_name="H")
        state = mapper.get_initial_state()

        with self.assertRaises(KeyError):
            mapper.extract_value(m, state)

    def test_extract_value_should_give_correct_indices_with_equal_length(self):
        mapper = mapping.PeakMapper("test", n_peaks=3, atom_names=["N", "H"])
        mapper.add_atom_group(N=AtomIndex(0), H=AtomIndex(1))
        mapper.add_atom_group(N=AtomIndex(2), H=AtomIndex(3))
        mapper.add_atom_group(N=AtomIndex(4), H=AtomIndex(5))
        state = mapper.get_initial_state()

        self.assertEqual(
            mapper.extract_value(
                mapping.PeakMapping(map_name="test", peak_id=0, atom_name="N"), state
            ),
            0,
        )
        self.assertEqual(
            mapper.extract_value(
                mapping.PeakMapping(map_name="test", peak_id=0, atom_name="H"), state
            ),
            1,
        )
        self.assertEqual(
            mapper.extract_value(
                mapping.PeakMapping(map_name="test", peak_id=1, atom_name="N"), state
            ),
            2,
        )
        self.assertEqual(
            mapper.extract_value(
                mapping.PeakMapping(map_name="test", peak_id=1, atom_name="H"), state
            ),
            3,
        )
        self.assertEqual(
            mapper.extract_value(
                mapping.PeakMapping(map_name="test", peak_id=2, atom_name="N"), state
            ),
            4,
        )
        self.assertEqual(
            mapper.extract_value(
                mapping.PeakMapping(map_name="test", peak_id=2, atom_name="H"), state
            ),
            5,
        )

    def test_extract_value_should_raise_with_illegal_peak_id(self):
        mapper = mapping.PeakMapper("test", n_peaks=1, atom_names=["N", "H"])
        mapper.add_atom_group(N=AtomIndex(0), H=AtomIndex(1))
        mapper.add_atom_group(N=AtomIndex(2), H=AtomIndex(3))
        mapper.add_atom_group(N=AtomIndex(4), H=AtomIndex(5))
        state = mapper.get_initial_state()

        with self.assertRaises(KeyError):
            mapper.extract_value(
                mapping.PeakMapping(map_name="test", peak_id=1, atom_name="H"), state
            )

    def test_extract_value_should_return_not_mapped(self):
        mapper = mapping.PeakMapper("test", n_peaks=3, atom_names=["N", "H"])
        mapper.add_atom_group(N=AtomIndex(0), H=AtomIndex(1))
        state = mapper.get_initial_state()

        val1 = mapper.extract_value(
            mapping.PeakMapping(map_name="test", peak_id=0, atom_name="N"), state
        )
        val2 = mapper.extract_value(
            mapping.PeakMapping(map_name="test", peak_id=0, atom_name="H"), state
        )
        val3 = mapper.extract_value(
            mapping.PeakMapping(map_name="test", peak_id=1, atom_name="N"), state
        )

        val4 = mapper.extract_value(
            mapping.PeakMapping(map_name="test", peak_id=1, atom_name="H"), state
        )
        val5 = mapper.extract_value(
            mapping.PeakMapping(map_name="test", peak_id=2, atom_name="N"), state
        )

        val6 = mapper.extract_value(
            mapping.PeakMapping(map_name="test", peak_id=2, atom_name="H"), state
        )

        # We have 3 peaks, but only one atom group, so only one peak will
        # map to something. With the default initialization, this will be
        # the first peak.
        self.assertEqual(val1, 0)
        self.assertEqual(val2, 1)
        self.assertIsInstance(val3, mapping.NotMapped)
        self.assertIsInstance(val4, mapping.NotMapped)
        self.assertIsInstance(val5, mapping.NotMapped)
        self.assertIsInstance(val6, mapping.NotMapped)

    def test_sample_has_correct_shape(self):
        mapper = mapping.PeakMapper(
            "test", n_peaks=5, atom_names=["N", "H"], mc_perms=5
        )
        state = mapper.get_initial_state()

        trial_states = mapper.sample_permutations(state)

        self.assertEqual(len(trial_states), 120)  # 5! = 120 permutations
        self.assertEqual(trial_states[0].shape[0], 5)  # 5 peaks


class TestPeakMapManager(unittest.TestCase):
    def test_should_raise_if_duplicate_name(self):
        manager = mapping.PeakMapManager()
        with self.assertRaises(ValueError):
            manager.add_map("test", 100, ["N", "H"])
            manager.add_map("test", 100, ["N", "H"])

    def test_get_mapping_from_multiple_mappers(self):
        manager = mapping.PeakMapManager()
        map1 = manager.add_map("test1", 100, ["N", "H"])
        map2 = manager.add_map("test2", 100, ["N", "H"])

        m1 = map1.get_mapping(peak_id=0, atom_name="H")
        m2 = map2.get_mapping(peak_id=1, atom_name="N")

        self.assertEqual(m1.map_name, "test1")
        self.assertEqual(m2.map_name, "test2")
        self.assertEqual(m1.peak_id, 0)
        self.assertEqual(m2.peak_id, 1)
        self.assertEqual(m1.atom_name, "H")
        self.assertEqual(m2.atom_name, "N")

    def test_initial_state_should_have_correct_size(self):
        manager = mapping.PeakMapManager()

        # map1 has initial state of size 3
        map1 = manager.add_map("test1", 3, ["h"])
        map1.add_atom_group(h=AtomIndex(0))

        # map2 has initial state of size 2
        map1 = manager.add_map("test2", 1, ["h"])
        map1.add_atom_group(h=AtomIndex(0))
        map1.add_atom_group(h=AtomIndex(1))

        # initial state should have size 5
        init = manager.get_initial_state()
        self.assertEqual(init.shape[0], 5)

    def test_initial_state_should_have_correct_order(self):
        manager = mapping.PeakMapManager()

        # map1 has initial state of size 3
        map1 = manager.add_map("test1", 3, ["h"])
        map1.add_atom_group(h=AtomIndex(0))

        # map2 has initial state of size 2
        map1 = manager.add_map("test2", 1, ["h"])
        map1.add_atom_group(h=AtomIndex(0))
        map1.add_atom_group(h=AtomIndex(1))

        # should be map1, then map2
        # so 0, 1, 2, 0, 1
        init = manager.get_initial_state()
        self.assertEqual(init[0], 0)
        self.assertEqual(init[1], 1)
        self.assertEqual(init[2], 2)
        self.assertEqual(init[3], 0)
        self.assertEqual(init[4], 1)

    def test_initial_state_should_handle_case_when_empty(self):
        manager = mapping.PeakMapManager()

        state = manager.get_initial_state()

        self.assertEqual(state.shape[0], 0)

    def test_extract_value_should_return_correct_values(self):
        manager = mapping.PeakMapManager()

        # map1 has 2 pekas, 2 atom_groups
        map1 = manager.add_map("map1", 2, atom_names=["N", "H"])
        map1.add_atom_group(N=AtomIndex(0), H=AtomIndex(1))
        map1.add_atom_group(N=AtomIndex(2), H=AtomIndex(3))

        # map2 has 2 peaks, but only 1 atom_group -> 1 peak will be NotMapped
        map2 = manager.add_map("map2", 2, atom_names=["N", "H"])
        map2.add_atom_group(N=AtomIndex(4), H=AtomIndex(5))

        state = manager.get_initial_state()

        ind1 = manager.extract_value(mapping.PeakMapping("map1", 0, "N"), state)
        ind2 = manager.extract_value(mapping.PeakMapping("map1", 0, "H"), state)
        ind3 = manager.extract_value(mapping.PeakMapping("map1", 1, "N"), state)
        ind4 = manager.extract_value(mapping.PeakMapping("map1", 1, "H"), state)
        ind5 = manager.extract_value(mapping.PeakMapping("map2", 0, "N"), state)
        ind6 = manager.extract_value(mapping.PeakMapping("map2", 0, "H"), state)
        ind7 = manager.extract_value(mapping.PeakMapping("map2", 1, "N"), state)
        ind8 = manager.extract_value(mapping.PeakMapping("map2", 1, "H"), state)

        self.assertEqual(ind1, 0)
        self.assertEqual(ind2, 1)
        self.assertEqual(ind3, 2)
        self.assertEqual(ind4, 3)
        self.assertEqual(ind5, 4)
        self.assertEqual(ind6, 5)
        self.assertIsInstance(ind7, mapping.NotMapped)
        self.assertIsInstance(ind8, mapping.NotMapped)

    def test_sample_has_correct_shape(self):
        manager = mapping.PeakMapManager()
        manager.add_map("map1", n_peaks=5, atom_names=["N", "H"], mc_perms=5)
        manager.add_map("map2", n_peaks=5, atom_names=["N", "H"], mc_perms=5)

        state = manager.get_initial_state()

        trial_state = manager.sample_permutations(state)

        self.assertEqual(len(trial_state), 120)  # 5! = 120 permutations
        self.assertEqual(trial_state[0].shape[0], 10)  # 5 + 5 peaks
