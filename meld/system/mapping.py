#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to handle sampling the mappings of peaks to atom indices
"""

import random
from typing import Dict, List, NamedTuple, Tuple

import numpy as np  # type: ignore

from meld.system import indexing


class PeakMapping(NamedTuple):
    """
    A mapping from a peak to an atom
    """

    map_name: str
    peak_id: int
    atom_name: str


class PeakMapper:
    name: str
    n_peaks: int
    n_active: int
    atom_names: List[str]
    atom_groups: List[Dict[str, int]]
    _frozen: bool

    def __init__(self, name: str, n_peaks: int, n_active: int, atom_names: List[str]):
        if n_peaks <= 0:
            raise ValueError("n_peaks must be > 0")
        self.name = name
        self.n_peaks = n_peaks
        self.n_active = n_active
        assert n_peaks > 0
        assert n_active > 0
        assert n_active <= n_peaks
        self.atom_names = atom_names
        self.atom_groups = []
        self.frozen = False

    def add_atom_group(self, **kwargs: indexing.AtomIndex):
        if self.frozen:
            raise RuntimeError(
                "Cannot add an atom group after get_initial_state or extract_value have been called."
            )

        for name in self.atom_names:
            if not name in kwargs:
                raise KeyError(f"Expected argument {name} not given.")

        for name in kwargs:
            if not name in self.atom_names:
                raise KeyError(f"Unexpected argument {name}.")

        for name, value in kwargs.items():
            if not isinstance(value, indexing.AtomIndex):
                raise ValueError(
                    f"Values should be AtomIndex, but got {type(value)} for {name}."
                )
        self.atom_groups.append({k: int(v) for k, v in kwargs.items()})

    def get_mapping(self, peak_id: int, atom_name: str) -> PeakMapping:
        if peak_id < 0:
            raise KeyError("peak_id must be >= 0.")
        if peak_id >= self.n_peaks:
            raise KeyError(f"peak_id must be <= {self.n_peaks - 1}.")
        if atom_name not in self.atom_names:
            raise KeyError(f"atom_name={atom_name} not in {self.atom_names}.")

        return PeakMapping(map_name=self.name, peak_id=peak_id, atom_name=atom_name)

    def get_initial_state(self) -> np.ndarray:
        # Freeze so we can't add more atom_groups
        self._frozen = True
        if self.n_active > self.n_atom_groups:
            raise ValueError("n_active must be <= n_atom_groups")
        state = -np.ones(self.n_peaks, dtype=int)
        state[: self.n_active] = np.arange(self.n_active)
        return state

    def sample(self, state: np.ndarray) -> np.ndarray:
        r = random.random()

        # We don't need to do peak reassignment, because all groups
        # will always be assigned to a peak.
        if self.n_active == self.n_atom_groups:
            if r < 0.1:
                return self._sample_peak_swap(state)
            else:
                return self._sample_neighbour_swap(state)
        # We have some groups that are unassigned, so we need to
        # include the peak reassignment step.
        else:
            if r < 0.1:
                return self._sample_peak_swap(state)
            elif r < 0.2:
                return self._sample_peak_reassign(state)
            else:
                return self._sample_neighbour_swap(state)

    def _sample_peak_swap(self, state: np.ndarray) -> np.ndarray:
        trial_state = state.copy()

        # Sample a pair of peaks to swap.
        i, j = random.sample(range(self.n_peaks), k=2)

        # Swap them
        group_i = trial_state[i]
        group_j = trial_state[j]
        trial_state[i] = group_j
        trial_state[j] = group_i

        return trial_state

    def _sample_neighbour_swap(self, state: np.ndarray) -> np.ndarray:
        trial_state = state.copy()

        # Choose two neighbouring residues
        i = random.randrange(self.n_atom_groups - 1)
        j = i + 1

        # Identify the corresponding peaks
        peaks_i = np.argwhere(trial_state == i)
        peaks_j = np.argwhere(trial_state == j)
        peak_i = None if len(peaks_i) == 0 else peaks_i[0]
        peak_j = None if len(peaks_j) == 0 else peaks_j[0]

        # Neither residue is assigned to a peak, so we don't do anything
        if (peak_i is None) and (peak_j is None):
            pass
        # One of the residues is assigned but the other is not.
        elif peak_i is None:
            trial_state[peak_j] = i
        elif peak_j is None:
            trial_state[peak_i] = j
        # Both residues are assigned, so we swap them.
        else:
            trial_state[peak_i] = j
            trial_state[peak_j] = i

        return trial_state

    def _sample_peak_reassign(self, state: np.ndarray) -> np.ndarray:
        trial_state = state.copy()

        # Choose an assigned peak
        assigned_peaks = [peak[0] for peak in np.argwhere(trial_state != -1)]
        peak = random.choice(assigned_peaks)

        # Choose an unassigned atom group
        atom_groups = set(range(self.n_atom_groups))
        assigned_groups = set(trial_state)
        unassigned_groups = list(atom_groups - assigned_groups)
        # Raise an error if we have no unassigned groups, as we shouldn't
        # be calling this function in that case.
        if not unassigned_groups:
            raise RuntimeError(
                "There are no unassigned groups, so _sample_peak_reassign shouldn't be called."
            )
        group = random.choice(unassigned_groups)

        # Swap
        trial_state[peak] = group

        return trial_state

    @property
    def n_atom_groups(self) -> int:
        return len(self.atom_groups)


class PeakMapManager:
    mappers: Dict[str, PeakMapper]
    _name_to_range: Dict[str, Tuple[int, int]]

    def __init__(self):
        self.mappers = {}
        self._name_to_range = None

    def add_map(
        self, name: str, n_peaks: int, n_active: int, atom_names: List[str]
    ) -> PeakMapper:
        # don't allow duplicates
        if name in self.mappers:
            raise ValueError(f"Trying to insert duplicate entry for {name}.")

        mapper = PeakMapper(name, n_peaks, n_active, atom_names)
        self.mappers[name] = mapper

        return mapper

    def get_initial_state(self) -> np.ndarray:
        if self._name_to_range is None:
            self._setup_name_to_range()

        # If we don't have any mappers, just return an empty array.
        if not self.mappers:
            return np.array([], dtype=int)

        # Loop through our mappers in the order they were added and get the
        # initial state.
        states = [mapper.get_initial_state() for mapper in self.mappers.values()]

        # Concatenate them together
        return np.hstack(states)

    def extract_value(self, mapping: PeakMapping, state: np.ndarray) -> int:
        if self._name_to_range is None:
            self._setup_name_to_range()

        range_ = self._name_to_range[mapping.map_name]

        mapper = self.mappers[mapping.map_name]
        mapper.frozen = True

        if mapping.map_name != mapper.name:
            raise KeyError(f"Map name {mapping.map_name} does not match {mapper.name}.")

        peak_id = mapping.peak_id
        if peak_id < 0:
            raise KeyError("peak_id must be >= 0.")
        if peak_id >= mapper.n_peaks:
            raise KeyError(f"peak_id must be < {mapper.n_peaks}")

        group_index = state[mapping.peak_id + range_[0]]

        if group_index == -1:
            return -1
        else:
            return mapper.atom_groups[group_index][mapping.atom_name]

    def sample(self, state: np.ndarray) -> np.ndarray:
        if self._name_to_range is None:
            self._setup_name_to_range()

        sub_states = []
        for name in self.mappers:
            range_ = self._name_to_range[name]
            sub_state = state[range_[0] : range_[1]]
            sub_states.append(sub_state)

        trial_sub_samples = []
        perturbed = random.randrange(0, len(sub_states))
        for i, (mapper, sub_state) in enumerate(zip(self.mappers.values(), sub_states)):
            if i == perturbed:
                trial_sub_sample = mapper.sample(sub_state)
                trial_sub_samples.append(trial_sub_sample)
            else:
                trial_sub_samples.append(sub_state)

        return np.hstack(trial_sub_samples)

    def get_index(self, mapping: PeakMapping) -> int:
        if self._name_to_range is None:
            self._setup_name_to_range()

        range_ = self._name_to_range[mapping.map_name]

        mapper = self.mappers[mapping.map_name]
        mapper.frozen = True

        if mapping.map_name != mapper.name:
            raise KeyError(f"Map name {mapping.map_name} does not match {mapper.name}.")

        peak_id = mapping.peak_id
        return peak_id + range_[0]

    def has_mappers(self) -> bool:
        if self.mappers:
            return True
        else:
            return False

    def _setup_name_to_range(self):
        start = 0
        self._name_to_range = {}
        for name in self.mappers:
            length = self.mappers[name].get_initial_state().shape[0]
            self._name_to_range[name] = (start, start + length)
            start += length
