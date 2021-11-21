#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module to handle sampling the mappings of peaks to atom indices
"""

from meld.system import indexing

import numpy as np  # type: ignore
import random
from typing import List, Dict, NamedTuple, Union, Tuple


class PeakMapping(NamedTuple):
    """
    A mapping from a peak to an atom
    """

    map_name: str
    peak_id: int
    atom_name: str


class NotMapped:
    """
    Represents a peak that isn't mapped to any atom
    """

    pass


class PeakMapper:
    name: str
    n_peaks: int
    atom_names: List[str]
    atom_groups: List[Dict[str, int]]
    _frozen: bool

    def __init__(self, name: str, n_peaks: int, atom_names: List[str]):
        if n_peaks <= 0:
            raise ValueError("n_peaks must be > 0")
        self.name = name
        self.n_peaks = n_peaks
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
        # The initial state is the longer of n_peaks and n_atom_groups
        # with the state just assigned in order.
        size = max(self.n_peaks, self.n_atom_groups)
        return np.arange(size)

    def extract_value(
        self, mapping: PeakMapping, state: np.ndarray
    ) -> Union[int, NotMapped]:
        # Freeze so we can't add more atom_groups
        self.frozen = True

        if mapping.map_name != self.name:
            raise KeyError(f"Map name {mapping.map_name} does not match {self.name}.")

        peak_id = mapping.peak_id
        if peak_id < 0:
            raise KeyError("peak_id must be >= 0.")
        if peak_id >= self.n_peaks:
            raise KeyError(f"peak_id must be < {self.n_peaks}")

        group_index = state[mapping.peak_id]
        # If we have more peaks than atom_groups, some of the peaks will
        # not be mapped to anything.
        if group_index >= self.n_atom_groups:
            return NotMapped()
        else:
            return self.atom_groups[group_index][mapping.atom_name]

    def sample(self, state: np.ndarray) -> np.ndarray:
        trial_state = state.copy()

        indices = list(range(trial_state.shape[0]))
        i, j = random.sample(indices, k=2)
        trial_state[[i, j]] = trial_state[[j, i]]

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

    def add_map(self, name: str, n_peaks: int, atom_names: List[str]) -> PeakMapper:
        # don't allow duplicates
        if name in self.mappers:
            raise ValueError(f"Trying to insert duplicate entry for {name}.")

        mapper = PeakMapper(name, n_peaks, atom_names)
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

    def extract_value(
        self, mapping: PeakMapping, state: np.ndarray
    ) -> Union[int, NotMapped]:
        if self._name_to_range is None:
            self._setup_name_to_range()

        range_ = self._name_to_range[mapping.map_name]
        sub_state = state[range_[0] : range_[1]]
        return self.mappers[mapping.map_name].extract_value(mapping, sub_state)

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
