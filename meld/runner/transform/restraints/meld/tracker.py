from meld import interfaces
from meld import system
import meld
from meld.system import restraints, scalers
from meld.system import param_sampling
from meld.system import mapping
from meld.system import density
import numpy as np  # typing: ignore
import collections
from typing import List, Tuple, Optional, Union, Set, Dict, DefaultDict
###
import logging

logger = logging.getLogger(__name__)
###
class RestraintTracker:
    """
    A data structure to keep track of restraints, groups, and collections.

    For restraints, we keep track of the dependence on scalers,
    ramps, positioners, and peak mappings. We only update a
    restraint when those dependencies have changed.

    For groups and collections, we keep track of which ones
    depend on parameter sampling. Only those ones are updated
    each step.
    """

    param_manager: param_sampling.ParameterManager
    peak_mapper: mapping.PeakMapManager
    rdc_restraints: List[restraints.RdcRestraint]
    distance_restraints: List[restraints.DistanceRestraint]
    hyperbolic_distance_restraints: List[restraints.HyperbolicDistanceRestraint]
    torsion_restraints: List[restraints.TorsionRestraint]
    dist_prof_restraints: List[restraints.DistProfileRestraint]
    torsion_profile_restraints: List[restraints.TorsProfileRestraint]
    gmm_restraints: List[restraints.GMMDistanceRestraint]
    groups_with_dep: List[Tuple[restraints.RestraintGroup, int]]
    collections_with_dep: List[Tuple[restraints.SelectivelyActiveCollection, int]]
    scaler_map: DefaultDict[restraints.RestraintScaler, List[Tuple[str, int]]]
    ramp_map: DefaultDict[restraints.TimeRamp, List[Tuple[str, int]]]
    positioner_map: DefaultDict[restraints.Positioner, List[Tuple[str, int]]]
    peak_mapping_map: DefaultDict[int, List[Tuple[str, int]]]
    scaler_values: Dict[restraints.RestraintScaler, float]
    ramp_values: Dict[restraints.TimeRamp, float]
    positioner_values: Dict[restraints.Positioner, float]
    peak_mapping_values: Optional[np.ndarray]
    need_update: Set[Tuple[str, int]]
    densities: List[density.DensityMap]
    density_restraints: List[restraints.DensityRestraint]
    scaler_density_map: DefaultDict[
        restraints.BlurScaler, List[Tuple[int, density.DensityMap]]
    ]
    scaler_density_values: Dict[restraints.BlurScaler, float]

    def __init__(
        self,
        param_manager: param_sampling.ParameterManager,
        peak_mapper: mapping.PeakMapManager,
    ):
        self.param_manager = param_manager
        self.peak_mapper = peak_mapper

        # These hold lists of meld restraints in the order that they were added
        # to the system.
        self.distance_restraints = []
        self.rdc_restraints = []
        self.hyperbolic_distance_restraints = []
        self.torsion_restraints = []
        self.dist_prof_restraints = []
        self.torsion_profile_restraints = []
        self.gmm_restraints = []
        self.densities = []
        self.density_restraints = []

        self.groups_with_dep = []
        self.collections_with_dep = []

        # These map from scalers, ramps, etc to the restraints that depend on them.
        self.scaler_map = collections.defaultdict(list)
        self.ramp_map = collections.defaultdict(list)
        self.positioner_map = collections.defaultdict(list)
        self.peak_mapping_map = collections.defaultdict(list)
        self.scaler_density_map = collections.defaultdict(list)

        # These maintain the previous values for these quantities
        self.scaler_values = {}
        self.ramp_values = {}
        self.positioner_values = {}
        self.peak_mapping_values = None
        self.scaler_density_values = {}

        # We maintain a set of restraints that need to be updated.
        self.need_update = set()

    def update(self, alpha: float, timestep: int, state: interfaces.IState):
        self._update_scalers(alpha)
        self._update_ramps(timestep)
        self._update_positioners(alpha)
        self._update_peak_mappings(state)

    def get_and_reset_need_update(self) -> Set[Tuple[str, int]]:
        need_update = self.need_update
        self.need_update = set()
        return need_update

    def density_to_update(self, alpha):
        to_update = []
        for scaler in self.scaler_density_values:
            old_value = self.scaler_density_values[scaler]
            new_value = scaler(alpha) 
            if new_value != old_value or alpha==0.0:
                to_update.extend(self.scaler_density_map[scaler])
                self.scaler_density_values[scaler] = new_value
        return to_update

    def _update_scalers(self, alpha: float):
        for scaler in self.scaler_values:
            old_value = self.scaler_values[scaler]
            new_value = scaler(alpha)
            if new_value != old_value:
                for category, index in self.scaler_map[scaler]:
                    self.need_update.add((category, index))
                self.scaler_values[scaler] = new_value

    def _update_ramps(self, timestep: int):
        for ramp in self.ramp_values:
            old_value = self.ramp_values[ramp]
            new_value = ramp(timestep)
            if new_value != old_value:
                for category, index in self.ramp_map[ramp]:
                    self.need_update.add((category, index))
                self.ramp_values[ramp] = new_value

    def _update_positioners(self, alpha: float):
        for positioner in self.positioner_values:
            old_value = self.positioner_values[positioner]
            new_value = positioner(alpha)
            if new_value != old_value:
                for category, index in self.positioner_map[positioner]:
                    self.need_update.add((category, index))
                self.positioner_values[positioner] = new_value

    def _update_peak_mappings(self, state: interfaces.IState):
        changes = np.argwhere(self.peak_mapping_values != state.mappings)
        for global_peak_index in changes:
            global_peak_index = global_peak_index[0]
            for category, index in self.peak_mapping_map[global_peak_index]:
                self.need_update.add((category, index))
            if self.peak_mapping_values is not None:
                self.peak_mapping_values[global_peak_index] = state.mappings[
                    global_peak_index
            ]

    def add_rdc_restraint(
        self,
        rest: restraints.RdcRestraint,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ):
        assert isinstance(rest, restraints.RdcRestraint)
        self.rdc_restraints.append(rest)
        index = len(self.rdc_restraints) - 1
        self.need_update.add(("rdc", index))

        self._add_scaler_dependency(rest.scaler, "rdc", index, alpha)
        self._add_ramp_dependency(rest.ramp, "rdc", index, timestep)
        self._add_peak_mapping_dependency(rest.atom_index_1, "rdc", index, state)
        self._add_peak_mapping_dependency(rest.atom_index_2, "rdc", index, state)

    def add_density(self, index: int, density: density.DensityMap, alpha: float):
        self.densities.append(density)
        self._add_scaler_density_dependency(index, density,alpha)


    def add_density_restraint(
        self,
        rest: restraints.DensityRestraint,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ):
        assert isinstance(rest, restraints.DensityRestraint)
        self.density_restraints.append(rest)
        index = len(self.density_restraints) - 1
        self.need_update.add(("density",index))
        self._add_scaler_dependency(rest.scaler, "density", index, alpha)
        self._add_ramp_dependency(rest.ramp, "density", index, timestep)

        
    def add_distance_restraint(
        self,
        rest: restraints.DistanceRestraint,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ):
        assert isinstance(rest, restraints.DistanceRestraint)

        self.distance_restraints.append(rest)
        index = len(self.distance_restraints) - 1
        self.need_update.add(("distance", index))

        self._add_scaler_dependency(rest.scaler, "distance", index, alpha)
        self._add_ramp_dependency(rest.ramp, "distance", index, timestep)
        self._add_positioner_dependency(rest.r1, "distance", index, alpha)
        self._add_positioner_dependency(rest.r2, "distance", index, alpha)
        self._add_positioner_dependency(rest.r3, "distance", index, alpha)
        self._add_positioner_dependency(rest.r4, "distance", index, alpha)
        self._add_peak_mapping_dependency(rest.atom_index_1, "distance", index, state)
        self._add_peak_mapping_dependency(rest.atom_index_2, "distance", index, state)

    def add_hyperbolic_distance_restraint(
        self,
        rest: restraints.HyperbolicDistanceRestraint,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ):
        assert isinstance(rest, restraints.HyperbolicDistanceRestraint)

        self.hyperbolic_distance_restraints.append(rest)
        index = len(self.hyperbolic_distance_restraints) - 1
        self.need_update.add(("hyperbolic_distance", index))

        self._add_scaler_dependency(rest.scaler, "hyperbolic_distance", index, alpha)
        self._add_ramp_dependency(rest.ramp, "hyperbolic_distance", index, timestep)

    def add_torsion_restraint(
        self,
        rest: restraints.TorsionRestraint,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ):
        assert isinstance(rest, restraints.TorsionRestraint)

        self.torsion_restraints.append(rest)
        index = len(self.torsion_restraints) - 1
        self.need_update.add(("torsion", index))

        self._add_scaler_dependency(rest.scaler, "torsion", index, alpha)
        self._add_ramp_dependency(rest.ramp, "torsion", index, timestep)

    def add_distance_profile_restraint(
        self,
        rest: restraints.DistProfileRestraint,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ):
        assert isinstance(rest, restraints.DistProfileRestraint)

        self.dist_prof_restraints.append(rest)
        index = len(self.dist_prof_restraints) - 1
        self.need_update.add(("dist_profile", index))

        self._add_scaler_dependency(rest.scaler, "dist_profile", index, alpha)
        self._add_ramp_dependency(rest.ramp, "dist_profile", index, timestep)

    def add_torsion_profile_restraint(
        self,
        rest: restraints.TorsProfileRestraint,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ):
        assert isinstance(rest, restraints.TorsProfileRestraint)

        self.torsion_profile_restraints.append(rest)
        index = len(self.torsion_profile_restraints) - 1
        self.need_update.add(("tors_profile", index))

        self._add_scaler_dependency(rest.scaler, "tors_profile", index, alpha)
        self._add_ramp_dependency(rest.ramp, "tors_profile", index, timestep)

    def add_gmm_distance_restraint(
        self,
        rest: restraints.GMMDistanceRestraint,
        alpha: float,
        timestep: int,
        state: interfaces.IState,
    ):
        assert isinstance(rest, restraints.GMMDistanceRestraint)

        self.gmm_restraints.append(rest)
        index = len(self.gmm_restraints) - 1
        self.need_update.add(("gmm", index))

        self._add_scaler_dependency(rest.scaler, "gmm", index, alpha)
        self._add_ramp_dependency(rest.ramp, "gmm", index, timestep)

    def _add_scaler_dependency(
        self,
        scaler: restraints.RestraintScaler,
        category: str,
        index: int,
        alpha: float,
    ):
        if not isinstance(scaler, restraints.ConstantScaler):
            self.scaler_map[scaler].append((category, index))
            if scaler not in self.scaler_values:
                self.scaler_values[scaler] = scaler(alpha)
            else:
                assert scaler(alpha) == self.scaler_values[scaler]

    def _add_ramp_dependency(
        self, ramp: restraints.TimeRamp, category: str, index: int, timestep: int
    ):
        if not isinstance(ramp, restraints.ConstantRamp):
            self.ramp_map[ramp].append((category, index))
            if ramp not in self.ramp_values:
                self.ramp_values[ramp] = ramp(timestep)
            else:
                assert ramp(timestep) == self.ramp_values[ramp]

    def _add_positioner_dependency(
        self, positioner: restraints.Positioner, category: str, index: int, alpha: float
    ):
        if not isinstance(positioner, restraints.ConstantPositioner):
            self.positioner_map[positioner].append((category, index))
            if positioner not in self.positioner_values:
                self.positioner_values[positioner] = positioner(alpha)
            else:
                assert positioner(alpha) == self.positioner_values[positioner]

    def _add_peak_mapping_dependency(
        self,
        peak_mapping: Union[int, mapping.PeakMapping],
        category: str,
        index: int,
        state: interfaces.IState,
    ):
        if isinstance(peak_mapping, mapping.PeakMapping):
            global_peak_index = self.peak_mapper.get_index(peak_mapping)
            self.peak_mapping_map[global_peak_index].append((category, index))

            if self.peak_mapping_values is None:
                self.peak_mapping_values = state.mappings.copy()
            else:
                assert (state.mappings == self.peak_mapping_values).all()
   
    def _add_scaler_density_dependency(
        self, index: int, density: density.DensityMap, alpha: float
    ):
        if not isinstance(density.blur_scaler, scalers.ConstantBlurScaler):
            self.scaler_density_map[density.blur_scaler].append((index,density))
            if density.blur_scaler not in self.scaler_density_values:
                self.scaler_density_values[density.blur_scaler] = density.blur_scaler(alpha)
            else:
                assert density.blur_scaler(alpha) == self.scaler_values[density.blur_scaler]
