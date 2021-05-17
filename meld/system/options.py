"""
Module to handle options for a MELD run
"""

from .temperature import REST2Scaler
from .montecarlo import MonteCarloScheduler
from .patchers import RdcAlignmentPatcher
from typing import Optional
from simtk.unit import atmosphere  # type: ignore


class RunOptions:
    """
    Class to store options for MELD run
    """

    def __setattr__(self, name, value):
        # open we only allow setting of these attributes
        # all others will raise an error, which catches
        # typos
        _allowed_attributes = [
            "remove_com",
            "runner",
            "timesteps",
            "minimize_steps",
            "implicit_solvent_model",
            "cutoff",
            "use_big_timestep",
            "use_bigger_timestep",
            "use_amap",
            "amap_alpha_bias",
            "amap_beta_bias",
            "min_mc",
            "run_mc",
            "solvation",
            "enable_pme",
            "enable_pressure_coupling",
            "pressure",
            "pressure_coupling_update_steps",
            "pme_tolerance",
            "use_rest2",
            "rest2_scaler",
            "soluteDielectric",
            "solventDielectric",
            "implicitSolventSaltConc",
            "rdc_patcher",
        ]
        _allowed_attributes += ["_{}".format(item) for item in _allowed_attributes]
        if name not in _allowed_attributes:
            raise ValueError(f"Attempted to set unknown attribute {name}")
        else:
            object.__setattr__(self, name, value)

    def __init__(self, solvation="implicit"):
        """
        Initialize RunOptions

        Args:
            solvation: solvation model [implicit, explicit, vacuum]
        """
        self._solvation = solvation
        if solvation == "implicit":
            self.implicit_solvent_model = "gbNeck2"
            self.cutoff = None
            self.enable_pme = False
            self.enable_pressure_coupling = False
        elif solvation == "explicit":
            self.implicit_solvent_model = "vacuum"
            self.cutoff = 0.9
            self.enable_pme = True
            self.enable_pressure_coupling = True
        else:
            raise RuntimeError(f"Unknown value {solvation} for solvation")
        self._runner = "openmm"
        self._timesteps = 5000
        self._minimize_steps = 1000
        self._use_big_timestep = False
        self._use_bigger_timestep = False
        self._use_amap = False
        self._amap_alpha_bias = 1.0
        self._amap_beta_bias = 1.0
        self._min_mc = None
        self._run_mc = None
        self._remove_com = True
        self._pressure = 1.0 * atmosphere
        self._pressure_coupling_update_steps = 25
        self._pme_tolerance = 0.0005
        self._use_rest2 = False
        self._rest2_scaler = None
        self._implicitSolventSaltConc = None
        self._solventDielectric = None
        self._soluteDielectric = None
        self._rdc_patcher = None

    # solvation is a read-only property that must be set
    # when the options are created
    @property
    def solvation(self) -> str:
        """
        Solvation model. One of [implicit, explicit, vacuum]

        .. note::
           This property is read-only
        """
        return self._solvation

    @property
    def enable_pme(self) -> bool:
        """
        Enable PME electrostatics
        """
        return self._enable_pme

    @enable_pme.setter
    def enable_pme(self, new_value: bool) -> None:
        if new_value not in [True, False]:
            raise ValueError("enable_pme must be True or False")
        if new_value:
            if self._solvation == "implicit":
                raise ValueError("Tried to set enable_pme=True with implicit solvation")
        self._enable_pme = new_value

    @property
    def pme_tolerance(self) -> float:
        """
        PME tolerance
        """
        return self._pme_tolerance

    @pme_tolerance.setter
    def pme_tolerance(self, new_value: float) -> None:
        if new_value <= 0:
            raise ValueError("pme_tolerance must be > 0")
        self._pme_tolerance = new_value

    @property
    def enable_pressure_coupling(self) -> bool:
        """
        Enable pressure coupling
        """
        return self._enable_pressure_coupling

    @enable_pressure_coupling.setter
    def enable_pressure_coupling(self, new_value: bool) -> None:
        if new_value not in [True, False]:
            raise ValueError("enable_pressure_coupling must be True or False")
        if new_value:
            if self._solvation == "implicit":
                raise ValueError(
                    "Tried to set enable_pressure_coupling=True with "
                    "implicit solvation"
                )
        self._enable_pressure_coupling = new_value

    @property
    def pressure(self) -> float:
        """
        Target pressure
        """
        return self._pressure

    @pressure.setter
    def pressure(self, new_value: float) -> None:
        if new_value <= 0:
            raise ValueError("pressure must be > 0")
        self._pressure = new_value

    @property
    def pressure_coupling_update_steps(self) -> int:
        """
        Steps between pressure coupling
        """
        return self._pressure_coupling_update_steps

    @pressure_coupling_update_steps.setter
    def pressure_coupling_update_steps(self, new_value: int) -> None:
        if new_value <= 0:
            raise ValueError("pressure_coupling_update_steps must be > 0")
        self._pressure_coupling_update_steps = new_value

    @property
    def use_rest2(self) -> bool:
        """
        User REST2 for temperature scaling
        """
        return self._use_rest2

    @use_rest2.setter
    def use_rest2(self, new_value: bool) -> None:
        self._use_rest2 = new_value

    @property
    def rest2_scaler(self) -> REST2Scaler:
        """
        Scaler for REST2 temperature scaling
        """
        return self._rest2_scaler

    @rest2_scaler.setter
    def rest2_scaler(self, new_value: REST2Scaler) -> None:
        self._rest2_scaler = new_value

    @property
    def min_mc(self) -> MonteCarloScheduler:
        """
        MonteCarlo Scheduler used during minimization
        """
        return self._min_mc

    @min_mc.setter
    def min_mc(self, new_value: MonteCarloScheduler) -> None:
        self._min_mc = new_value

    @property
    def run_mc(self) -> MonteCarloScheduler:
        """
        MonteCarloScheduler used during run
        """
        return self._run_mc

    @run_mc.setter
    def run_mc(self, new_value: MonteCarloScheduler) -> None:
        self._run_mc = new_value

    @property
    def remove_com(self) -> bool:
        """
        Remove COM motion every step
        """
        return self._remove_com

    @remove_com.setter
    def remove_com(self, new_value: bool) -> None:
        self._remove_com = bool(new_value)

    @property
    def runner(self) -> str:
        """
        ReplicaRunner to use during run. One of [openmm, fake_runner]
        """
        return self._runner

    @runner.setter
    def runner(self, value: str) -> None:
        if value not in ["openmm", "fake_runner"]:
            raise RuntimeError(f"unknown value for runner {value}")
        self._runner = value

    @property
    def implicitSolventSaltConc(self) -> float:
        """
        Implicit solvent salt concentration
        """
        return self._implicitSolventSaltConc

    @implicitSolventSaltConc.setter
    def implicitSolventSaltConc(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise RuntimeError("implicitSolventSaltConc must be > 0")
        self._implicitSolventSaltConc = value

    @property
    def solventDielectric(self) -> float:
        """
        Implicit solvent dielectric
        """
        return self._solventDielectric

    @solventDielectric.setter
    def solventDielectric(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise RuntimeError("solventDielectric must be > 0")
        self._solventDielectric = value

    @property
    def soluteDielectric(self) -> float:
        """
        Implicit solut dielectric
        """
        return self._soluteDielectric

    @soluteDielectric.setter
    def soluteDielectric(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise RuntimeError("soluteDielectric must be > 0")
        self._soluteDielectric = value

    @property
    def timesteps(self) -> int:
        """
        Number of timesteps per stage
        """
        return self._timesteps

    @timesteps.setter
    def timesteps(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            raise RuntimeError("timesteps must be > 0")
        self._timesteps = value

    @property
    def minimize_steps(self) -> int:
        """
        Number of minimization steps at start of calculation
        """
        return self._minimize_steps

    @minimize_steps.setter
    def minimize_steps(self, value: int) -> None:
        value = int(value)
        if value <= 0:
            raise RuntimeError("minimize_steps must be > 0")
        self._minimize_steps = value

    @property
    def implicit_solvent_model(self) -> str:
        """
        Implict solvent model. One of [None, obc, gbNeck, gbNeck2, vacuum]
        """
        return self._implicit_solvent_model

    @implicit_solvent_model.setter
    def implicit_solvent_model(self, value: str) -> None:
        if value not in [None, "obc", "gbNeck", "gbNeck2", "vacuum"]:
            raise RuntimeError(f"unknown value for implicit solvent model {value}")
        self._implicit_solvent_model = value

    @property
    def cutoff(self) -> Optional[float]:
        """
        Nonbonded cutoff distance in nm
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: Optional[float]) -> None:
        if value is None:
            self._cutoff = None
        else:
            value = float(value)
            if value <= 0:
                raise RuntimeError("cutoff must be > 0")
            self._cutoff = value

    @property
    def use_big_timestep(self) -> bool:
        """
        Flag to use big timesteps

        This sets hydrogen masses to 3 and the timestep to 3.5 fs (default is 2 fs).
        """
        return self._use_big_timestep

    @use_big_timestep.setter
    def use_big_timestep(self, value: bool) -> None:
        self._use_big_timestep = bool(value)

    @property
    def use_bigger_timestep(self) -> bool:
        """
        Flag to use bigger timesteps

        This sets hydrogen masses to 4 and the timestep to 4.0 fs (default is 2 fs).
        """
        return self._use_bigger_timestep

    @use_bigger_timestep.setter
    def use_bigger_timestep(self, value: bool) -> None:
        self._use_bigger_timestep = bool(value)

    @property
    def use_amap(self) -> bool:
        """
        Use AMAP torsion correction

        .. warning::
           Not compatible with explicit solvent calculations
        """
        return self._use_amap

    @use_amap.setter
    def use_amap(self, value: bool) -> None:
        if not value in [True, False]:
            raise ValueError("use_amap must be True or False")
        if value and self._solvation == "explicit":
            raise ValueError("use_amap can not be set with explicit solvent")
        self._use_amap = bool(value)

    @property
    def amap_alpha_bias(self) -> float:
        """
        Strength of amap alpha bias
        """
        return self._amap_alpha_bias

    @amap_alpha_bias.setter
    def amap_alpha_bias(self, value: float) -> None:
        if value < 0:
            raise RuntimeError("amap_alpha_bias < 0")
        self._amap_alpha_bias = value

    @property
    def amap_beta_bias(self) -> float:
        """
        Strength of amap beta bias
        """
        return self._amap_beta_bias

    @amap_beta_bias.setter
    def amap_beta_bias(self, value: float) -> None:
        if value < 0:
            raise RuntimeError("amap_beta_bias < 0")
        self._amap_beta_bias = value

    @property
    def rdc_patcher(self) -> RdcAlignmentPatcher:
        """
        Patcher for RDC alignments
        """
        return self._rdc_patcher

    @rdc_patcher.setter
    def rdc_patcher(self, value: RdcAlignmentPatcher) -> None:
        self._rdc_patcher = value

    def sanity_check(self) -> None:
        """
        Check that options are compatible
        """
        if self._solvation == "implicit":
            if self._enable_pme:
                raise ValueError("enable_pme == True for implicit solvation simulation")
            if self._enable_pressure_coupling:
                raise ValueError(
                    "enable_pressure_coupling == True for implicit"
                    "solvation simulation"
                )

        if self._solvation == "explicit":
            if not self._implicit_solvent_model == "vacuum":
                raise ValueError(
                    'implicit_solvent_model != "vacuum" for explicit '
                    "solvation simulation"
                )
            if self._use_amap == True:
                raise ValueError("use_amap cannot be set with explicit solvent")
