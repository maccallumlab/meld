#
# Copyright 2023 The MELD Contributors
# All rights reserved
#

"""
Options for building a system with the Grappa force field.
"""

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

from openmm import unit as u  # type: ignore

logger = logging.getLogger(__name__)


@partial(dataclass, frozen=True)
class GrappaOptions:
    grappa_model_tag: str = "latest"
    base_forcefield_files: List[str] = field(default_factory=lambda: ["amber99sbildn.xml", "tip3p.xml"])
    default_temperature: u.Quantity = field(default_factory=lambda: 300.0 * u.kelvin)
    cutoff: Optional[float] = None  # in nanometers
    use_big_timestep: bool = False  # 3fs
    use_bigger_timestep: bool = False  # 4fs
    remove_com: bool = True

    def __post_init__(self):
        if isinstance(self.default_temperature, u.Quantity):
            # Use object.__setattr__ because the class is frozen
            object.__setattr__(
                self,
                "default_temperature",
                self.default_temperature.value_in_unit(u.kelvin),
            )
        if self.default_temperature < 0:
            raise ValueError("default_temperature must be >= 0")

        if self.use_big_timestep and self.use_bigger_timestep:
            raise ValueError("Cannot set both use_big_timestep and use_bigger_timestep to True.")

        if not self.grappa_model_tag:
            raise ValueError("grappa_model_tag cannot be empty.")

        if not self.base_forcefield_files:
            raise ValueError("base_forcefield_files cannot be empty.")
        for ff_file in self.base_forcefield_files:
            if not ff_file.endswith(".xml"):
                raise ValueError(f"Force field file {ff_file} should be an XML file.")

        logger.info(f"GrappaOptions initialized with model tag: {self.grappa_model_tag}")
        logger.info(f"Base force field files: {self.base_forcefield_files}")
        logger.info(f"Default temperature: {self.default_temperature} K")
        logger.info(f"Cutoff: {self.cutoff} nm")
        logger.info(f"Use big timestep (3fs): {self.use_big_timestep}")
        logger.info(f"Use bigger timestep (4fs): {self.use_bigger_timestep}")
        logger.info(f"Remove COM motion: {self.remove_com}")
