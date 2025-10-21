#
# Copyright 2025 by Alberto Perez, Imesh Ranaweera
# All rights reserved
#

"""
Options for building a system with the Grappa force field.
"""

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

from openmm import unit as u  

logger = logging.getLogger(__name__)


@partial(dataclass, frozen=True)
class GrappaOptions:
    solvation_type: str
    grappa_model_tag: str
    base_forcefield_files: List[str] = field(default_factory=lambda: ['amber14-all.xml', 'implicit/gbn2.xml'])
    default_temperature: u.Quantity = field(default_factory=lambda: 300.0 * u.kelvin)
    cutoff: Optional[u.Quantity] = None
    use_big_timestep: bool = False
    use_bigger_timestep: bool = False
    remove_com: bool = False
    amap_beta_bias: float = 1.0
    enable_amap: bool = False
    prebuilt_system: Optional[object] = None  # <-- allow an OpenMM System (or None)

    def __post_init__(self):
        # Unit conversions must happen before validation if validation depends on the converted value type
        if isinstance(self.default_temperature, u.Quantity):
            object.__setattr__(
                self,
                "default_temperature",
                self.default_temperature.value_in_unit(u.kelvin),
            )
        
        if self.cutoff is not None:
            if not isinstance(self.cutoff, u.Quantity): 
                if isinstance(self.cutoff, (float, int)):
                    object.__setattr__(self, "cutoff", float(self.cutoff) * u.nanometer)
                else:
                    raise ValueError("Cutoff must be a float/int (assumed nm) or an OpenMM Quantity with length units.")

            if isinstance(self.cutoff, u.Quantity):
                 object.__setattr__(self, "cutoff", self.cutoff.value_in_unit(u.nanometer))
            elif isinstance(self.cutoff, (float, int)):
                 object.__setattr__(self, "cutoff", float(self.cutoff)) # Store as float in nm
            elif self.cutoff is not None: # Should not be reached if previous conditions are exhaustive
                 raise ValueError("Cutoff must be a float/int (assumed nm), an OpenMM Quantity, or None.")


        # Validations
        if self.solvation_type not in ["implicit", "explicit"]:
            raise ValueError(f"solvation_type must be 'implicit' or 'explicit', got {self.solvation_type}")

        ALLOWED_GRAPPA_TAGS = {
            "grappa-1.3.0": "Covers peptides, small molecules, rna and peptide radicals",
            "grappa-1.4.0": "Covers peptides, small molecules, rna",
            "grappa-1.4.1-radical": "Covers peptides, small molecules, rna and peptide radicals",
            "grappa-1.4.1-light": "Lightweight version with significantly less parameters for testing. Covers peptides, small molecules, rna and peptide radicals"
        }
        if self.grappa_model_tag not in ALLOWED_GRAPPA_TAGS:
            error_msg = f"Invalid grappa_model_tag: '{self.grappa_model_tag}'. Allowed tags are:\n"
            for tag, desc in ALLOWED_GRAPPA_TAGS.items():
                error_msg += f"  {tag}: {desc}\n"
            raise ValueError(error_msg)

        if self.default_temperature < 0:
            raise ValueError("Default_temperature must be >= 0")

        if self.use_big_timestep and self.use_bigger_timestep:
            raise ValueError("Cannot set both use_big_timestep and use_bigger_timestep to True.")

        if not self.base_forcefield_files:
            raise ValueError("Base_forcefield_files cannot be empty.")
        for ff_file in self.base_forcefield_files:
            if not ff_file.endswith(".xml"):
                raise ValueError(f"Force field file {ff_file} should be an XML file.")

        # Logging
        logger.info(f"GrappaOptions initialized with model tag: {self.grappa_model_tag}")
        logger.info(f"Solvation type: {self.solvation_type}")
        logger.info(f"Base force field files: {self.base_forcefield_files}")
        logger.info(f"Default temperature: {self.default_temperature} K")
        logger.info(f"Cutoff: {self.cutoff} nm" if self.cutoff is not None else "Cutoff: None")
        logger.info(f"Use big timestep (3.5fs): {self.use_big_timestep}")
        logger.info(f"Use bigger timestep (4.5fs): {self.use_bigger_timestep}")
        logger.info(f"Remove COM motion: {self.remove_com}")
        if self.prebuilt_system is not None:
            logger.info("A prebuilt OpenMM System was provided in GrappaOptions (prebuilt_system).")
