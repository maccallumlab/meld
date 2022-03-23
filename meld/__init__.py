#
# Copyright 2015 by Justin MacCallum, Alberto Perez, and Ken Dill
# All rights reserved
#

import logging

logger = logging.getLogger("meld")
logger.addHandler(logging.NullHandler())

__version__ = "0.6.0"

from meld.comm import MPICommunicator
from meld.vault import DataStore
from meld.parse import (
    get_sequence_from_AA1,
    get_sequence_from_AA3,
    get_secondary_structure_restraints,
    get_rdc_restraints,
)
from meld.remd.ladder import NearestNeighborLadder
from meld.remd.adaptor import AdaptationPolicy, EqualAcceptanceAdaptor
from meld.remd.leader import LeaderReplicaExchangeRunner
from meld.system.options import RunOptions
from meld.system.montecarlo import DoubleTorsionMover, MonteCarloScheduler
from meld.system.builders.amber.subsystem import (
    AmberSubSystemFromSequence,
    AmberSubSystemFromPdbFile,
)
from meld.system.builders.amber.builder import AmberSystemBuilder, AmberOptions
from meld.system.temperature import (
    ConstantTemperatureScaler,
    LinearTemperatureScaler,
    GeometricTemperatureScaler,
    REST2Scaler,
)
from meld.system.indexing import AtomIndex, ResidueIndex
from meld.system.param_sampling import (
    UniformDiscretePrior,
    UniformContinuousPrior,
    ExponentialDiscretePrior,
    ExponentialContinuousPrior,
    ScaledExponentialDiscretePrior,
    DiscreteSampler,
    ContinuousSampler,
)
from meld.system.patchers.rdc_alignment import add_rdc_alignment
from meld.system.patchers.spin_label import add_virtual_spin_label