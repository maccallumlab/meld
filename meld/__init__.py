#
# Copyright 2015 by Justin MacCallum, Alberto Perez, and Ken Dill
# All rights reserved
#

import logging

logger = logging.getLogger("meld")
logger.addHandler(logging.NullHandler())

__version__ = "0.6.1"

from openmm import unit  # type: ignore

from meld.comm import MPICommunicator
from meld.helpers import setup_data_store, setup_replica_exchange
from meld.parse import (
    get_rdc_restraints,
    get_secondary_structure_restraints,
    get_sequence_from_AA1,
    get_sequence_from_AA3,
)
from meld.remd.adaptor import AdaptationPolicy, EqualAcceptanceAdaptor
from meld.remd.ladder import NearestNeighborLadder
from meld.remd.leader import LeaderReplicaExchangeRunner
from meld.system.builders.amber.builder import AmberOptions, AmberSystemBuilder
from meld.system.builders.amber.subsystem import (
    AmberSubSystemFromPdbFile,
    AmberSubSystemFromSequence,
)
from meld.system.builders.build_elastic_network_restraints import (
    add_elastic_network_restraints,
    create_elastic_network_restraints,
)
from meld.system.indexing import AtomIndex, ResidueIndex
from meld.system.montecarlo import DoubleTorsionMover, MonteCarloScheduler
from meld.system.options import RunOptions
from meld.system.param_sampling import (
    ContinuousSampler,
    DiscreteSampler,
    ExponentialContinuousPrior,
    ExponentialDiscretePrior,
    ScaledExponentialDiscretePrior,
    UniformContinuousPrior,
    UniformDiscretePrior,
)
from meld.system.patchers.freeze import freeze_atoms
from meld.system.patchers.potential import remove_potential
from meld.system.patchers.rdc_alignment import add_rdc_alignment
from meld.system.patchers.spin_label import add_virtual_spin_label
from meld.system.temperature import (
    ConstantTemperatureScaler,
    GeometricTemperatureScaler,
    LinearTemperatureScaler,
    REST2Scaler,
)
from meld.vault import DataStore
