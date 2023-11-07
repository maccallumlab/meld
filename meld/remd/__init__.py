#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Module for running replica exchange

The following classes are most useful to users:

- :class:`NearestNeighborLadder` defnes a ladder with exchanges between nearest neighbours.
- :class:`LeaderReplicaExchangeRunner` coordinates a replica exchange run
- The following are useful for replica exchange adaptation

  - :class:`AdaptationPolicy`
  - :class:`NullAdaptor`
  - :class:`EqualAcceptanceAdaptor`
  - :class:`SwitchingCompositeAdaptor`
"""
