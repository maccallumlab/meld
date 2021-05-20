#
# Copyright 2015 by Justin MacCallum, Alberto Perez, Ken Dill
# All rights reserved
#

"""
Helper functions for restraint transformers
"""


def _delete_from_always_active(restraints, always_active):
    for restraint in restraints:
        always_active.remove(restraint)
