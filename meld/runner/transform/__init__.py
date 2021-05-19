"""
Transformers are objects that take an openmm system
as created by loading an amber topology and modify
it in various ways. Example transformations include:
- Adding extra forces for restraints
- Modifying an existing force, like REST2
- Replacing an existing force, like softcore interactions

One thing transformers should not do is add particles to
the system. Instead, particles can be added using Patcher
objects.

The transformers are chained together, so that at
the end, all necessary modifications have been made
and the system can be simulated with OpenMM. To
add a new transformer, you must first create a new
transformer class that implements the transformer
protocol given in `TransformerBase`. Next, you
must add this class to the list in
`OpenMMRunner._setup_transformers`.

"""

from meld.system import options
from meld.system import restraints
from simtk import openmm as mm  # type: ignore
from simtk.openmm import app  # type: ignore

from typing import List


class TransformerBase:
    """
    Base class to document how transformers work.

    The transformation process proceeds in several stages.
    - The transformer is initialized with the run options
    and lists of restraints. The transformer must store
    any relevant information, as these parameters will
    not be given again. The `__init__` method must also
    delete any restraints that are handled by this
    transformer from `always_active_restraints` and
    `selectively_active_restraints`.
    - Next, `add_interactions` is called. This provides
    an opportunity for the transformer to add new
    forces, or to modify or replace existing forces.
    - Next `finalize` is called. All changes to particles
    and forces are completed before this call, which
    can be useful when we have to, e.g. store
    parameters.
    - Finally, `update` is called every stage. This provides
    an opportunity to update parameters like force constants
    depending on alpha or on the time step.

    The number of particles cannot be changed by a transformer,
    instead additional particles should be added using a
    patcher.
    """

    def __init__(
        self,
        options: options.RunOptions,
        always_active_restraints: List[restraints.Restraint],
        selectively_active_restraints: List[restraints.SelectivelyActiveCollection],
    ):
        """
        Initialize a Transformer

        Args:
            options: the options for the runner
            always_active_restraints: these restraints are always active
            selectively_active_collections: these restraints are selected by the MELD algorithm
        """
        raise NotImplementedError("TransformerBase cannot be instantiated.")

    def add_interactions(self, system: mm.System, topology: app.Topology) -> mm.System:
        """
        Add new interactions to the system.

        This may involve:
        - Adding new forces, e.g. for restraints
        - Replacing an existing force with another, e.g. softcore
          interactions

        This method must return the modified system. If the
        transformer does not add interactions, it may simply
        return the passed values.

        Args:
            system: OpenMM system object to be modified
            topology: OpenMM topology object to be modified and/or used for indexing
        """
        return system

    def finalize(self, system: mm.System, topology: app.Topology) -> None:
        """
        Finalize the transformer.

        This method is guaranteed to be called after all forces
        are added to the system and provides an opportunity to
        do bookkeeping.

        This method should not add any new forces.

        Args:
            system: OpenMM system object to be modified
            topology: OpenMM topology object to be modified and/or used for indexing
        """
        pass

    def update(self, simulation: app.Simulation, alpha: float, timestep: int) -> None:
        """
        Update the system according to alpha and timestep.

        This method is called at the beginning of every stage.
        It should update forces and parameters as necessary.

        Parameters
        ----------
        simulation: OpenMM simulation object to be modified
        alpha: current value of alpha, ranges from 0 to 1
        stage: current stage of the simulation, starting from 0

        """
        pass


from meld.runner.transform.restraints import (
    ConfinementRestraintTransformer,
    RDCRestraintTransformer,
    CartesianRestraintTransformer,
    YZCartesianTransformer,
    COMRestraintTransformer,
    AbsoluteCOMRestraintTransformer,
    MeldRestraintTransformer,
)

from meld.runner.transform.rest2 import REST2Transformer
from meld.runner.transform.cmap import CMAPTransformer