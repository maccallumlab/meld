'''
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

'''


class TransformerBase(object):
    '''
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

    Parameters
    ----------
    options: meld.system.RunOptions
        the options for the runner
    always_active_restraints: list of restraints
        these restraints are always active
    selectively_active_collections: list of SelectivelyActiveCollection
        these restraints are selected by the MELD algorithm

    '''
    def __init__(self, options, always_active_restraints,
                 selectively_active_restraints):
        raise NotImplementedError('TransformerBase cannot be instantiated.')

    def add_interactions(self, system, topology):
        '''
        Add new interactions to the system.

        This may involve:
        - Adding new forces, e.g. for restraints
        - Replacing an existing force with another, e.g. softcore
          interactions

        This method must return the modified system. If the
        transformer does not add interactions, it may simply
        return the passed values.

        Parameters
        ----------
        system: simtk.openmm.System
            OpenMM system object to be modified
        topology: simtk.openmm.Topology
            OpenMM topology object to be modified and/or used
            for indexing

        '''
        return system

    def finalize(self, system, topology):
        '''
        Finalize the transformer.

        This method is guaranteed to be called after all forces
        are added to the system and provides an opportunity to
        do bookkeeping.

        This method should not add any new forces.

        Parameters
        ----------
        system: simtk.openmm.System
            OpenMM system object to be modified
        topology: simtk.openmm.Topology
            OpenMM topology object to be modified and/or used
            for indexing

        '''
        pass

    def update(self, simulation, alpha, timestep):
        '''
        Update the system according to alpha and timestep.

        This method is called at the beginning of every stage.
        It should update forces and parameters as necessary.

        Parameters
        ----------
        simulation: simtk.openmm.app.simulation
            OpenMM simulation object to be modified
        alpha: float
            Current value of alpha, ranges from 0 to 1
        stage: int
            Current stage of the simulation, starting from 0

        '''
        pass


from meld.system.openmm_runner.transform.restraints import (
    ConfinementRestraintTransformer,
    RDCRestraintTransformer,
    CartesianRestraintTransformer,
    YZCartesianTransformer,
    COMRestraintTransformer,
    AbsoluteCOMRestraintTransformer,
    MeldRestraintTransformer)

from meld.system.openmm_runner.transform.rest2 import REST2Transformer
