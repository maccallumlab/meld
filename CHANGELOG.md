## 0.6.3

### Enhancements
- Add best-fitting model selection workflow for CryoFold
- CryFold tutorial script adapted for command-line use
- Update python dependencies in setup.py

## 0.6.2

### Enhancements
- Decoupled the number of workers from the number of replicas.
  - You can now use any combination that divides evenly
    - e.g. 64 replicas on 16 workers
  - `launch_remd_multiplex` has been removed
    - use `launch_remd` with one worker instead
- RDCs now work with peak mapping
- Ability to freeze atoms with `freeze_atoms`
- Ability to remove the potential from the force field with `remove_potential`
  - This can be useful when trying to infer assignments using peak mapping

## 0.6.1

### Enhancements
- Added helper methods to simplify setting up simulations
  - `setup_replica_exchange`
  - `setup_data_store`

## 0.6.0

### Bug Fixes
- Previous versions of MELD could give erroneous values for forces and energies
  - Due to a data race in the CUDA kernels, different threads could calculate
    different values for the cutoff energy within each collection. This would
    result in groups occasionally being activated when they shouldn't be and
    vice versa.
  - We now use nvidia's cub library to handle the sort operation required for
    groups and collections.
  - The new version of the CUDA kernel agrees well with the CPU verion.
  - Compiling cub library requires the use of nvcc, which must be forced by
    setting `OPENMM_CUDA_COMPILER`.
- Previous versions could sometimes incorrectly negotiate device ids between
  nodes which would lead to failure at runtime.

### Enhancements
- Amber-specific system buiding tools are in their own modules
- RDCs no longer add particles to the system
- Energies are saved to different energy groups. The current mapping is:
  - Base forcefield energy: 0
  - MELD force: 1
  - RDC force: 2
  - Confinement restraints: 3
  - COM restraints: 4
  - Cartesian restraints: 5
  - Log Prior for parameter sampling: 6
- Most user facing modules are now imported into the root `meld` namespace.
- Most user-facing functions now require units, which will help to eliminate
  errors.
- You can now specify the platform to use.
  - Options are: "CUDA", "CPU", "Reference"
  - E.g. `launch_remd --platform CPU`
- Many changes to indexing.
  - All atom indexing is zero-based.
  - By default, residue and chain indexing is zero-based, but individual
    lookups can specify one-based indexing.
  - `system.index.atom(resid, atom_name, chainid=None, one_based=False)`
    - Can find the index of an atom.
    - Absolute index if `chainid=None`.
    - Relative index within a chain if `chainid` is specified.
    - Both `resid` and `chainid` are zero-based by default, but can be one based
      if `one_based=True`.
    - Returns an `AtomIndex`, which is a zero-based absolute atom index.
  - `system.index.residue(resid, chainid=None, one_based=False)`
    - Can find the index of a residue.
    - Absolute index if `chainid=None`.
    - Relative index within a chain if `chainid` is specified.
    - Both `resid` and `chainid` are zero-based by default, but can be one based
      if `one_based=True`.
    - Returns a `ResidueIndex`, which is a zero-based absolute residue index.
  - All functions defining restraints now take `AtomIndex` rather sets of residues and
    atom names.
  - "follower" changed to "worker"

### Changes
- No longer use network logging.
  - Previous versions used network logging to aggregate all log messages into a
    single remd.log file.
  - This required the leader task to spawn an additional process to run the log
    server. Getting this right was difficult and was the source of many developer
    headaches.
  - This feature has now been removed and all worker nodes write to their own log
    file in the `/Logs` directory.

## 0.5.0

### Enhancements
- Implented MELD plugin for CPU and Reference plugins
  - This should allow for improved minimization. When large forces are encountered
    during minimization, OpenMM switches to the CPU platform. Previously, this would
    cause calculations using forces in `meldplugin` to crash. With this version,
    `meldplugin` forces are now supported on the CPU platform, so the minimizer
    __may__ succeed.
  - This implmentation is incomplete, as it does not (yet) allow for the whole
    meld calculation to run on the CPU, but that should be possible in the
    future.
  - Only the basic distance and torsion restraints are currently implemented. Use of
    other restraint types will cause an error on the CPU platform.

## 0.4.20

## Bug Fixes
- Changed negotiation of device ids, which previously caused problems
  with startup on some HPC systems.

## 0.4.19

### Enhancements
- Improved minimization with RDCs

## 0.4.18

### Bug Fixes
- Fixed race condition in logging.

## 0.4.17

### Enhancements
- Changed default minimum value for restraint scalers
  - Previously, restraints would be scaled to 0.0 by default when alpha=1.0
  - This could cause problems with some restraints, like RDCs
  - The new default scales the restraints down to 1e-3, which is effectively off, but
    prevents problems with restraints like RDCs
  - The previous behaviour can be obtained by setting `stregnth_at_alpha_max=0.0`
- Combined license into a single file

### Bug Fixes
- Fixed error that caused simulations to hang upon completion or upon error

## 0.4.16

### Bug Fixes
- Minor changes to plugin build process.

## 0.4.15

### Enhancements
- Continuous integration now uses github actions.

### Bug Fixes
- `quadratic_cut` option of `get_secondary_structure_restraints` is now interpreted correctly.
  Previously, this value was interpreted as tens of nanometers, instead of tenths of nanometers.

## 0.4.14

### Enhancements

- Implemented flat-bottom / quadratic / linear RDC restraints

## 0.4.13

### Enhancements

- Much faster handling of RDCs following the approach of Habeck and co-workiers. DOI 10.1007/s10858-007-9215-1
## 0.4.12

### Enhancements

- You can now specify quadratic_cut for RDC restraints. The energy will increase
  quadratically when |d_obs - d_calc| is less than quadratic cut and linearly
  when it is larger.

### Bug Fixes

- Fixed a bug where RDC energies were incorrect in mixed precision.

## 0.4.11

### Bug Fixes

- Fixed updating of RDC forces. Previously, the RDC forces were not correctly
  updated with alpha and timestep.

## 0.4.10

### Enhancements

- Major improvement in simulation startup time with large numbers of restraints.

## 0.4.9

### Bug Fixes

- Fixed inconsistencies in RDC calculations

## 0.4.8

### Enhancements

- Added support for implicit ion concentration and solute/solvent dielectric constants.

### Bug Fixes

- Fixed major bug in handling of groups where the number of
  restraints to be satisfied was less than the total size
  of the group. On GTX 1080 Ti, this resulted in frequent
  crashes with `CUDA Error (700)`. It is unknown what
  effect this would have on other platforms. Simulations
  with groups that were 100% satisfied are unaffected.

## 0.4.7

### Enchancements

- Support for building explicit solvent systems
- Support for periodic boundary conditions
- Started working on documentation

### Bug Fixes

- Fixed incorrect tests for GMM restraints

## 0.4.6

### Enhancements

- Exceptions now flush buffers which fixes issue with some MPI implmentations dropping diagnostic information
- Long amino acid sequences no longer crash `tleap`

### Major Changes

- Drop osx support
  - There really isn't a compelling use case and most recent Macs don't have NVIDIA GPUs
  
## 0.4.5

### Bug Fixes

- Fixed bug in positioners with distance restraints that caused runs to fail
  immediately.
  
## 0.4.4

### Enhancements

- Added support for positioners in distance restraints

## 0.4.3

### Bug Fixes

- Changed handling of GMM scaling factors

## 0.4.2

### Bug Fixes

- Enhanced numerical stability of GMM simulations

## 0.4.1

### Bug Fixes

- Change the way the scale factor for GMMs is passed
  to the GPU. The previous version gave incorrect
  results.
  
## 0.4.0

### Enhancements

- Added Gaussian Mixture Model (GMM) restraints

### Bug Fixes

- Fixed bug in timeout handling

## 0.3.13

### Bug Fixes
- Fixed an issue that could cause ImportErrors

## 0.3.12

### Enhancements

- `meld.comm.MPICommunicator` objects now have a timeout.
  The default is 600 seconds, or 10 minutes. This should
  prevent hangs that were occasionally observed.

## 0.3.11

### Enhancements

- Virtual spin labels now use a restricted angle potential
  that avoids 180 and 0 degree angles that can cause the
  dihedral potential to blow up.
  
## 0.3.10

### Enhancements

- Completed implementation of virtual spin label sites

## 0.3.9

### Bug Fixes

- Fixed a bug where OpenMM would re-order identical
  chains involved in MELD forces. This could only
  occur if identical molecules were involved in a 
  MELD force. MELD now correctly specifies that any
  molecules involved in a MELD force are unique and
  should not be reordered.
  
## 0.3.8

### Bug Fixes

- *(CRITICAL)* The fix introduced in 0.3.7 was incorrect
   and sometimes gave incorrect results due to a data race.
   This version uses a work efficient, in place, parallel
   algorithm. It passes several hundred iterations of the
   plugin test-suite, whereas the previous version would
   fail every few trials.
   
### Enhancements

- Added improved diagnostics to aid in future debugging of
  collections.

## 0.3.7

### Enhancements

- *(CRITICAL)* Fixed a bug in collection handling introduced in
   e71Daa0 that caused undefined behaviour. The previous version
   gave correct results on test systems with GTX980 cards, but
   incorrect results on Tesla m2070. The status on other
   architectures is unknown.

## 0.3.6

### Bug Fixes

- *(CRITICAL)* Fixed bug in mixed precision introduced in a727279.
  The energies reported in previous versions were undefined.

## 0.3.5

### Bug Fixes

- *(CRITICAL)* Fixed bug in secondary structure handling that was
   introduced in 0.3.2.

## 0.3.4

### Enhancments
- Added ability to specify arbitrary leap commands in header

## 0.3.3

### Enhancements
- TranslationMover for translational Monte Carlo moves

## 0.3.2

### Enhancements
- Support for reading in secondary structure from multiple chains

## 0.3.1

### Bug Fixes
- *(CRITICAL)* Fixed critical bug where MELD forces were not
  being updated on GPU.
  
## 0.3.0

### Enhancements
- This is the first version where we are tracking changes.
- The plugin has been merged into the main MELD repository.
- We now build a single anaconda package (meld-test) on each commit.
