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
