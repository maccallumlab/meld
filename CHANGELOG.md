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
