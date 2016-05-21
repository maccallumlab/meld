### Latest Release

We are not yet building the release version automatically.

### Testing

Github `master`: [![Build Status](https://travis-ci.org/maccallumlab/meld.svg?branch=master)](https://travis-ci.org/maccallumlab/meld)
                [![Anaconda-Server Badge](https://anaconda.org/maccallum_lab/meld-test/badges/version.svg)](https://anaconda.org/maccallum_lab/meld-test)

Github `dev`: [![Build Status](https://travis-ci.org/maccallumlab/meld.svg?branch=dev)](https://travis-ci.org/maccallumlab/meld)
             [![Anaconda-Server Badge](https://anaconda.org/maccallum_lab/meld-dev-test/badges/version.svg)](https://anaconda.org/maccallum_lab/meld-dev-test)

## MELD

Modeling with limited data

## Installation

The easiest way to install MELD is using [Anaconda packages](https://anaconda.org/omnia/meld) from the `omnia` channel:
```
conda config --add channels omnia
conda install meld
```

## Building from Scratch

MELD requires a CUDA compatible GPU.

* ambermini or ambertools
* netcdf4
* mpi4py
* [openmm](https://github.com/pandegroup/openmm)
* [meld-plugin](https://github.com/maccallumlab/meld)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

To install:
```
python setup.py install
```

## Documentation

See the [project website](http://meldmd.org) for full documentation.
