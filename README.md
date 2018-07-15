## MELD

Modeling with limited data

JL MacCallum, A Perez, and KA Dill, Determining protein structures by combining semireliable data
with atomistic physical models by Bayesian inference, PNAS, 2015, 112(22), pp. 6985-6990.

### Latest Release

Release versions are built [here](https://github.com/maccallumlab/meld-pkg) and can be installed from the
[maccallum_lab anaconda channel](https://anaconda.org/maccallum_lab).

### Installation

The preferred way to install is:
```
conda config --add channels maccallum_lab omnia
conda install meld-cuda{VER}
```
where `VER` is currently one of `75`, `80`, `90`, or `92`.

This will install MELD and all of its dependencies.

### Testing

Test versions of MELD are built automatically. Current status:

Github `master`: [![Build Status](https://travis-ci.org/maccallumlab/meld.svg?branch=master)](https://travis-ci.org/maccallumlab/meld)
                [![Anaconda-Server Badge](https://anaconda.org/maccallum_lab/meld-test/badges/version.svg)](https://anaconda.org/maccallum_lab/meld-test)

Github `dev`: [![Build Status](https://travis-ci.org/maccallumlab/meld.svg?branch=dev)](https://travis-ci.org/maccallumlab/meld)
             [![Anaconda-Server Badge](https://anaconda.org/maccallum_lab/meld-dev-test/badges/version.svg)](https://anaconda.org/maccallum_lab/meld-dev-test)


## Building from Scratch

MELD requires a CUDA compatible GPU.

* ambermini or ambertools
* netcdf4
* mpi4py
* [openmm](https://github.com/pandegroup/openmm)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* python >= 3.6
* numpy
* scipy
* sklearn
* parmed

To install the python portion:
```
python setup.py install
```

To install the C++ / CUDA portion:
```
cd plugin
mkdir build
cd build
ccmake ..
make install
make PythonInstall
```

## Documentation

Documentation will eventually be at [project website](http://meldmd.org), but this is currently a placeholder.
