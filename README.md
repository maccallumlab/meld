# MELD: Modeling employing limited data

Please cite:

JL MacCallum, A Perez, and KA Dill, Determining protein structures by combining semireliable data
with atomistic physical models by Bayesian inference, PNAS, 2015, 112(22), pp. 6985-6990.

## Current release info

| Name | Downloads | Version | Platforms |
| --- | --- | --- | --- |
| [![Conda Recipe](https://img.shields.io/badge/recipe-meld-green.svg)](https://anaconda.org/conda-forge/meld) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/meld.svg)](https://anaconda.org/conda-forge/meld) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/meld.svg)](https://anaconda.org/conda-forge/meld) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/meld.svg)](https://anaconda.org/conda-forge/meld) |

## Testing

Test versions of MELD are built automatically. Current status:
[![meld_build](https://github.com/maccallumlab/meld/actions/workflows/CI.yml/badge.svg)](https://github.com/maccallumlab/meld/actions)


## Installation

MELD can be installed either through [conda-forge](https://conda-forge.org/) or from source. Installation through 
conda-forge is generally simpler and should be preferred.

### Conda Forge

First install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#installing) or
[miniforge](https://github.com/conda-forge/miniforge) by following the appropriate instructions.

If using `miniconda`, we recommend setting `conda-forge` as the default channel. (This is already enabled for `miniforge`.)
```
conda config --add channels conda-forge 
conda config --set channel_priority strict
```

We recommend installing MELD into a conda environment. You can name this however you want. We usually name this by the
meld version or by the project name, e.g.
```
conda create -n my-meld-project python
conda activate my-meld-project
conda install meld
```

This will create and activate an environment called `my-meld-project`, activate it, and install MELD and its dependencies.

The current supported CUDA versions are `10.2`, `11.0`, `11.1`, and `11.2`. By default, `conda` will install MELD
for the higest supported version on your system. On some HPC systems, you may be able to load different versions of the cuda
library using the `module` command or similar. If you need to install MELD for a different version of CUDA than is
auto-detected, you can use e.g. `conda install cudatoolkit=10.2 meld`.

The last step is to **install mpi4py**, see below.

### Building from Scratch

MELD requires a CUDA compatible GPU.

* ambermini or ambertools
* netcdf4
* [openmm](https://github.com/pandegroup/openmm)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
* python >= 3.6
* numpy
* scipy
* sklearn
* progressbar
* eigen3
* mpi4py (see below)

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

### Installing mpi4py

MELD requires `mpi4py`, but does not include it as a dependency, as there are multiple prefered ways to install
it, depending on your environment.

If your cluster does not use mpi libraries that are tightly coupled to a high-performance network or to the queuing system,
you can simply use the version provided by conda-forge.

To use openmpi:
```
conda install openmpi mpi4py
```
To use mpich:
```
conda install mpich mpi4py
```

If your cluster uses mpi libraries that are system-specific, you will likely need to compile from source:
```
pip install --no-deps mpi4py
```
You may need to load `module`s and/or configure environment variables for this to work. Consult your system adminstrator or
cluster documentation for guidance.

## Documentation

There is a limited amount of documentation at [meldmd.org](http://meldmd.org). Assistance in building out the documentation
is appreciated.
