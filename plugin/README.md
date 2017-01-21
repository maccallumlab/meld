### Latest Release

Omnia `master`:[![Build Status](https://travis-ci.org/omnia-md/conda-recipes.svg?branch=master)](https://travis-ci.org/omnia-md/conda-recipes) 
[![Anaconda-Server Badge](https://anaconda.org/omnia/meld-plugin/badges/version.svg)](https://anaconda.org/omnia/meld-plugin)

### Testing
Branch `master`: [![Build Status](https://travis-ci.org/maccallumlab/meld-openmm-plugin.svg?branch=master)](https://travis-ci.org/maccallumlab/meld-openmm-plugin)
[![Anaconda-Server Badge](https://anaconda.org/maccallum_lab/meld-plugin-test/badges/version.svg)](https://anaconda.org/maccallum_lab/meld-plugin-test)

Branch `dev`: [![Build Status](https://travis-ci.org/maccallumlab/meld-openmm-plugin.svg?branch=dev)](https://travis-ci.org/maccallumlab/meld-openmm-plugin)
[![Anaconda-Server Badge](https://anaconda.org/maccallum_lab/meld-plugin-dev-test/badges/version.svg)](https://anaconda.org/maccallum_lab/meld-plugin-dev-test)
  

## OpenMM Meld Plugin

This plugin implements several forces that are used by the [MELD](https://github.com/maccallumlab/meld) package.

## Installing

The easiest way to install is to use the
[Anaconda packages](https://anaconda.org/omnia/meld-plugin)
from the `omnia` channel:
```
conda config --add channels omnia
conda install meld-plugin
```

## Building The Plugin

If you need to build the plugin yourself, you first need to install OpenMM, Eigen3, and CUDA.

Next, after editing the paths as appropriate, you can use something like the following to build and install.

    export OPENMM_DIR=/home/ec2-user/openmm
    export OPENMM_INCLUDE_PATH=$OPENMM_DIR/include
    export OPENMM_LIB_PATH=$OPENMM_DIR/lib
    export LD_LIBRARY_PATH=$OPENMM_DIR/lib
    export LD_LIBRARY_PATH=/opt/nvidia/cuda/lib:/opt/nvidia/cuda/lib64:$LD_LIBRARY_PATH
    export OPENMM_CUDA_COMPILER=/opt/nvidia/cuda/bin/nvcc
    export EIGEN3_INCLUDE_DIR=/usr/local/include/eigen3

    git clone https://github.com/maccallumlab/meld-openmm-plugin.git
    cd meld-openmm-plugin
    mkdir _build && cd _build
    cmake .. -DCMAKE_INSTALL_PREFIX=/home/ec2-user/openmm
    make
    make install
    make PythonInstall

## Documentation

See the following:
* [C++ API](http://plugin-api.meldmd.org)
* [MELD](http://meldmd.org)
