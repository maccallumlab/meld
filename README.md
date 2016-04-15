Build Status
============
* Anaconda Release version:
  * Release [![Anaconda-Server Badge](https://anaconda.org/omnia/meld-plugin/badges/version.svg)](https://anaconda.org/omnia/meld-plugin)
* Testing:
  * github `master`: [![Build Status](https://travis-ci.org/maccallumlab/meld-openmm-plugin.svg?branch=master)](https://travis-ci.org/maccallumlab/meld-openmm-plugin)
  * anaconda `meld-plugin-test`: [![Anaconda-Server Badge](https://anaconda.org/maccallum_lab/meld-plugin-test/badges/version.svg)](https://anaconda.org/maccallum_lab/meld-plugin-test)
  * github `dev`: [![Build Status](https://travis-ci.org/maccallumlab/meld-openmm-plugin.svg?branch=dev)](https://travis-ci.org/maccallumlab/meld-openmm-plugin)
  * anaconda `meld-plugin-dev-test`: [![Anaconda-Server Badge](https://anaconda.org/maccallum_lab/meld-plugin-dev-test/badges/version.svg)](https://anaconda.org/maccallum_lab/meld-plugin-dev-test)
  
OpenMM Meld Plugin
=====================

This plugin defines two OpenMM Force subclasses: MeldForce and RdcForce. To install it, you'll first need to install OpenMM, Eigen3, and CUDA.

Building The Plugin
===================

    export OPENMM_DIR=/home/ec2-user/openmm
    export OPENMM_INCLUDE_PATH=$OPENMM_DIR/include
    export OPENMM_LIB_PATH=$OPENMM_DIR/lib
    export LD_LIBRARY_PATH=$OPENMM_DIR/lib
    export LD_LIBRARY_PATH=/opt/nvidia/cuda/lib:/opt/nvidia/cuda/lib64:$LD_LIBRARY_PATH
    export OPENMM_CUDA_COMPILER=/opt/nvidia/cuda/bin/nvcc
    export EIGEN3_INCLUDE_DIR=/usr/local/include/eigen3

    git clone https://github.com/grollins/meld-openmm-plugin.git
    cd meld-openmm-plugin
    mkdir _build && cd _build
    cmake .. -DCMAKE_INSTALL_PREFIX=/home/ec2-user/openmm
    make
    make install
    make PythonInstall

Documentation
=============

See the following:
* [C++ API](http://plugin-api.meldmd.org)
* [MELD](http://github.com/maccallumlab/meld)
