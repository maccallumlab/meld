[![Build Status](https://travis-ci.org/maccallumlab/meld-openmm-plugin.svg?branch=master)](https://travis-ci.org/maccallumlab/meld-openmm-plugin)

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

### [C++ API](http://meld.dillgroup.io/annotated.html)
