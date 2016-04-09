#!/bin/bash

set -e

# Activate Holy Build Box environment
source /hbb_exe/activate

# Disable PYTHONPATH
unset PYTHONPATH

set -x

# install miniconda
curl -s -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /anaconda
PATH=/opt/rh/devtoolset-2/root/usr/bin:/opt/rh/autotools-latest/root/usr/bin:/anaconda/bin:$PATH
conda config --add channels omnia
# use Juanlu001 repository until we have an internal
conda config --add channels Juanlu001
conda install -yq conda-build jinja2 anaconda-client openmm-dev eigen3

# build the meld conda package
conda-build /io/devtools/conda

