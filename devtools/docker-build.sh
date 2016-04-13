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
conda install -yq conda-build jinja2 anaconda-client

# get the git revision for the version string
cd /io
export GIT_DESCRIBE=`git describe --tags --long | tr - .`
cd /

# build the meld conda package
conda-build --no-binstar-upload --python 2.7 --python 3.4 --python 3.5 /io/devtools/conda

# upload to anaconda.org
anaconda --token $ANACONDA_TOKEN upload --user maccallumlab /anaconda/conda-bld/linux-64/meld*.bz2
