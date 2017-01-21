#!/bin/bash

set -e

# Activate Holy Build Box environment
source /hbb_exe/activate

# Disable PYTHONPATH
unset PYTHONPATH

# install miniconda
curl -s -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /anaconda
PATH=/opt/rh/devtoolset-2/root/usr/bin:/opt/rh/autotools-latest/root/usr/bin:/anaconda/bin:$PATH
conda config --add channels omnia
conda install -yq conda-build jinja2 anaconda-client

# install aws
pip install awscli
yum install -y groff

# get the git revision for the version string
cd /io
GIT_DESCRIBE=`git describe --tags --long | tr - .`
MAJOR=`echo $GIT_DESCRIBE | cut -f1 -d.`
MINOR=`echo $GIT_DESCRIBE | cut -f2 -d.`
PATCH=`echo $GIT_DESCRIBE | cut -f3 -d.`
POST=`echo $GIT_DESCRIBE | cut -f4 -d.`
export VERSTRING=${MAJOR}.${MINOR}.${PATCH}.post${POST}
cd /

# decide if we should upload to anaconda cloud
if [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "dev" ]]; then
    UPLOAD="--upload maccallum_lab"
elif [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "master" ]]; then
    UPLOAD="--upload maccallum_lab"
else
    UPLOAD=""
fi

# build the meld conda package
if [[ "${TRAVIS_BRANCH}" == "master" ]]; then
    /io/devtools/conda-build-all $UPLOAD -- /io/devtools/conda/master
else
    /io/devtools/conda-build-all $UPLOAD -- /io/devtools/conda/dev
fi
