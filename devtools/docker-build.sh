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

# build the meld conda package
if [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "dev" ]]; then
    conda-build --no-binstar-upload --python 2.7 --python 3.4 --python 3.5 /io/devtools/conda/dev
else
    conda-build --no-binstar-upload --python 2.7 --python 3.4 --python 3.5 /io/devtools/conda/master
fi

# upload to anaconda.org
if [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "master" ]]; then
    anaconda --token "$ANACONDA_TOKEN" upload --user maccallum_lab /anaconda/conda-bld/linux-64/meld*.bz2
elif [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "dev" ]]; then
    anaconda --token "$ANACONDA_TOKEN" upload --user maccallum_lab /anaconda/conda-bld/linux-64/meld*.bz2
fi

# upload docs to S3

if [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "master" ]]; then
    aws s3 sync --region us-west-2 --delete /anaconda/conda-bld/work/build/meld-api-c++/ s3://plugin-api.meldmd.org/
fi
