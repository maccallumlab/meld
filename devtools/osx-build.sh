set -e -x

brew update -y --quiet

# install miniconda
curl -s -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh;
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/anaconda;
export PATH=$HOME/anaconda/bin:$PATH;
conda config --add channels omnia;
conda install -yq conda-build jinja2 anaconda-client;

# install cuda
curl -O -s http://developer.download.nvidia.com/compute/cuda/7.5/Prod/network_installers/mac/x86_64/cuda_mac_installer_tk.tar.gz
curl -O -s http://developer.download.nvidia.com/compute/cuda/7.5/Prod/network_installers/mac/x86_64/cuda_mac_installer_drv.tar.gz
sudo tar -zxf cuda_mac_installer_tk.tar.gz -C /;
sudo tar -zxf cuda_mac_installer_drv.tar.gz -C /;
rm -f cuda_mac_installer_tk.tar.gz cuda_mac_installer_drv.tar.gz

# get the git revision for the version string
GIT_DESCRIBE=`git describe --tags --long | tr - .`
MAJOR=`echo $GIT_DESCRIBE | cut -f1 -d.`
MINOR=`echo $GIT_DESCRIBE | cut -f2 -d.`
PATCH=`echo $GIT_DESCRIBE | cut -f3 -d.`
POST=`echo $GIT_DESCRIBE | cut -f4 -d.`
export VERSTRING=${MAJOR}.${MINOR}.${PATCH}.post${POST}

pwd
ls


# build the meld conda package
if [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "dev" ]]; then
    conda-build --no-binstar-upload --python 2.7 --python 3.4 --python 3.5 devtools/conda/dev
else
    conda-build --no-binstar-upload --python 2.7 --python 3.4 --python 3.5 devtools/conda/master
fi

# upload to anaconda.org
if [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "master" ]]; then
    anaconda --token "$ANACONDA_TOKEN" upload --user maccallum_lab /Users/travis/anaconda/conda-bld/osx-64/meld*.bz2
elif [[ "${TRAVIS_PULL_REQUEST}" == "false" && "${TRAVIS_BRANCH}" == "dev" ]]; then
    anaconda --token "$ANACONDA_TOKEN" upload --user maccallum_lab /Users/travis/anaconda/conda-bld/osx-64/meld*.bz2
fi
