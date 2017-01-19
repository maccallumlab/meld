set -e -x

brew update -y --quiet

# install miniconda
curl -s -O https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh;
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/anaconda;
export PATH=$HOME/anaconda/bin:$PATH;
conda config --add channels omnia;
conda install -yq conda-build jinja2 anaconda-client;

# install cuda
curl -O -s http://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/Prod/network_installers/mac/x86_64/cuda_mac_installer_tk.tar.gz
curl -O -s http://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/Prod/network_installers/mac/x86_64/cuda_mac_installer_drv.tar.gz
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
    devtools/conda-build-all $UPLOAD -- devtools/conda/master
else
    devtools/conda-build-all $UPLOAD -- devtools/conda/dev
fi
