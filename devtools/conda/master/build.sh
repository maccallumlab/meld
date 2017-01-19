#!/bin/bash

CMAKE_FLAGS="-DCMAKE_INSTALL_PREFIX=$PREFIX -DBUILD_TESTING=OFF"

CMAKE_FLAGS+=" -DCMAKE_BUILD_TYPE=Release"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
    export OPENMM_DIR=$PREFIX

    CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
    CMAKE_FLAGS+=" -DCUDA_CUDART_LIBRARY=${CUDA_PATH}/lib64/libcudart.so"
    CMAKE_FLAGS+=" -DCUDA_NVCC_EXECUTABLE=${CUDA_PATH}/bin/nvcc"
    CMAKE_FLAGS+=" -DCUDA_SDK_ROOT_DIR=${CUDA_PATH}/"
    CMAKE_FLAGS+=" -DCUDA_TOOLKIT_INCLUDE=${CUDA_PATH}/include"
    CMAKE_FLAGS+=" -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_PATH}/"
    CMAKE_FLAGS+=" -DCMAKE_CXX_FLAGS_RELEASE=-I/usr/include/nvidia/"
    CMAKE_FLAGS+=" -DOPENMM_DIR=$PREFIX"

    export EIGEN3_INCLUDE_DIR=$PREFIX/include/eigen3
    export LD_LIBRARY_PATH=$OPENMM_DIR/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CMAKE_FLAGS+=" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
    CMAKE_FLAGS+=" -DCMAKE_OSX_DEPLOYMENT_TARGET=10.9"
    CMAKE_FLAGS+=" -DCUDA_SDK_ROOT_DIR=/Developer/NVIDIA/CUDA-7.5"
    CMAKE_FLAGS+=" -DCUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-7.5"
    CMAKE_FLAGS+=" -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk"
    CMAKE_FLAGS+=" -DOPENMM_DIR=$PREFIX"

    export EIGEN3_INCLUDE_DIR=$PREFIX/include/eigen3/
    export OPENMM_DIR=$PREFIX
    export OPENMM_INCLUDE_PATH=$OPENMM_DIR/include
    export OPENMM_LIB_PATH=$OPENMM_DIR/lib
    export LD_LIBRARY_PATH=$OPENMM_DIR/lib:$LD_LIBRARY_PATH
fi

mkdir build
cd build

cmake .. $CMAKE_FLAGS
make -j$CPU_COUNT all
make -j$CPU_COUNT install PythonInstall

if [[ "$OSTYPE" == "linux-gnu" ]]; then
make -j$CPU_COUNT DoxygenApiDocs
fi
