#!/bin/bash
set -e
THISDIR=$(dirname $(realpath $0))
cd $THISDIR

# libtorch
if [ ! -d libtorch ]; then
    wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip
    unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip
    rm libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip
fi

# eigen rand
if [ ! -d EigenRand ]; then
    git clone https://github.com/bab2min/EigenRand.git
fi

# tiny expr
if [ ! -d tinyexpr ]; then
    git clone https://github.com/codeplea/tinyexpr.git
fi
