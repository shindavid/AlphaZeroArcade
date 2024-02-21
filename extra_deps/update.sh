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

