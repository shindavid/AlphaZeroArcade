#!/bin/bash
set -e

# basics
apt-get update
apt-get install -y wget curl rsync unzip emacs git cmake gcc-12 g++-12 python3-pip libeigen3-dev libboost-all-dev libncurses5-dev
update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-12   60 || true
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 || true
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60 || true
rm -rf /var/lib/apt/lists/*
apt-get clean

# conda
export CONDA_DIR=/opt/conda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
chmod +x /tmp/miniconda.sh
/tmp/miniconda.sh -b -p $CONDA_DIR
rm -rf /tmp/miniconda.sh
# Put conda in path so we can use conda activate
echo "export CONDA_DIR=$CONDA_DIR" >> /root/.bashrc
echo "export PATH=$CONDA_DIR/bin:$PATH" >> /root/.bashrc
echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> /root/.bashrc
echo "conda activate base" >> /root/.bashrc

# install just
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/bin

# hack to disable compiler version checking for nvcc:
#   this is needed because we are using gcc-12 and cuda checks for gcc <=11
#   it has an --allow-unsupported-compiler mode but I cannot figure out how to
#   push that through cmake. So, instead, we just modify the system header to disable
#   that check. Ideally we should probably just stay on a compatible g++ version
echo "#define __NV_NO_HOST_COMPILER_CHECK 1" > /tmp/host_config.h
cat /usr/local/cuda-11.8/targets/x86_64-linux/include/crt/host_config.h >> /tmp/host_config.h
cp /tmp/host_config.h /usr/local/cuda-11.8/targets/x86_64-linux/include/crt/host_config.h

# env vars
# echo "export PYTHONPATH=/AlphaZeroArcade/py" >> /root/.bashrc

# libtorch dependency
wget --quiet https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu118.zip -O /libtorch.zip
unzip /libtorch.zip -d /dependencies
rm -rf /libtorch.zip
echo "export A0A_LIBTORCH_DIR=/dependencies/libtorch" >> /root/.bashrc
