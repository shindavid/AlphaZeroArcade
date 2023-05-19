#!/bin/bash
set -e

# basics
apt-get update
apt-get install -y wget curl rsync unzip emacs git cmake gcc-12 g++-12 python3-pip libeigen3-dev libboost-all-dev libncurses5-dev
update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-12   60 || true
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 60 || true
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 60 || true

# cuda & python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install ipython natsort tqdm termcolor

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
echo "export PYTHONPATH=/AlphaZeroArcade/py" > /root/.bashrc
