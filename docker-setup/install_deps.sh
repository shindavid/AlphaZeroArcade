#!/bin/bash
set -e

# Configure timezone and install dependencies - needed for tzdata
ln -fs /usr/share/zoneinfo/UTC /etc/localtime

# Update package lists and upgrade existing packages
apt-get update && apt-get upgrade -y

# Install required dependencies
apt-get install -y \
    ack wget curl rsync unzip emacs vim git cmake gcc-12 g++-12 python3-pip \
    ninja-build software-properties-common libeigen3-dev libncurses5-dev \
    python-is-python3 libgtest-dev python3-cffi tzdata sqlite3 && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add PPA for the latest Boost
add-apt-repository -y ppa:mhier/libboost-latest
apt-get update && apt-get install -y libboost-json1.81-dev libboost-program-options1.81-dev \
    libboost-filesystem1.81-dev libboost-log1.81-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Update alternatives for GCC
for tool in gcc g++; do
    update-alternatives --install /usr/bin/$tool $tool /usr/bin/${tool}-12 60 || true
done

# Python dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install ipython==8.14.0 natsort==8.3.1 tqdm==4.66.1 termcolor==2.3.0 cffi numpy matplotlib \
    bokeh scipy flask

# Download and install libtorch C++ library
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip"
LIBTORCH_DIR="/opt/"

echo "Downloading libtorch from $LIBTORCH_URL..."
mkdir -p $LIBTORCH_DIR
wget -O libtorch.zip "$LIBTORCH_URL"
unzip -q libtorch.zip -d $LIBTORCH_DIR
rm libtorch.zip

echo "Libtorch installed in $LIBTORCH_DIR"

# Environment variables
echo "export PYTHONPATH=/workspace/py" >> /root/.bashrc
echo "export A0A_OUTPUT_DIR=/output" >> /root/.bashrc

# Misc
echo -e ".mode column\n.headers on" > /root/.sqliterc
