#!/bin/bash -l
set -e
REPODIR=$(dirname $(dirname $(dirname $(realpath $0))))
cd $REPODIR

# add docker group and re-enter this script with it
groups | grep docker || {
    # set up user for docker
    sudo groupadd -f docker
    sudo usermod -aG docker $USER
    exec sg - docker $0
}

# set up just
if [ ! -e /usr/bin/just ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | sudo bash -s -- --to /usr/bin
fi

# build docker container
./docker/build.sh

# generate build config
./docker/shell just genconfig
