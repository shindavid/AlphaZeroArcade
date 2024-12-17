#!/bin/bash
set -euo pipefail

# We install libtorch at RUNTIME, rather than at build time, for faster image-loading
# The intended usage is for /workspace to be a persistent volume, so the libtorch installation
# only needs to be done once, and will be available for all subsequent runs of the container.

LIBTORCH_VERSION="libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu118/$LIBTORCH_VERSION.zip"
LIBTORCH_BASE_DIR="/workspace/libtorch"
LIBTORCH_DIR="$LIBTORCH_BASE_DIR/$LIBTORCH_VERSION"

if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "$LIBTORCH_VERSION installation not found."
    echo "*****************************************************"
    echo "* PERFORMING ONE-TIME SLOW INSTALLATION OF libtorch *"
    echo "*****************************************************"
    echo "Downloading from $LIBTORCH_URL"
    mkdir -p "$LIBTORCH_BASE_DIR"
    wget -O /tmp/libtorch.zip "$LIBTORCH_URL"
    unzip -q /tmp/libtorch.zip -d $LIBTORCH_BASE_DIR
    mv "$LIBTORCH_BASE_DIR/libtorch" "$LIBTORCH_DIR"
    rm /tmp/libtorch.zip
    cd "$LIBTORCH_BASE_DIR"
    rm -f current
    ln -s "$LIBTORCH_VERSION" current
    cd -
    echo "Libtorch installed in $LIBTORCH_DIR"
else
    echo "Found libtorch installation in $LIBTORCH_BASE_DIR"
fi

echo "Running as UID: $(id -u), GID: $(id -g)"
echo "User: $(id -nu)"
echo "Group: $(id -ng)"

exec "$@"
