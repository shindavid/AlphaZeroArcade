#!/bin/bash
set -euo pipefail

# Set default values if environment variables are not set
USER_ID=${USER_ID:-1000}
GROUP_ID=${GROUP_ID:-1000}
USER_NAME=${USER_NAME:-devuser}
GROUP_NAME=${GROUP_NAME:-devgroup}

echo "Starting with UID: $USER_ID, GID: $GROUP_ID"

# Create the group if it doesn't exist
if ! getent group "$GROUP_NAME" >/dev/null; then
    groupadd -g "$GROUP_ID" "$GROUP_NAME"
    echo "Created group $GROUP_NAME with GID $GROUP_ID"
else
    echo "Group $GROUP_NAME already exists"
fi

# Create the user if it doesn't exist
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
    useradd -u "$USER_ID" -g "$GROUP_NAME" -m "$USER_NAME"
    echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
    echo "Created user $USER_NAME with UID $USER_ID and GID $GROUP_ID"
else
    echo "User $USER_NAME already exists"
fi

# Change ownership of the workspace directory
chown -R "$USER_NAME":"$GROUP_NAME" /workspace

# We install libtorch at RUNTIME, rather than at build time, for faster image-loading
# The intended usage is for /workspace to be a persistent volume, so the libtorch installation
# only needs to be done once, and will be available for all subsequent runs of the container.
LIBTORCH_VERSION="libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124"
LIBTORCH_URL="https://download.pytorch.org/libtorch/cu124/$LIBTORCH_VERSION.zip"
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
    ln -s "$LIBTORCH_VERSION" current
    cd -
    chown -R "$USER_NAME":"$GROUP_NAME" $LIBTORCH_BASE_DIR
    echo "Libtorch installed in $LIBTORCH_DIR"
else
    echo "Found libtorch installation in $LIBTORCH_BASE_DIR"
fi

# Execute the command as the created user
exec gosu "$USER_NAME" "$@"
