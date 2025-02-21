#!/usr/bin/env bash
set -e

HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}
USERNAME=${USERNAME:-devuser}
PLATFORM=${PLATFORM:-native}

if ! getent group "$HOST_GID" >/dev/null; then
  groupadd -g "$HOST_GID" $USERNAME
fi

if ! id -u "$HOST_UID" >/dev/null 2>&1; then
  useradd -m -u "$HOST_UID" -g "$HOST_GID" -s /bin/bash "$USERNAME"
fi

echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

if [ "$PLATFORM" = "native" ]; then
  mkdir -p /workspace
  chown $USERNAME:$USERNAME /workspace
fi

service ssh start

gosu "$USERNAME" /usr/local/bin/devuser-setup.sh
exec gosu "$USERNAME" "$@"
