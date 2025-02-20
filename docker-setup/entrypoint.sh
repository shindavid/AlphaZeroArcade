#!/usr/bin/env bash
set -e

HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}
USERNAME=${USERNAME:-devuser}
LOCAL=${LOCAL:-false}

if ! getent group "$HOST_GID" >/dev/null; then
  groupadd -g "$HOST_GID" $USERNAME
fi

if ! id -u "$HOST_UID" >/dev/null 2>&1; then
  useradd -m -u "$HOST_UID" -g "$HOST_GID" -s /bin/bash "$USERNAME"
fi

echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

if [ "$LOCAL" = "true" ]; then
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
  mkdir -p /workspace
  chown $USERNAME:$USERNAME /workspace
fi

gosu "$USERNAME" /usr/local/bin/devuser-setup.sh
exec gosu "$USERNAME" "$@"
