#!/usr/bin/env bash
set -e

HOST_UID=${HOST_UID:-1000}
HOST_GID=${HOST_GID:-1000}
USERNAME=${USERNAME:-devuser}
PLATFORM=${PLATFORM:-native}

echo "GROUPS:" && getent group | grep "^.*:.*:${HOST_GID}:"

if ! getent group "$USERNAME" >/dev/null; then
  if getent group "$HOST_GID" >/dev/null; then
    # GID already exists under another name; rename it
    existing=$(getent group "$HOST_GID" | cut -d: -f1)
    groupmod -n "$USERNAME" "$existing"
  else
    groupadd -g "$HOST_GID" "$USERNAME"
  fi
fi

if ! getent passwd "$USERNAME" >/dev/null 2>&1; then
  if getent passwd "$HOST_UID" >/dev/null 2>&1; then
    # UID already exists under another name; rename it
    existing_user=$(getent passwd "$HOST_UID" | cut -d: -f1)
    usermod -l "$USERNAME" "$existing_user"
    usermod -d "/home/$USERNAME" -m "$USERNAME"
    usermod -g "$HOST_GID" "$USERNAME"
  else
    useradd -m -u "$HOST_UID" -g "$HOST_GID" -s /bin/bash "$USERNAME"
  fi
fi

echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

if [ "$PLATFORM" = "native" ]; then
  mkdir -p /workspace
  chown $USERNAME:$USERNAME /workspace
fi

service ssh start

# Write vscode settings file
mkdir -p /root/.vscode-server/data/Machine
cat << 'EOF' > /root/.vscode-server/data/Machine/settings.json
{
  "clangd.path": "/usr/bin/clangd",
  "clangd.arguments": [
      "--compile-commands-dir=target/Release"
  ]
}
EOF

DOCKER_SOCK=/var/run/docker.sock
if [ -S "$DOCKER_SOCK" ]; then
  DOCKER_GID="$(stat -c '%g' "$DOCKER_SOCK")"
  echo "[entrypoint] docker.sock gid=$DOCKER_GID"

  # Ensure a group exists with that numeric gid
  SOCK_GRP="$(getent group "$DOCKER_GID" | cut -d: -f1)"
  if [ -z "$SOCK_GRP" ]; then
    SOCK_GRP=docker
    if getent group "$SOCK_GRP" >/dev/null; then
      SOCK_GRP=dockersock
    fi
    echo "[entrypoint] creating group '$SOCK_GRP' (gid=$DOCKER_GID)"
    groupadd -g "$DOCKER_GID" "$SOCK_GRP"
  else
    echo "[entrypoint] using existing group '$SOCK_GRP' for gid=$DOCKER_GID"
  fi

  # Add the user (idempotent)
  if ! id -nG "$USERNAME" | tr ' ' '\n' | grep -qx "$SOCK_GRP"; then
    echo "[entrypoint] adding $USERNAME to '$SOCK_GRP'"
    usermod -aG "$SOCK_GRP" "$USERNAME"
  else
    echo "[entrypoint] $USERNAME already in '$SOCK_GRP'"
  fi

  echo "[entrypoint] groups($USERNAME) now: $(id -nG "$USERNAME" 2>/dev/null || true)"
fi


gosu "$USERNAME" /usr/local/bin/devuser-setup.sh
exec gosu "$USERNAME" "$@"
