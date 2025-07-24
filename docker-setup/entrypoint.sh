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

gosu "$USERNAME" /usr/local/bin/devuser-setup.sh
exec gosu "$USERNAME" "$@"
