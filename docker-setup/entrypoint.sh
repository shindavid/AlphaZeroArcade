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

# Execute the command as the created user
exec gosu "$USER_NAME" "$@"
