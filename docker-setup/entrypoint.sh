#!/bin/bash
set -euo pipefail

cat << 'EOF' >> ~/.bashrc
# Show git branch name with dirty bit inside parentheses

# Variables to cache Git state
__git_branch=""
__git_dirty=""
__git_staged=""

# Function to update Git state
update_git_state() {
  # Check if the current directory is part of a Git repository
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # Cache the branch name
    __git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

    # Cache the dirty (unstaged) state
    __git_dirty=""
    if git diff --name-only 2>/dev/null | grep -q .; then
      __git_dirty="*"
    fi

    # Cache the staged state
    __git_staged=""
    if git diff --cached --name-only 2>/dev/null | grep -q .; then
      __git_staged="+"
    fi
  else
    # Clear cached values if not in a Git repository
    __git_branch=""
    __git_dirty=""
    __git_staged=""
  fi
}

# Prompt with Git state
PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[01;31m\]$(
  update_git_state
  if [ -n "$__git_branch" ]; then
    echo " ($__git_branch$([ -n "$__git_staged$__git_dirty" ] && echo " $__git_staged$__git_dirty"))"
  fi
)\[\033[00m\]\$ '

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

EOF

ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
echo "Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null" > ~/.ssh/config
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/known_hosts
sudo service ssh start

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
