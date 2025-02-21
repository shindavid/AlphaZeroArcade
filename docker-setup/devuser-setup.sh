#!/usr/bin/env bash
set -e

echo "Setting up devuser environment in home: $HOME"

ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
echo "Host *
    StrictHostKeyChecking accept-new
    UserKnownHostsFile ~/.ssh/known_hosts" > ~/.ssh/config
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys

# write .sqliterc:
cat << 'EOF' >> ~/.sqliterc
.mode column
.headers on
EOF

# write .bashrc:
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

cd /workspace/repo
EOF
