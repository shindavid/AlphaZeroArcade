#!/usr/bin/env bash

sudo apt-get update && sudo apt-get upgrade -y

# Set up 32G swap
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# NVIDIA Driver
sudo apt-get install -y linux-headers-$(uname -r)
curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin \
    | sudo tee /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
yes '' | sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update
sudo apt-get -y install cuda-drivers

sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Docker
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor \
    | sudo tee /etc/apt/keyrings/docker.gpg > /dev/null
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
   https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) stable" \
   | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor \
    | sudo tee /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg > /dev/null

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Required python packages
sudo apt-get install -y python3 python3-pip
pip3 install --upgrade packaging

# Some additional defaults in /etc/skel/.bashrc
#
# -Override PS1 to be more informative
# -If root owns /persistent-disk, sudo chown it to $USER
# -If root owns /local-ssd, sudo chown it to $USER
sudo tee -a /opt/project/bashrc_extras > /dev/null << 'EOF'

##########################
# AlphaZeroArcade extras #
##########################

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

# chown /persistent-disk and /local-ssd if root owns them
if [ "$(stat -c %U /persistent-disk)" = "root" ]; then
    sudo chown $USER:$(id -gn $USER) /persistent-disk
fi

if [ "$(stat -c %U /local-ssd)" = "root" ]; then
    sudo chown $USER:$(id -gn $USER) /local-ssd
fi

EOF
