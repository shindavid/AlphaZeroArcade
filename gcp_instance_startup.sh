#!/bin/bash

GCP=1

if [[ -b /dev/sdb ]] && ! mountpoint -q /persistent-disk; then
    sudo mkfs.ext4 -F /dev/sdb
    sudo mkdir -p /persistent-disk
    sudo mount /dev/sdb /persistent-disk
    echo "/dev/sdb /persistent-disk ext4 defaults 0 0" | sudo tee -a /etc/fstab
fi

if ! mountpoint -q /local-ssd; then
    sudo mkfs.ext4 -F /dev/nvme0n1
    sudo mkdir -p /local-ssd
    sudo mount /dev/nvme0n1 /local-ssd
fi
