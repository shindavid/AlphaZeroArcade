#!/bin/bash

if ! sudo blkid /dev/sdb; then
  # blkid returns non-zero if /dev/sdb has no identifiable filesystem
  echo "No filesystem detected on /dev/sdb, formatting..."
  sudo mkfs.ext4 /dev/sdb
fi

sudo mkdir -p /persistent-disk
sudo mount /dev/sdb /persistent-disk
if ! grep -q '/dev/sdb /persistent-disk' /etc/fstab; then
  echo "/dev/sdb /persistent-disk ext4 defaults 0 0" | sudo tee -a /etc/fstab
fi

# Wait a bit or check the device is present
for i in {1..10}; do
  if [ -b /dev/nvme0n1 ]; then
    break
  fi
  sleep 1
done

if [ -b /dev/nvme0n1 ]; then
    if ! mountpoint -q /local-ssd; then
        sudo mkdir -p /local-ssd
        # If no filesystem yet, format it
        if ! sudo blkid /dev/nvme0n1; then
            sudo mkfs.ext4 -F /dev/nvme0n1
        fi
        sudo mount /dev/nvme0n1 /local-ssd
        # (Optional) Add to /etc/fstab
        # echo "/dev/nvme0n1 /local-ssd ext4 defaults 0 0" | sudo tee -a /etc/fstab
    fi

    # Create a 32 GB swapfile
    if [ ! -f /local-ssd/swapfile ]; then
      sudo fallocate -l 32G /local-ssd/swapfile
      sudo chmod 600 /local-ssd/swapfile
      sudo mkswap /local-ssd/swapfile
      sudo swapon /local-ssd/swapfile
    else
      # In case it's already there but not active
      sudo swapon /local-ssd/swapfile || true
    fi
else
    echo "No local SSD found at /dev/nvme0n1"
fi
