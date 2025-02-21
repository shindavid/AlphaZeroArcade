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

# Wait for local SSD devices to appear
for i in {1..20}; do
  if lsblk | grep -q "nvme"; then
    break
  fi
  sleep 1
done

# Identify local SSD devices (assuming they appear as nvme0n1, nvme1n1, nvme2n1)
LOCAL_SSD_DEVICES=$(lsblk -dn -o NAME | grep '^nvme' | sed 's/^/\/dev\//')

echo "Detected local SSD devices: $LOCAL_SSD_DEVICES"

# Create RAID 0 array over all detected local SSD devices if not already done.
if [ ! -e /dev/md0 ]; then
  NUM_DEVICES=$(echo $LOCAL_SSD_DEVICES | wc -w)
  echo "Creating RAID 0 over $NUM_DEVICES devices..."
  sudo mdadm --create /dev/md0 --level=0 --raid-devices=$NUM_DEVICES $LOCAL_SSD_DEVICES --force
  sleep 10  # allow RAID array to initialize
  sudo mkfs.ext4 -F /dev/md0
fi

# Mount RAID device to /local-ssd
sudo mkdir -p /local-ssd
sudo mount /dev/md0 /local-ssd

# (Optional) Update /etc/fstab for persistence (ephemeral local SSDs don't persist reboots)
if ! grep -q '/dev/md0 /local-ssd' /etc/fstab; then
  echo "/dev/md0 /local-ssd ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
fi

# (Optional) Create a 32GB swapfile on the RAID array if needed
if [ ! -f /local-ssd/swapfile ]; then
  sudo fallocate -l 32G /local-ssd/swapfile
  sudo chmod 600 /local-ssd/swapfile
  sudo mkswap /local-ssd/swapfile
  sudo swapon /local-ssd/swapfile
fi
