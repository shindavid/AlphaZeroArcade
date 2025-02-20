#!/usr/bin/env bash

rm -rf ~/*
rm -rf ~/.*

sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo rm -rf /var/log/*.gz
sudo truncate -s 0 /var/log/syslog
