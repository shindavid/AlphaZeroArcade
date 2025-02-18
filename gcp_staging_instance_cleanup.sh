#!/usr/bin/env bash

# This is to be run as root, with the user passed as an argument

USER=$1

userdel -r $USER

apt-get clean
rm -rf /var/lib/apt/lists/*
rm -rf /var/log/*.gz
truncate -s 0 /var/log/syslog

rm -rf ~/.ssh
rm -f ~/.bash_history
