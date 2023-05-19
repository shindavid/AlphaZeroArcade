#!/bin/bash -l
set -e

if [ "$#" != 1 ]; then
    echo "usage: $0 <hostname>"
    exit 1
fi
hostname=$1

just push $hostname
ssh $hostname ./AlphaZeroArcade/cloud/lambda/setup_local.sh
