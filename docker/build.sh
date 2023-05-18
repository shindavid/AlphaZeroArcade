#!/bin/bash
set -e
THISDIR=$(realpath $(dirname $0))
cd $THISDIR

docker build --progress=plain --no-cache  -t alphazeroarcade -f Dockerfile .
