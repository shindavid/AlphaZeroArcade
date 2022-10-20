#!/usr/bin/env bash

cd "$(dirname "$0")"

cmake CMakeLists.txt -B target
cd target
make
