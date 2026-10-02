#!/bin/bash

mkdir -p /tmp/gcc12-bin
ln -sf "$(command -v gcc-12)" /tmp/gcc12-bin/gcc
ln -sf "$(command -v g++-12)" /tmp/gcc12-bin/g++
export PATH="/tmp/gcc12-bin:$PATH"

gcc --version
g++ --version