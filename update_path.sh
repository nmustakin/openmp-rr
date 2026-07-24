#!/bin/bash

CURDIR=$(pwd)
LLVM_SRC_DIR=$(realpath $CURDIR)

export PATH=${LLVM_SRC_DIR}/build/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_SRC_DIR}/build/runtimes/runtimes-bins/openmp/runtime/src/:$LD_LIBRARY_PATH
export RR_TUNING_PASS=/data2/nmn/rr-metrics/openmp-rr/record-replay/Passes/build/turing.cs.ucr.edu/libTuneDevice.so
