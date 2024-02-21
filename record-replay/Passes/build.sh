#!/bin/bash
#
host=$(hostname)
host=${host//[0-9]/}

mkdir -p build/${host}
echo $host
cd build/${host}

cmake \
    -DLT_LLVM_INSTALL_DIR=/data2/nmn/record-replay/build/ \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    ../../

make
cd -
