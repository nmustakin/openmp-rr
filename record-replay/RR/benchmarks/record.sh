#!/bin/bash

# Set up environment variables
export LD_LIBRARY_PATH=/data2/nmn/nvmetrics/build/lib:/usr/local/cuda-12.3/lib64:/data2/nmn/RR-back/openmp-rr/build/runtimes/runtimes-bins/openmp/runtime/src:/opt/rh/devtoolset-11/root/usr/lib64:/opt/rh/devtoolset-11/root/usr/lib:/opt/rh/devtoolset-11/root/usr/lib64/dyninst:/opt/rh/devtoolset-11/root/usr/lib/dyninst
export PATH=/usr/local/cuda-12.3/bin:/data2/nmn/RR-back/openmp-rr/build/bin:/opt/rh/devtoolset-11/root/usr/bin:/home/nmust004/anaconda3/condabin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/nmust004/.local/bin:/home/nmust004/bin

# Optionally enable devtoolset-11 if needed
source scl_source enable devtoolset-11

# Set OpenMP related environment variables
export LIBOMPTARGET_RR_DEVMEM_SIZE=4
export LIBOMPTARGET_RR_SAVE_OUTPUT=1
export OMP_TARGET_OFFLOAD=mandatory
export LIBOMPTARGET_NEXTGEN_PLUGINS=1
export LIBOMPTARGET_RECORD=1

# Execute the main application
./vAdd/vecAdd 10

