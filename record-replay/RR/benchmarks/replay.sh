#!/bin/bash

LIBOMPTARGET_RR_DEVMEM_SIZE=8 OMP_TARGET_OFFLOAD=mandatory LIBOMPTARGET_NEXTGEN_PLUGINS=1 nvprof llvm-omp-kernel-replay ${@:1}
