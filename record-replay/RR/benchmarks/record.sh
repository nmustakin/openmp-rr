#!/bin/bash

LIBOMPTARGET_RR_DEVMEM_SIZE=15 LIBOMPTARGET_RR_SAVE_OUTPUT=1 OMP_TARGET_OFFLOAD=mandatory LIBOMPTARGET_NEXTGEN_PLUGINS=1 LIBOMPTARGET_RECORD=1 nvprof ${@:1}
