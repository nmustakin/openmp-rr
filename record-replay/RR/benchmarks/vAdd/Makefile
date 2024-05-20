LECPP ?=clang++
LEOPT ?=-O3
#LEFLAGS ?=-fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_75 -fopenmp-offload-mandatory --libomptarget-nvptx-bc-path=/data2/nmn/llvm-new/llvm-project/build/runtimes/runtimes-bins/offload/DeviceRTL/libomptarget-nvptx-sm_75.bc -foffload-lto -Wl,--embed-bitcode
LEFLAGS ?=-lnv_metrics -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target=nvptx64 -march=sm_75 -fopenmp-offload-mandatory --libomptarget-nvptx-bc-path=/data2/nmn/RR-back/openmp-rr/build/runtimes/runtimes-bins/offload/DeviceRTL/libomptarget-nvptx-sm_75.bc -foffload-lto -Wl,--embed-bitcode
INCLUDE_FLAGS ?=-I/data2/nmn/nvmetrics/lib -L/data2/nmn/nvmetrics/build/lib
CPPFLAGS=${LEOPT} ${INCLUDE_FLAGS} ${LEFLAGS} -std=c++14
LDFLAGS?=
BUILDDIR?=./
EXEDIR?=${BUILDDIR}
RBUILDDIR=$(shell realpath ${BUILDDIR})
REXEDIR=$(shell realpath ${EXEDIR})
SRCS=vecAdd.cpp
OBJS=$(SRCS:%.cpp=$(RBUILDDIR)/%.o)
BINS=${REXEDIR}/vecAdd

#This is done for AMD systems which can
#link in the libomptarget from the system installation
compDIR=$(shell dirname $(shell dirname $(shell which ${LECPP})))
LDFLAGS+=-L${compDIR}/lib -L${compDIR}/lib64

.PHONY = all info

all: ${BINS}

${BINS}: $(OBJS)
	${LECPP} -v ${LDFLAGS} ${CPPFLAGS} ${OBJS} -o $@

${RBUILDDIR}/%.o: %.cpp
	$(LECPP) -v $(CPPFLAGS) -c $< -o $@

clean:
	@echo "Cleaning up ..."
	rm -rvf  $(OBJS) ${BINS}

info:
	$(info $$BINS is [${BINS}])
	$(info $$SRCS is [${SRCS}])
	$(info $$OBJS is [${OBJS}])
	$(info $$LECPP is [${LECPP}])
	$(info $$LEOPT is [${LEOPT}])
	$(info $$LEFLAGS is [${LEFLAGS}])
	$(info $$LDFLAGS is ${LDFLAGS})

