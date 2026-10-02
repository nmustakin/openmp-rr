# NVIDIA build and optimizer workflow

Run inside `apptainer shell --nv --bind /data2/nmn:/data2/nmn /data2/nmn/rr_current.sif`.
The repository is `/data2/nmn/rr-metrics/openmp-rr`.

```bash
cd /data2/nmn/rr-metrics/openmp-rr
source ./update-gcc.sh

cd /data2/nmn/nvmetrics
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake -S . -B build
make -C build

cd /data2/nmn/rr-metrics/openmp-rr/build
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 cmake \
  -DLLVM_ENABLE_PROJECTS=clang -DLLVM_ENABLE_RUNTIMES=openmp \
  -DLIBOMPTARGET_ENABLE_DEBUG=ON -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DBUILD_SHARED_LIBS=ON -DLLVM_APPEND_VC_REV=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_CXX_FLAGS=-Wno-address ../llvm
make -j40

cd /data2/nmn/rr-metrics/openmp-rr/record-replay/Passes
cmake -S . -B build/turing.cs.ucr.edu \
  -DLT_LLVM_INSTALL_DIR=/data2/nmn/rr-metrics/openmp-rr/build \
  -DLLVM_DIR=/data2/nmn/rr-metrics/openmp-rr/build/lib/cmake/llvm \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12
make -C build/turing.cs.ucr.edu

cd /data2/nmn/rr-metrics/openmp-rr
source ./update_path.sh
export RR_TUNING_PASS="$PWD/record-replay/Passes/build/turing.cs.ucr.edu/libTuneDevice.so"
export LD_LIBRARY_PATH="/data2/nmn/nvmetrics/build/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

cd record-replay/RR/benchmarks/miniFE-omp
make all
cd ../..
python3 optimize.py -b benchmarks/miniFE-omp/MINIFE.py -d nvidia --record
python3 optimize.py -b benchmarks/miniFE-omp/MINIFE.py -d nvidia -prk
python3 optimize.py -b benchmarks/miniFE-omp/MINIFE.py -d nvidia \
  --kernel all --scenario BOMN --init-samples 30 --opt-samples 200 \
  --objective speedup
```

Use a kernel name from `-prk` instead of `all` to optimize one kernel.
The default objective is `memory`; add `--objective speedup` for timing optimization.
miniFE defaults to a 200 × 200 × 200 problem.
`--record` reuses an existing checkpoint: preserve or move old checkpoints and
recorded execution artifacts before requesting a fresh recording.

`update_path.sh` must be sourced from the repository root because it uses the
current directory. The existing `Passes/build.sh` points at an old
`/data2/nmn/RR-back` checkout; the explicit CMake command above uses this repository.
Use ordinary ASCII `--` options and absolute paths (no leading `./data2`).

For repeated speedup runs after building, run `bash run-minife-optimizer.sh`
from this directory inside the container. Pass a recorded kernel name as the
first argument to restrict the run. The script defaults to 30 initialization and 200
optimizer samples per kernel. Optional second and third arguments override
those counts, for example `bash run-minife-optimizer.sh all 5 30`.
