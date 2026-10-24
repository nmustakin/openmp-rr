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

# AMD build and optimizer workflow

Tested on 8 × MI50 (gfx906) with ROCm 5.7.1, GCC 12, and a native build (no
container). `AMD_PORT_PLAN.md` explains the design and the MI50 measurements.

## Prerequisites

- Membership in the `render` and `video` groups for `/dev/kfd`. A session
  started before the change does not have the group yet; run GPU commands
  through `sg render -c '...'`. `sg` runs `/bin/sh` and drops
  `LD_LIBRARY_PATH`, so source `env_amd.sh` inside the command, under bash.
- `gcc-12` and `g++-12`.
- Python 3.9 or newer (`optimize.py` uses `argparse.BooleanOptionalAction`).
  Install the requirements without `pycuda`, plus `optuna==3.6.1` (and
  `xgboost==2.1.4` for the ranking strategies), for example into a directory
  on `PYTHONPATH`:
  `python3.10 -m pip install --target <pylib> -r <requirements without pycuda>`.

## Build

```bash
cd openmp-rr
./setup.sh amd 56   # LLVM + clang + lld + OpenMP into install_amd_60, writes env_amd.sh
source ./env_amd.sh
```

`setup.sh` also runs `pip`; to skip that, run its two CMake configurations by
hand with `-DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12`. The AMD build
needs `lld` (the plugin JIT-links the bitcode images), and the CUDA plugins
are skipped automatically when nvmetrics is not found.

## Run

```bash
sg render -c "bash -c 'source env_amd.sh; export PYTHONPATH=<pylib> ROCR_VISIBLE_DEVICES=0; \
  cd record-replay/RR && python3.10 optimize.py -b benchmarks/miniFE-omp/MINIFE.py -d amd --record'"
```

Then run `-prk` and the optimization as in the NVIDIA section, with `-d amd`.
The tuner selects the GPU with `ROCR_VISIBLE_DEVICES`.

## How AMD differs

- **Recording and replay without ASLR.** ROCm 5.7.1 cannot map memory at a
  fixed device address, so the replay region lands where the allocator puts
  it, and that depends on ASLR.
  - The tuner records under `setarch -R`.
  - `llvm-omp-kernel-replay` re-executes itself without ASLR
    (`LIBOMPTARGET_RR_KEEP_ASLR=1` opts out).
  - Recordings use a 12 GB region with 1 GB headroom
    (`LIBOMPTARGET_RR_DEVMEM_HEADROOM_MB`). Replays use the device memory
    size.
  - A replay that cannot place the region at the recorded address aborts.
    This happens with recordings made with ASLR on.
  - **Make exactly one GPU visible per replay process**, as during recording
    (for example `ROCR_VISIBLE_DEVICES=0`). With all 8 MI50s visible, the
    device address layout changes and the replay region no longer reaches the
    recorded address. For multi-GPU searches (`options['devices']`), make
    exactly the GPUs used visible; the forked workers restrict themselves to
    one each.
- **Timing.**
  - Timing replays use HSA dispatch timestamps.
  - The timing warm-up defaults to 100 ms, because the MI50 has no
    multi-step clock ramp.
  - Kernel energy is integrated from ROCm SMI's average power, because the
    MI50's energy counter is broken (`method: power_average` on the
    `ReplayEnergy` line).
- **Metrics.** Metrics replays run under `rocprof` with four counter passes,
  one replay process each. The counters are listed in
  `RRUtil/amd/rr_metrics.xml` and `Device.AMDGPU.MetricPasses`. The tuner
  stores derived metrics: `achieved_occupancy_pct`, `valu_busy_pct`,
  `dram_throughput_pct`, `l2_hit_pct`, `dram_read_bytes`,
  `dram_write_bytes` and `scratch_bytes`.
- **Launch bounds and registers.**
  - `MaxThreads` and `MinTeams` become `amdgpu-flat-work-group-size=1,N` and
    `amdgpu-waves-per-eu`.
  - Registers and spills come from `llc -pass-remarks-analysis=kernel-resource-usage`.
- **Not ported.** The `loads`, `stores` and `memory` objectives remain
  NVIDIA-only. The bottleneck-classification thresholds are still the T4's.
