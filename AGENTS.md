# Record/replay build and experiments

This directory is the Git repository. For NVIDIA builds and optimizer runs,
follow `record-replay/RR/BUILD_AND_RUN.md`. This is the user's reusable workflow.

Build nvmetrics, LLVM/OpenMP, the tuning pass, and benchmarks inside
`/data2/nmn/rr_current.sif` using Apptainer with NVIDIA support and an explicit
`/data2/nmn` bind mount. Use GCC 12. Source `update-gcc.sh` and `update_path.sh`
from this repository root so their environment changes persist.

For miniFE, the benchmark module is `benchmarks/miniFE-omp/MINIFE.py`.
The user's current experiment uses BOMN, 30 initialization samples, 200 optimizer
samples, and the speedup objective. `record-replay/RR/run-minife-optimizer.sh`
encodes those settings. Preserve existing recordings and results when rerunning.
