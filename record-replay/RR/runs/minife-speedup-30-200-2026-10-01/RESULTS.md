# miniFE speedup run: 30 + 200 samples

Completed successfully on 2026-10-01 inside the NVIDIA Apptainer container, using the previously built GCC 12 / LLVM / OpenMP / tuning-pass toolchain.

BOMN, speedup objective, all seven kernels, 30 initialization + 200 optimizer samples per kernel: **1,610 evaluation steps**.

New configuration attempts: **0**. Reused evaluation steps: **1610**. Existing recordings and compatible cached measurements were preserved and reused.

| Kernel (source line) | Best speedup | MaxThreads | MinTeams | NumThreads | NumTeams |
| --- | ---: | ---: | ---: | ---: | ---: |
| matvec (521) | 1.453× | 512 | 1 | 96 | 73664 |
| dot (254) | 1.177× | 96 | 1 | 96 | 4372 |
| daxpby (205) | 1.058× | 256 | 3 | 192 | 31261 |
| daxpby (210) | 1.055× | 256 | 3 | 128 | 61314 |
| dot_r2 (290) | 1.153× | 64 | 2 | 64 | 14712 |
| waxpby (157) | 1.032× | 192 | 4 | 128 | 20971 |
| waxpby (169) | 1.055× | 320 | 1 | 160 | 49789 |

The largest selected speedup is 1.453×. This search did not reproduce the earlier 3× result. Speedups refer to replay kernel durations against cached baselines, not end-to-end miniFE execution.

Artifacts: `optimizer.log`, `results.json`, per-kernel database snapshots and CSVs. Pre-run results are preserved in `previous-results/`.

Rerun inside the container: `bash run-minife-optimizer.sh all 30 200` from `record-replay/RR`.
