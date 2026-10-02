# miniFE speedup run — 2026-10-01

Completed successfully inside Apptainer with GCC 12 on Tesla T4. nvmetrics, LLVM/Clang/OpenMP, the tuning pass, and miniFE built successfully.

BOMN; speedup objective; 5 initialization + 30 optimizer samples per kernel (245 evaluation steps total).

| Kernel (source line) | Best speedup | MaxThreads | MinTeams | NumThreads | NumTeams | New attempts | Reused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matvec (521) | 1.384× | 32 | 1 | 32 | 87781 | 18 | 17 |
| dot (254) | 1.108× | 32 | 21 | 32 | 5367 | 24 | 11 |
| daxpby (205) | 1.056× | 96 | 10 | 96 | 45930 | 24 | 11 |
| daxpby (210) | 1.053× | 960 | 1 | 128 | 49458 | 14 | 21 |
| dot_r2 (290) | 1.072× | 288 | 3 | 32 | 19486 | 24 | 11 |
| waxpby (157) | 1.037× | 576 | 1 | 128 | 28292 | 24 | 11 |
| waxpby (169) | 1.052× | 384 | 2 | 192 | 41295 | 22 | 13 |

The previous 3× speedup was not reproduced in this run. Existing recordings and compatible cached measurements, including baselines, were reused. These results describe replay kernel speedups, not an end-to-end miniFE speedup.

Artifacts: `build-and-optimizer.log`, `results.json`, per-kernel database snapshots and CSVs. Previous CSVs are preserved in `previous-summaries/`; the original checkpoint is saved as `MINIFE.ckpt.json`.

Workflow: [BUILD_AND_RUN.md](../../BUILD_AND_RUN.md). Runner: [run-minife-optimizer.sh](../../run-minife-optimizer.sh).
