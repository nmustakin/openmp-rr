# Porting the measurement-v3 tuner to AMD GPUs

Branch: `measurement-v3-amd` (from `measurement-v3` @ `2846c97e3`).
Target machine: 8x AMD Radeon Instinct MI50 16GB (Vega 20, `gfx906`), ROCm 5.2.0 /
5.7.1 (default `/opt/rocm`) / 6.0.0, GCC 9.4 system compiler.

## What has to be ported

The paper (`record-replay/paper/autotuning_study.tex`) shows that measurement,
not the search algorithm, limited the original tuner. Most of the work is
therefore in the measurement path, and almost all of it is NVIDIA-only today.

| Piece (paper section) | NVIDIA implementation | Status on AMD |
|---|---|---|
| Record / replay, `--batch`, `LIBOMPTARGET_RR_DEVICE_RESTORE` (2.4, 9.6) | vendor-neutral (`PluginInterface.cpp`, `omptarget.cpp`, `llvm-omp-kernel-replay.cpp`) | Works, but AMDGPU has no `mMap` (fixed VA uses the allocation heuristic) and no device-to-device `dataExchange`, so `DEVICE_RESTORE` silently restores from host |
| v3 timing: warm-up, re-warm, repeats, median (4) | `cuEvent` in `plugins-nextgen/cuda/src/rtl.cpp` `launchImpl` | Missing |
| Kernel energy, SM clock (4) | NVML via `dlopen` (`rrnvml`) | Missing |
| Hardware metrics, `EXTRA_METRICS`, metrics warm-up (6, 8) | nvmetrics (CUPTI/perfworks) | Missing |
| `kernel_activities.csv` (grid, block, regs, smem, ...) | CUPTI activity API | Missing; Python always reads this file |
| Exclusive-GPU guard, device mapping, energy (4) | `pynvml` in `Benchmark.py`, `optimize.py` | Missing; `pynvml` is imported and initialised at module load |
| Compile-time features: regs, spills, occupancy, waves (6.3) | `llc` + `ptxas -v`, T4 constants in `Search.py` | `AMDGPU.deviceCompile` returns `None`; constants are T4 |
| Bottleneck classification (8) | perfworks metric names and T4-derived thresholds | Needs AMD counters and recalibrated thresholds |
| Launch-bound pass | `nvvm.annotations` (`maxntidx`, `minctasm`) | `AMDGPUAttributePass` exists but is untested; Python side has bugs (below) |
| Benchmarks | Makefiles target `nvptx64`/`sm_75` and link `-lcupti -lnv_metrics` | AMD lines exist but are commented out or stale (`gfx90a`) |

Keep the plugin's environment variable names (`LIBOMPTARGET_RR_*`) and stdout tags
(`||ReplayTiming||`, `||ReplayEnergy||`, `||ReplayClock||`, `||NVMetric||`,
`||MetricsClock||`, `||ReplayConfig||`) identical, so the Python parser needs no
protocol changes. Rename the tags later if wanted.

## Phase 0: environment (blockers)

1. **GPU access.** `nmust004` is not in the `render` group, so `/dev/kfd` is
   unusable and `rocminfo` fails. An administrator has to add the user to
   `render` (and `video`). Nothing below can be tested until this is done.
2. **ROCm version.** This determines which APIs the plugin can use:
   - 5.7.1 has rocprofiler v1/v2 (`librocprofiler64`, `gfx906` in
     `gfx_metrics.xml`) but no `hsa_amd_vmem_*`.
   - 6.0.0 has `hsa_amd_vmem_*` (fixed-VA mapping) but no rocprofiler or
     roctracer installed.
   - rocprofiler-sdk (ROCm 6.2+) is unlikely to support `gfx906`.

   Recommendation: start on **5.7.1**. Use the existing allocation heuristic for
   fixed-VA replay; the SC23 AMD results (Tioga) used that path. Move to 6.0 plus
   VMM only if replays fail to map at the recorded address.
3. **Compiler.** GCC 12 is not installed (only gcc-8 and gcc-9), and there is no
   Apptainer container on this host. Either get GCC 12 through spack
   (`spack_env/rr_amd.yaml`: add `gcc@12`) or try GCC 9.4, which LLVM 16
   accepts. As in the RTX 6000 replication, build natively.
4. **Python.** Add `env_amd.sh`, written by `setup.sh amd`. Optuna 3.6.1 and
   XGBoost go in a local `pylib`, as in the studies. Make `pycuda` and `pynvml`
   optional (see Phase 3).

## Phase 1: build and baseline record/replay

1. Run `./setup.sh amd <threads>`, which builds `install_amd_60` with the
   AMDGPU target and `libTuneDevice.so`.
2. Build `vAdd`, then miniFE and LULESH, for `amdgcn-amd-amdhsa` /
   `-march=gfx906`, with the same `-foffload-lto -Wl,--embed-bitcode` flags.
   Drop `-lcupti -lnv_metrics`.
3. Record and replay one kernel each with `LIBOMPTARGET_RR_SAVE_OUTPUT` plus
   verification. Then check `--batch` with and without `DEVICE_RESTORE`.
4. Measure the MI50 clock ramp the same way as Fig. `clock_ramp`: per-launch
   times in cold processes vs. warmed processes. This decides the warm-up
   defaults for AMD, which may differ from 1000 ms / 3 launches.

Exit criterion: replays of LULESH and miniFE kernels verify, and batch replay
works.

### Phase 1 findings (2026-10-20, vAdd on MI50)

The build needs:
- `lld`. The AMDGPU plugin JIT-compiles IR images and links them with `lld`.
  `setup.sh` / `setup_spack.sh` now build `clang;lld` for `amd`.
- No hard dependency on nvmetrics in the CUDA plugins. They now look for it
  under `NVMETRICS_ROOT` / `CUPTI_ROOT`, and are skipped if it is missing.

Running:
- `sg render` (used until the session picks up the new group) strips
  `LD_LIBRARY_PATH`. Source `env_amd.sh` inside the `sg` command.
- **Fixed-VA replay needs ASLR off.** Device allocations land at
  ASLR-dependent addresses. With `setarch x86_64 -R` on both the recording run
  and the replay, the recorder and `llvm-omp-kernel-replay` get the same base
  (`0x7ffb17c00000` on MI50).
  - With ASLR off: replay verifies at every launch configuration tried, and
    `--batch` works with and without `DEVICE_RESTORE`.
  - With ASLR on: 3 of 4 replays page-faulted.
  - This replaces `hsa_amd_vmem_*` (ROCm 6.0 only) for now. Still to be
    confirmed on LULESH/miniFE, which allocate more before device init.
- `initMemoryHeuristic` bugs:
  - "Cannot Replay requested address" only logs. The replay then continues
    with a null device pointer and faults, instead of aborting.
  - The size adjustment can grow `Size` beyond the allocation.
- Planned implementation:
  - `llvm-omp-kernel-replay` re-execs itself with
    `personality(ADDR_NO_RANDOMIZE)`.
  - The Python recorder prefixes `setarch -R` on AMD.

  Both are done (`87e10546e`). In the same commit:
  - The heuristic recording region keeps 1 GB of headroom
    (`LIBOMPTARGET_RR_DEVMEM_HEADROOM_MB`). The replay tool's block ends about
    24 MB lower than the application's, which made every miniFE replay fault.
  - The region allocator is bounds-checked.
  - AMD records with 12 GB and replays with 15 GB.
- Result: all 17 LULESH and 7 miniFE kernels replay and verify on MI50.
  Recordings are in `~/rr-amd-work/{lulesh,minife}/rec`.

### Phase 2 progress (2026-10-21)

Done:
- **Timing** (`a19f1c397`). HSA dispatch timestamps, same env vars and
  output lines as CUDA.
- **Energy** (`f19fb0e8c`). The MI50 energy counter reads a constant
  `0xFFFFFFFF`. Energy is instead integrated from `rsmi_dev_power_ave_get`
  (updates every ~8 ms, 1 W resolution), sampled every 2 ms over exactly the
  launch window. The line is tagged `method: power_average`.
  - LULESH l1721, two processes: 4.227 / 4.223 ms and 0.680 / 0.679 J per
    launch, at about 160 W.
- **Device-to-device restore** (`afb67290b`). An 8-configuration batch on
  l649 took 7.6 s instead of 11.9 s. `--verify` now compares only the
  recorded length; the restore backup made every restore replay fail.

**MI50 clock ramp** (LULESH l1721; 8 cold processes after 10 s idle vs 8
processes with 3 s warm-up; data in `~/rr-amd-work/clock-ramp/l1721.csv`):
- Only the first launch of a cold process is slow (4.55–5.20 ms vs 4.22 ms).
  From the second launch on, every process is at 1725 MHz.
- Cross-process CV of the median is 0.03% cold and 0.04% warm. On the T4 it
  was 8.4%.
- Short kernels need longer to settle:
  - LULESH l2103 (0.42 ms): about 20 launches.
  - miniFE l254 (0.79 ms): about 6 launches.
- Warm-up sweep on l2103 (interleaved, 6 rounds, one GPU;
  `sweep-l2103.csv`):

  | Warm-up | Median vs 0 ms | CV |
  |---|---|---|
  | 0 ms (3 launches only) | — | 0.43% |
  | 30 ms | −1.13% | 0.21% |
  | 100 ms | −1.20% | 0.09% |
  | 300 ms | −1.20% | 0.11% |
  | 1000 ms | −1.20% | 0.08% |
  | 3000 ms | −1.23% | 0.14% |

- **AMD default is therefore 100 ms** (`Device.AMDGPU.TimingWarmupMs`;
  `RR_TIMING_WARMUP_MS` still overrides). NVIDIA keeps 1000 ms.
- Open: in the cold-vs-warm ramp on GPU 1, warmed l2103 processes were 1%
  *slower* than cold ones, at the same reported clock. The sweep on GPU 3 did
  not reproduce this. Check GPU-to-GPU differences in phase 5.

**Metrics** (`a0a7ad346`): rocprof runs four counter passes, one replay
process each, so every pass starts from the recorded inputs. Plugin-side
rocprofiler collection was dropped: kernel packets have no barrier bit, so
its start/stop packets would not be ordered around the kernel. rocprof also
writes the `kernel_activities.csv` the tuner reads. Validation:
- vAdd: DRAM reads 128 MB and writes 64 MB, both exact.
- Waves per CU: 1.00 for 60 single-wave teams; 15.4 on l649 for 16 expected.

### Phase 3 progress

- `RRUtil/GPUMonitor.py`. The NVML monitor keeps the previous behavior. The
  AMD monitor uses:
  - KFD sysfs queues for the exclusive-GPU guard;
  - ROCm order plus `ROCR_VISIBLE_DEVICES` for ordinal mapping;
  - ROCm SMI for energy. Process-level board energy is NaN on MI50.
- No pynvml/pycuda import on the AMD path.
- Python: the tuner needs Python 3.9+ (`argparse.BooleanOptionalAction`); the
  system Python is 3.8.
  - Use `python3.10` with
    `PYTHONPATH=~/rr-amd-work/pylib`.
  - That pylib holds the requirements without pycuda, plus optuna 3.6.1 and
    xgboost 2.1.4.
- **Compile path** (`3bd6a4d70`). `Device.AMDGPU.deviceCompile` sets the same
  attributes as OpenMPOpt's `--bo-omp-autotune` (`amdgpu-flat-work-group-size=1,N`,
  `amdgpu-waves-per-eu`, now clamped to [1, 10]). VGPRs, SGPRs, LDS and spills
  come from llc's `kernel-resource-usage` remarks.
- **Search model** (`3bd6a4d70`). `Search.set_device` replaces the T4
  constants. The gfx9 VGPR occupancy model matches llc on all 17 LULESH
  kernels.
  - l844: 64 VGPRs and spills at the default bounds, 203 VGPRs and no spills
    at MaxThreads ≤ 256.
  - The table has 122 pairs and takes 156 s, with llc results cached per
    binary.
- **Metric roles and replay** (`3fbd3eee6`, `98d69f20f`).
  - `Search.METRIC_ROLES` maps the classification and steering roles to AMD
    metrics.
  - Replays and batch replays use the device replay size (15 GB); with 4 GB
    every AMD replay failed.
  - `Device.blockFits` replaces the registers × threads check. The old
    `MaxRegBlock = 0` rejected every AMD configuration.
- **End-to-end** on vAdd with `optimize.py -d amd`:
  - `--record` works.
  - A `BOMN` run completes.
  - `Kernel.Search` `adaptive` classifies from rocprof metrics and runs
    trees-meta with the compile table.
  - `tpe-batch` screens with batch replay.

### Phase 4 progress

- The HeCBench wrappers use `HECBENCH_ROOT` (`acb804d9b`).
- `benchmarks/hecbench/make_rr_amd.sh` writes `Makefile.rr` from HeCBench's
  own `Makefile.aomp`; all 152 wrapped applications have one. HeCBench is
  cloned at `~/HeCBench` (`f0c817aa3`) and `Makefile.rr` is generated there.
- XSBench builds, records and replays with verified output.
- Not yet done:
  - the testSNAP/miniBUDE data from HeCBench's data store;
  - miniWeather's MPI (stub or OpenMPI);
  - the input sizes chosen for the T4 (0.1–100 ms kernels), which need
    re-checking on MI50.

### Classification thresholds for MI50 (2026-10-21)

**Method.** For all 24 LULESH and miniFE kernels (the tuner's own
recordings), `~/rr-amd-work/calib/landscape.py` measured:
- three full warm baselines with rocprof metrics;
- an exhaustive batch landscape: every binary class × threads (multiples
  of 64) × 9 teams fractions, about 100–900 configurations per kernel, on 8
  GPUs in about 13 minutes;
- a full measurement of the best configuration.

`calib/analyze.py` labels each kernel by what that configuration changed:
- *none*: ≤ 1.05×;
- *register*: spills removed or a different binary;
- *locality*: ≥ 25% fewer DRAM reads with the same binary;
- *other*: runtime parameters only.

| App | Kernel | Validated best | Default → best binary (VGPRs, spill) | DRAM % | Occ % | VALU % |
|---|---|---|---|---|---|---|
| LULESH | l333 | 1.57× | (29, 0) → (24, 0) | 19 | 19 | 2.6 |
| LULESH | l584 | 3.03× | (64, 696) → (121, 0) | 32 | 37 | 3.2 |
| LULESH | l649 | 1.82× | (30, 0) → (30, 0) | 57 | 38 | 1.6 |
| LULESH | l844 | 4.32× | (64, 1824) → (203, 0) | 60 | 38 | 3.2 |
| LULESH | l1029 | 1.78× | (30, 0) → (24, 4) | 61 | 39 | 1.7 |
| LULESH | l1080 | 4.51× | (64, 1056) → (200, 0) | 25 | 38 | 0.9 |
| LULESH | l1201 | 1.34× | (43, 0) → (32, 16) | 5 | 6 | 4.8 |
| LULESH | l1235 | 1.51× | (26, 0) → (24, 0) | 18 | 31 | 6.6 |
| LULESH | l1279 | 1.07× | (31, 0) → (31, 0) | 22 | 36 | 6.4 |
| LULESH | l1312 | 1.08× | (26, 0) → (24, 0) | 5 | 4 | 0.8 |
| LULESH | l1340 | 1.02× | (26, 0) → (26, 0) | 25 | 38 | 4.7 |
| LULESH | l1377 | 1.01× | (26, 0) → (24, 0) | 27 | 38 | 4.4 |
| LULESH | l1721 | 4.63× | (64, 1088) → (128, 56) | 53 | 36 | 7.1 |
| LULESH | l1841 | 1.20× | (36, 0) → (35, 0) | 31 | 18 | 4.2 |
| LULESH | l1913 | 3.43× | (64, 472) → (178, 0) | 50 | 35 | 8.2 |
| LULESH | l2103 | 1.12× | (52, 0) → (46, 0) | 27 | 38 | 15.3 |
| LULESH | l2372 | 1.08× | (64, 8) → (64, 8) | 28 | 25 | 16.2 |
| miniFE | l157 | 1.25× | (24, 0) → (24, 0) | 27 | 37 | 14.4 |
| miniFE | l169 | 1.11× | (29, 0) → (29, 0) | 31 | 23 | 7.4 |
| miniFE | l205 | 1.06× | (26, 0) → (26, 0) | 25 | 36 | 9.1 |
| miniFE | l210 | 1.15× | (26, 0) → (26, 0) | 24 | 35 | 8.9 |
| miniFE | l254 | 1.18× | (43, 0) → (43, 0) | 12 | 12 | 9.7 |
| miniFE | l290 | 1.19× | (43, 0) → (43, 0) | 7 | 12 | 9.6 |
| miniFE | l521 | 1.10× | (60, 0) → (60, 0) | 57 | 38 | 6.2 |

**Findings.**
- The MI50 gains far more than the T4. LULESH l1721, l1080, l844, l1913 and
  l584 gain 3.0–4.6×. l584, l1721 and l1913 were compute-saturated, with
  nothing to gain, on the T4.
- All five spill at the default bounds. A 1024-thread work-group caps them
  at 64 VGPRs. MaxThreads ≤ 512 gives 121–203 VGPRs and no spills.
- Their DRAM reads also fall 80–88%, because the spill traffic disappears.
  The locality label therefore has to come after the spill test.
- No kernel is compute-saturated: VALU busy is at most 16%. Half-rate FP64
  makes these kernels memory- and latency-bound.
- The two kernels with nothing to gain (l1340, l1377) cannot be separated
  from 1.06–1.25× kernels by baseline metrics.

**Thresholds** (`Search.CLASS_THRESHOLDS['amd']`):
- Spill test first, using `compile_spill_bytes`: the profiled binary's
  compile-time spills, now added to every sample's metrics. `scratch_bytes`
  also counts stack, such as the 16 KB per lane of OpenMP reductions.
- Locality: DRAM throughput > 45%, with no occupancy condition.
- Register otherwise: occupancy < 20%.
- The compute rule is unchanged and does not fire.

**Agreement with the labels:** 17 of 24, against 11 for the T4 rules, which
classify every MI50 kernel as register. The disagreements:
- the two no-gain kernels, classified *other* (normal TPE search);
- the reductions l254 and l290, classified *register* (as on the T4);
- l2372, which has 8 bytes of spill;
- l1235 and l2103, small register-labeled gains classified *other*.

The thresholds were fit and evaluated on the same 24 kernels from two
applications, so HeCBench is the out-of-sample check.

### Phase 5.1: measurement validation (paper §4) on MI50 (2026-10-21)

**Setup** (`~/rr-amd-work/phase5/validate_measurement.py`, results in
`phase5/val/*.csv`, summary from `analyze_validation.py`):
- Six LULESH kernels (l649, l844, l1029, l1721, l1913, l2103).
- Five configurations each: baseline, full block, and the three best of the
  calibration landscape.
- Six rounds in randomized order, one GPU per kernel.
- v2: one launch per process, no warm-up, process energy.
- v3: 3 launches + 100 ms warm-up, median of 10 launches, kernel energy.

| Quantity | v2 | v3 | T4 (paper) v2 → v3 |
|---|---|---|---|
| Cross-process CV of duration (median) | 9.3% | 0.30% | 8.4% → 0.23% |
| Std of log speedup across rounds | 0.133 | 0.004 | 0.114 → 0.003 |
| Kendall τ between rounds | 0.60 | 1.00 | 0.24 → 0.77 |
| Spearman ρ, duration vs energy | −0.25 to 0.75 | 0.75 to 1.00 | −0.9..0.8 → 0.8..1.0 |
| Wall time per replay process | 2.9 s | 3.4 s | 2.9 → 4.3 s |

- The legacy noise on MI50 comes from the slow first launch of a process
  (8–130% slower), not from a clock ramp.
- The landscape winners replicate in the randomized rounds: l1721 4.63×,
  l844 4.33×, l1913 3.43×, l649 1.85×, l1029 1.76×, l2103 1.13×.

### Phase 5.2: final-tuner comparison on MI50 (2026-10-21)

**Setup.**
- Recordings: the tuner's own LULESH and miniFE recordings.
- Kernels: the 10 LULESH kernels of the paper's online studies plus all 7
  miniFE kernels, 3 seeds each, on GPUs 0–5. Drivers are in
  `~/rr-amd-work/phase5/`.
- Strategies:
  - `rank-confirm`: rank-batch, patience 2, batch confirmation, refinement,
    300 s cap. The ranking model is trained on `pool.csv`: the calibration
    landscapes, leaving the target kernel out, with classes from the first
    MI50 rules.
  - `tpe-batch`: 300 s cap.
  - `adaptive`: 30 full evaluations, 8 initial; the paper used 60.
- Compile tables are cached from the calibration, which is not counted in the
  times below.
- Ground truth: every shipped configuration re-measured with the baseline in
  5 randomized rounds (`validate_shipped.py`, `truth/`).

| Strategy | True speedup (geo-mean, 95% CI) | Share of attainable gain | End-to-end per kernel | vs tpe-batch |
|---|---|---|---|---|
| rank-confirm | 1.84× | 0.93 | 201 s | 1.032 [1.005, 1.065] |
| tpe-batch | 1.78× | 0.87 | 436 s | — |
| adaptive | 1.55× | 0.65 | 1381 s | 0.87 [0.81, 0.92] |

True speedups are geometric means; only ratios carry CIs.

**Findings.**
- The final tuner transfers to MI50. It is about 2.2× faster than
  tpe-batch, and on all 17 kernels its shipped configurations are better.
- On the T4's five-kernel set (l649, l844, l1201, miniFE l290, l521) the two
  tie on MI50 too: ratio 1.057 [0.993, 1.132], 242 s vs 490 s. The
  significant edge comes from the larger kernel set, not from a different
  tuner behavior.
- Its advantage comes mostly from l649, l1029, l254 and l290.
- `adaptive` with full evaluations is far slower here: every evaluation runs
  four rocprof replay processes. Its 30-evaluation budget also falls well
  short of the batch strategies.

### Phase 5.3: HeCBench on MI50

- **Built and recorded** all ten applications of the paper (51 kernels):
  XSBench, RSBench, SimpleMOC, SU3, HACCmk, AMGmk, minisweep, sw4ck, S3D,
  miniWeather.
- **Filtered** (paper's rules) to **22 kernels**, the paper had 31 on the
  T4:
  - S3D's 13 hand-indexed teams regions have no trip count, as on the T4.
  - 11 kernels run under 0.1 ms on MI50, 8 of them miniWeather.
  - Minisweep's four kernels loop over 200–240 iterations, so at AMD's
    default of 256 threads they launch one team. The tuner's kernel
    discovery (`dropSingleTeams`) drops single-team kernels, so they were
    never registered. Keeping them needs an AMD exception and a re-recording.
- **Landscapes** of the 22 kernels (`~/rr-amd-work/hecbench/calib`) serve
  as an out-of-sample check of the MI50 thresholds.
  - The committed rules match 15 of 22 labels; the T4 rules also match 15.
    15 of the 22 kernels are register-type, which the T4 rules'
    everything-is-register happens to fit.
  - In-sample (24 LULESH/miniFE kernels) the rules match 18 against 11.
  - The spill test is the part that generalizes.
  - The occupancy < 20% rule also flags S3D kernels that gain from runtime
    parameters only.
- **Gains:** sw4ck 2.2–3.9×, RSBench 3.06×, miniWeather l760 2.96×, S3D
  1.29–2.23×, SU3 1.90×, SimpleMOC 1.29×, XSBench 1.11×. HACCmk has nothing
  to gain.
- Two S3D kernels (l189, l205) were redone because their final baseline
  replay failed once. The calibration driver does not retry baselines; the
  tuner's `Search` does.

### HeCBench tuned as new applications, and input sensitivity (2026-10-22)

**Minisweep.**
- AMD kernel discovery now keeps single-team kernels that have a loop trip
  count (`Device.dropSingleTeams`).
- That registers minisweep's 4 kernels (1.55–2.23×), giving 26 HeCBench
  kernels.

**Online, as new applications** (`phase5/online-hec`, `truth-hec`).
- The final tuner is trained on the other applications (pool of 50 kernels),
  and batch TPE runs alongside, 3 seeds each.

| Strategy | True speedup | Share of attainable gain | Time | vs tpe-batch |
|---|---|---|---|---|
| rank-confirm | 1.79× | 0.95 | 157 s | 1.001 [0.989, 1.014] |
| tpe-batch | 1.79× | 0.96 | 380 s | — |

- On the T4 the paper had 121 s vs 332 s, also with the same speedups.

**Input sensitivity** (`~/rr-amd-work/inputs`).
- Six second inputs. miniWeather 1600×800 is a separate build with
  SIM_TIME=10.
- 26 kernels with headroom; the T4 had 15.

| At the second input | Transfer | Scaled | Retuned |
|---|---|---|---|
| Speedup | 1.37× | 1.13× | 1.54× |
| Ratio to retuned | 0.89 [0.82, 0.95] | 0.73 [0.65, 0.83] | — |
| Kernels slower than default | 4 | 11 | 0 |

- On the T4 the speedups were 1.19× / 1.35× / 1.50×.
- On MI50, rescaling the teams hurts: on S3D's smaller input and
  miniWeather's larger grid.

**Pool fix.**
- `build_pool.py` mapped kernel names to applications over all recordings.
  The second inputs reuse the names, so miniFE and SimpleMOC rows were
  labelled with second-input applications. Leave-application-out then kept
  them in training.
- Affected runs were rerun and the old ones archived in
  `~/rr-amd-work/leaky-pool`:
  - SimpleMOC l379 rank-confirm (item 2); the result is unchanged at
    1.269×.
  - The MINIFE_NX150 and SIMPLEMOC_S1500K retunes.

**Training-set size** (2026-10-23, `~/rr-amd-work/bulk`).

Data:
- 139 bulk HeCBench applications, recorded with a 4 GB region; large
  recording files are deleted after their landscapes.
- 235 kernels of 95 applications in the pool, 203 of them (76 applications)
  with headroom; the T4 had 162 kernels, 101 with headroom.

Offline harness (`training_size.py`, `summarize_ts.py`):

| Share after 1/5/10 evaluations | All 203 kernels | 156 new kernels |
|---|---|---|
| rank LOAO (~230 training kernels) | .63/.80/.89 | .70/.85/.92 |
| rank LOAO50 (original 50) | .35/.57/.80 | .33/.55/.80 |
| TPE | .08/.45/.58 | .06/.45/.57 |

- On new kernels after 5 evaluations: LOAO − LOAO50 = +0.30
  [+0.24, +0.36], and LOAO − TPE = +0.40.
- Unlike the T4, LOAO50 also beats TPE (+0.10).

Pipeline fixes found on the way:
- AMD parser: one row per kernel.
- Landscape driver: a plain line number selects every kernel on that line.
- Pipeline: no landscapes when the recording fails; stale checkpoints moved
  aside before re-recording.

**Earlier blocker (resolved by the per-application cleanup).**
- The bulk app ace recorded 41 GB; `execute/H_ACE` must be removed by the
  user.
- The bulk pipeline (`~/rr-amd-work/bulk/pipeline.sh`) now:
  - records with a 4 GB region (`RR_RECORD_MEM_GB`);
  - removes the files of kernels over 1 GB.
- The offline harness is `bulk/training_size.py`.

### Final tuner with 1, 2, 4, 8 GPUs (2026-10-24, `phase5/mgpu`)

Setup: 5 kernels (l649, l844, l1201, miniFE l290, l521), 3 seeds, one run at
a time, validated in 5 rounds.

| GPUs | End to end | Before search | Screening | Confirm | True speedup |
|---|---|---|---|---|---|
| 1 | 227 s | 97 s | 114 s | 16 s | 1.858 |
| 2 | 280 s | 169 s | 98 s | 13 s | 1.851 |
| 4 | 202 s | 89 s | 102 s | 12 s | 1.849 |
| 8 | 188 s | 89 s | 87 s | 12 s | 1.845 |

- 8 GPUs save only 17% end to end, because per-process fixed costs dominate.
- 2 GPUs are slower: with 2 GPUs the multi-GPU path takes its third
  classification profile alone. The single-GPU path stops after one clear
  profile.
- Quality drops < 1% with 8 GPUs (cross-GPU timing calibration noise).
- AMD replays need exactly one visible GPU.

### Search strategies and energy (paper §5) on MI50 (2026-10-25, `phase5/strategies`)

Setup: 10 LULESH kernels, 3 seeds, 60 proposals (15 initial), shared cache
per kernel, two timing processes per configuration, no metrics replay
(`RR_SKIP_METRICS=1`), validated in 5 rounds.

| Strategy | True speedup | vs bo-speedup |
|---|---|---|
| tpe-speedup | 2.352 | 1.034 [0.998, 1.078] |
| motpe | 2.308 | 1.015 |
| bo-speedup | 2.275 | 1 |
| random | 2.235 | 0.982 [0.950, 1.023] |
| bo-energy | 2.195 | 1.002 without l1201 |

- l1201's energy is not physical (about 2100 W implied): the reduction's
  back-to-back energy launches run on modified inputs. It is excluded from
  the energy analysis.
- Over 9 kernels: shipping the fastest configuration leaves 3.6% of the
  energy gain; shipping the most efficient costs 4.7% in speed.
- Hypervolume: bo-energy 0.795, motpe 0.788, tpe 0.749, bo-speedup 0.720,
  random 0.691.
- One worker deadlocked inside Python (idle, no children) for about 10 h and
  was restarted. The cause is unknown.

**Training-set extension:** regression model, RGPE, GP-EI
(`bulk/training_size2.py`). After 5 evaluations on new kernels, the ranking
model beats regression by +0.10, and RGPE is about equal to the ranking model.

### Remaining

1. **Phase 5 measurement validation** (paper §4, Table `tab:measurement`):
   cross-process CV, Kendall τ, and energy vs duration on six LULESH kernels.
   This needs a driver like the paper's (not in the repo). Use the tuner's
   own LULESH/miniFE recordings (`optimize.py --record`).
2. **Recalibrate the classification thresholds** on MI50 warm baselines.
   MI50 has 1/2-rate FP64 and HBM2, so the T4's compute-saturated kernels
   may change class.
3. **Open questions:**
   - GPU-to-GPU timing differences (the l2103 warm-vs-cold discrepancy
     between GPU 1 and GPU 3).
   - Whether the work-group cap of 16 per CU in `_amd_blocks_per_cu` holds on
     gfx906.
4. **Pre-existing tuner issue, not AMD-specific:** proposals that repeat
   measured configurations cost no budget, so a search that only repeats
   runs until the wall-clock cap.

## Phase 2: AMDGPU plugin measurement (`plugins-nextgen/amdgpu`)

All changes are opt-in through the same environment variables as CUDA.

1. **Timing.** In `AMDGPUKernelTy::launchImpl`:
   - Enable `hsa_amd_profiling_set_profiler_enabled` on the queue.
   - For each timed launch, wait on the completion signal and read
     `hsa_amd_profiling_get_dispatch_time`. Convert ticks with
     `HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY`.
   - Port warm-up, `REWARM_MS` (static `Warmed`), `REPEATS` and the median from
     the CUDA `launchImpl`.
   - Kernel arguments are released by a stream post-action, so the repeated
     launches must either reuse one kernarg buffer or allocate one per launch.
   - Also handle `LIBOMPTARGET_RR_PROFILE_TIMING`.
2. **Clock and energy.** Add a `rrsmi` namespace, mirroring `rrnvml`:
   - `dlopen("librocm_smi64.so")`.
   - Map the HSA agent to an SMI index via PCI BDF
     (`HSA_AMD_AGENT_INFO_BDFID` vs `rsmi_dev_pci_id_get`).
   - Read the clock with `rsmi_dev_gpu_clk_freq_get(SYS)` and energy with
     `rsmi_dev_energy_count_get`.

   Verify early that the energy counter works on Vega 20. If it does not, fall
   back to integrating `rsmi_dev_power_ave_get` over the `ENERGY_MIN_MS` window
   and report it as such.
3. **Metrics.** Use rocprofiler v1 (`rocprofiler_open` around one dispatch, with
   a warm-up if `METRICS_WARMUP_MS` is set). Define a fixed AMD set equivalent
   to the eight CUDA metrics, plus `LIBOMPTARGET_RR_EXTRA_METRICS`. Validate
   every name against `rocprof --list-basic/--list-derived` on `gfx906`.
   Candidates:
   - DRAM traffic: `FETCH_SIZE`, `WRITE_SIZE`
   - L2 hit rate: `TCC_HIT_sum`, `TCC_MISS_sum` (or `L2CacheHit`)
   - Utilization: `VALUBusy`, `VALUUtilization`, `MemUnitBusy`
   - Instruction counts: `SQ_INSTS_VALU`, `SQ_INSTS_VMEM_RD`, `SQ_INSTS_VMEM_WR`,
     `SQ_INSTS_VALU_F64` (if present)
   - Occupancy: `SQ_WAVES`, `SQ_BUSY_CYCLES`, wave occupancy
   - Scratch (spill) traffic
   - `GRBM_GUI_ACTIVE`
4. **`kernel_activities.csv`.** Write it directly from `launchImpl` with the
   same 24-column header. Take times from the dispatch timestamps, grid and
   block from the launch, and VGPR/SGPR/LDS/scratch from code-object metadata.
   Flush the file in `AMDGPUPluginTy::deinitImpl`.
5. **`DEVICE_RESTORE`.** Implement same-device `dataExchangeImpl` with
   `hsa_amd_memory_async_copy` plus a signal wait. Make `isDataExchangable`
   return true when `Src == Dst`.
6. **(Only if needed) fixed-VA mapping.** Implement `implementsMemoryMap`,
   `mMap` and `munMap` with `hsa_amd_vmem_*` (ROCm 6.0+).
7. **CMake.** Link or `dlopen` `rocm_smi64` and `rocprofiler64` through
   `find_package` / `ROCM_PATH`, not hardcoded paths. While there, also replace
   the hardcoded `/data2/nmn/nvmetrics` paths in the CUDA CMake.

Exit criterion: on six LULESH kernels, reproduce the Section 4 validation
(Table `tab:measurement`): cross-process CV, Kendall τ between rounds, and the
energy–duration correlation.

## Phase 3: Python tuner (`record-replay/RR`)

1. **Vendor-neutral GPU monitor.** Add a small module with an NVML backend and a
   rocm-smi backend (ctypes on `librocm_smi64` or `amdsmi`). It provides energy,
   clock, running PIDs for the exclusivity guard, and ordinal-to-physical
   mapping.
   - Import `pynvml` lazily (`optimize.py:13-17`, `Benchmark.py:19-20`).
   - Use `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` in `_run_on_devices`
     and `executeOnGPU`.
2. **`Device.AMDGPU`.**
   - Query properties from `rocminfo` or HSA instead of hardcoding `gfx90a`:
     CUs (60 on MI50), wave64, max workgroup 1024, waves per SIMD/CU, VGPRs,
     LDS.
   - Fix the profile-command `KeyError` (`Benchmark.py:1681`).
   - Make `parse()` read the plugin's `kernel_activities.csv`, not rocprof CSV.
   - Implement `isValidConfig` and `getDefaultBounds` for wave64.
   - Allow `setOrigDescr` for AMD (`Benchmark.py:1702,1739`).
3. **Launch-bound mapping.**
   - `MaxThreads` becomes `amdgpu-flat-work-group-size="1,N"`, matching
     `OpenMPOpt`, not `{WarpSize},{N}`.
   - `MinTeams` becomes `amdgpu-waves-per-eu`, clamped to at least 1. It
     currently computes 0 for small configurations, which is an invalid
     attribute.
   - Revisit the normalized space: MaxThreads in multiples of 64, and the
     MinTeams range from waves per EU (1–10 on gfx906).
4. **Compile-time features.**
   - Implement `AMDGPU.deviceCompile` with `llc -mcpu=gfx906`. Take VGPR, SGPR,
     scratch (spill) and LDS from the code-object metadata, or from
     `-pass-remarks-analysis=kernel-resource-usage`.
   - Replace the T4 constants in `Search.py:95-97` and the hardcoded 32s
     (`:147,302-311,355-360,424,753`) with values from the device.
   - Write an AMD occupancy formula (VGPR/SGPR/LDS-limited waves per SIMD).
5. **Metrics and classification.**
   - Make `ExtraMetrics`, the spill/occupancy keys (`Benchmark.py:995-997`)
     and the saturation metric (`:775`) device attributes.
   - Map the classification rules (SM throughput, DRAM throughput, achieved
     occupancy, local loads) to AMD counters.
   - Recalibrate the thresholds from warm baselines on MI50. The paper derived
     them from 10 LULESH kernels on a T4.
6. **Cleanup.** Make `requirements.txt` vendor-conditional and update
   `BUILD_AND_RUN.md` / `AGENTS.md` with an AMD section and a
   `run-*-optimizer.sh` for AMD.

## Phase 4: benchmarks

- **miniFE, LULESH, vAdd.** Add `gfx906` Makefile variants, or switch the
  wrappers to `Device.GetCompileOptions` throughout.
- **HeCBench.** Its OpenMP makefiles natively target AMD, so the retargeting is
  simpler than on NVIDIA. However, the `Makefile.rr` files and sources live in
  `/data2/nmn/rr-metrics/hecbench`, which is not mounted on this host. They need
  to be copied over, ideally with the `Makefile.rr` files committed. The same
  applies to the testSNAP/miniBUDE data and the miniWeather MPI stub.

## Phase 5: experiments (replication on AMD)

In order, each gated on the previous:

1. Measurement validation (paper §4) and clock-ramp figure.
2. Landscapes for LULESH/miniFE kernels (batch replay), plus warm baseline
   classification. Recalibrate class thresholds.
3. Online tuner comparison: TPE, `tpe-batch`, `rank-confirm`, with refinement,
   validated in randomized rounds as in §12.
4. HeCBench transfer, input sensitivity, training-set size (§13–15).

Note for the write-up: MI50 differs from the T4 in ways that matter for the
conclusions:
- wave64
- different register and occupancy tiers
- strong FP64 (1/2 rate), which will move kernels out of the FP64-saturated
  class
- 300 W power limit
- HBM2 at about 1 TB/s

This is the "different architecture" replication that the Limitations section
says is missing.

## Decisions

- ROCm 5.7.1 with rocprofiler v1 (decided 2026-10-20).
- GCC 12 (12.5.0) is installed system-wide as `gcc-12` / `g++-12`.
- HeCBench is cloned from GitHub to `~/HeCBench` (commit `f0c817aa3`, the
  same commit as the RTX 6000 replication).
- Still open: keep the `||NVMetric||` tag for AMD, or generalize it to
  `||Metric||`.
