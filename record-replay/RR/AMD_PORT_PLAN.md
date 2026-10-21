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
- So the T4's 1000 ms warm-up is unnecessary on MI50; a few warm-up launches
  suffice. Confirm on a short kernel (l2103) before lowering the AMD defaults.

Remaining for phase 2:
- Hardware metrics through rocprofiler v1.
- `kernel_activities.csv`.

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
