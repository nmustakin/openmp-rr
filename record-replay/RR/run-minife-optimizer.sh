#!/usr/bin/env bash
# Run after completing BUILD_AND_RUN.md's container build steps.
set -eo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir"
source ./update-gcc.sh
source ./update_path.sh
export RR_TUNING_PASS="$repo_dir/record-replay/Passes/build/turing.cs.ucr.edu/libTuneDevice.so"
export LD_LIBRARY_PATH="/data2/nmn/nvmetrics/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cd record-replay/RR
python3 -u optimize.py -b benchmarks/miniFE-omp/MINIFE.py -d nvidia \
  --kernel "${1:-all}" --scenario BOMN --init-samples "${2:-30}" --opt-samples "${3:-200}" \
  --objective speedup
