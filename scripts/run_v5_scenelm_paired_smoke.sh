#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

export SCENELM_SOLVER=v5_scenelm
export SCENELM_REFERENCE_VERSION="${SCENELM_REFERENCE_VERSION:-v4_fast_adam400_control}"
export SCENELM_VERSION="${SCENELM_VERSION:-v5_scenelm}"
export SCENELM_MAX_ITERATIONS="${SCENELM_MAX_ITERATIONS:-30}"

exec bash scripts/run_scenelm_paired_smoke.sh
