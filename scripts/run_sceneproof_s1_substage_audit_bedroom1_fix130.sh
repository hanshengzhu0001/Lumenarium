#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"
PY="${HOME}/.venvs/lumenarium-py311/bin/python"
ROOT="${SCENEPROOF_FIX130_RESULTS_ROOT:-${HOME}/Lumenarium/a10_reusable_results/fix124_v5_fast_cold_paper30}"
OUT="${SCENEPROOF_FIX130_OUT:-${ROOT}/sceneba_audit/s1_substage_timing_bedroom1_fix131.json}"

"${PY}" audit_sceneproof_s1_substages_fix130.py \
  --results-root "${ROOT}" \
  --scene bedroom_01 \
  --source-version v4_deepsearch \
  --out "${OUT}"
