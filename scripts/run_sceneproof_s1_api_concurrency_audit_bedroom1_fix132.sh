#!/usr/bin/env bash
set -euo pipefail
cd "${HOME}/Lumenarium"
PY="${HOME}/.venvs/lumenarium-py311/bin/python"
ROOT="${SCENEPROOF_FIX132_RESULTS_ROOT:-${HOME}/Lumenarium/a10_reusable_results/fix124_v5_fast_cold_paper30}"
OUT="${ROOT}/sceneba_audit/s1_api_concurrency_bedroom1_fix132.json"
"${PY}" audit_sceneproof_s1_api_concurrency_fix132.py \
  --results-root "${ROOT}" --scene bedroom_01 --source-version v4_deepsearch --out "${OUT}"
