#!/usr/bin/env bash
set -euo pipefail

cd "${SCENEBA_REPO_ROOT:-$HOME/Lumenarium}"

PY="${SCENEBA_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
RESULTS_ROOT="${SCENEBA_RESULTS_ROOT:-a10_reusable_results/paper30}"
AUDIT_ROOT="${SCENEBA_ASSET_VLM_AUDIT_ROOT:-${RESULTS_ROOT}/sceneba_audit/asset_vlm_recoverable_v1}"

"${PY}" sceneba_asset_vlm_eval_recoverable.py \
  --blind-results "${AUDIT_ROOT}/blind_results.json" \
  --answer-key "${AUDIT_ROOT}/answer_key.json" \
  --out "${AUDIT_ROOT}/evaluation.json" \
  2>&1 | tee logs/sceneba_asset_vlm_recoverable_v1_eval.log
