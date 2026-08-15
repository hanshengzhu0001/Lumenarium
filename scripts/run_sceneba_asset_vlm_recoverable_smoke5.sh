#!/usr/bin/env bash
set -euo pipefail

cd "${SCENEBA_REPO_ROOT:-$HOME/Lumenarium}"

PY="${SCENEBA_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
RESULTS_ROOT="${SCENEBA_RESULTS_ROOT:-a10_reusable_results/paper30}"
SCENES="${SCENEBA_ASSET_VLM_SCENES:-/tmp/sceneba_asset_vlm_smoke5.txt}"
TOPK_ORACLE="${SCENEBA_ASSET_VLM_TOPK_ORACLE:-${RESULTS_ROOT}/sceneba_audit/top3_pose_bank_smoke5.json}"
POSE_VERSION="${SCENEBA_ASSET_VLM_POSE_VERSION:-v4_deepsearch}"
CONFIG="${SCENEBA_ASSET_VLM_CONFIG:-config/config_a10_paper30.yaml}"
DOTENV="${SCENEBA_ASSET_VLM_DOTENV:-.env}"
MAXIMUM="${SCENEBA_ASSET_VLM_MAXIMUM_SAMPLES:-8}"
MINIMUM_CONFIDENCE="${SCENEBA_ASSET_VLM_MIN_CONFIDENCE:-0.80}"
AUDIT_ROOT="${SCENEBA_ASSET_VLM_AUDIT_ROOT:-${RESULTS_ROOT}/sceneba_audit/asset_vlm_recoverable_v1}"
BLIND="${AUDIT_ROOT}/blind_manifest.json"
ANSWER="${AUDIT_ROOT}/answer_key.json"
SHEETS="${AUDIT_ROOT}/contact_sheets"
RESULTS="${AUDIT_ROOT}/blind_results.json"

if [[ ! -s "${SCENES}" ]]; then
  printf '%s\n' \
    bedroom_01 \
    livingroom_10 \
    casino_01 \
    official_01 \
    streelitter_01 > "${SCENES}"
fi

mkdir -p "${AUDIT_ROOT}" logs

"${PY}" sceneba_asset_vlm_build_recoverable.py \
  --saved-results "${RESULTS_ROOT}" \
  --scenes "${SCENES}" \
  --topk-oracle "${TOPK_ORACLE}" \
  --pose-version "${POSE_VERSION}" \
  --top-k 3 \
  --maximum-samples "${MAXIMUM}" \
  --blind-out "${BLIND}" \
  --answer-key-out "${ANSWER}"

env \
  IMAGINARIUM_GPT_LOCK_FILE="${IMAGINARIUM_GPT_LOCK_FILE:-/tmp/lumenarium_gemini.lock}" \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "${PY}" -u sceneba_asset_vlm_recoverable.py \
    --saved-results "${RESULTS_ROOT}" \
    --blind-manifest "${BLIND}" \
    --source-version "${POSE_VERSION}" \
    --reference-version v4 \
    --config "${CONFIG}" \
    --dotenv "${DOTENV}" \
    --minimum-confidence "${MINIMUM_CONFIDENCE}" \
    --repeats 2 \
    --sheet-dir "${SHEETS}" \
    --out "${RESULTS}"

echo "BLIND_MANIFEST=${BLIND}"
echo "ANSWER_KEY=${ANSWER}"
echo "BLIND_RESULTS=${RESULTS}"
