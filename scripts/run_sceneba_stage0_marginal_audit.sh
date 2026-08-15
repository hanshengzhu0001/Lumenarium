#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENEBA_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
ROOT="${SCENEBA_RESULTS_ROOT:-a10_reusable_results/paper30}"
AUDIT_ROOT="${SCENEBA_STAGE0_AUDIT_ROOT:-$ROOT/sceneba_audit/asset_vlm_recoverable_v1}"
BLIND="${SCENEBA_STAGE0_BLIND_MANIFEST:-$AUDIT_ROOT/blind_manifest.json}"
ANSWER="${SCENEBA_STAGE0_ANSWER_KEY:-$AUDIT_ROOT/answer_key.json}"
RULES="${SCENEBA_STAGE0_RULES:-config/sceneba_stage0_fixed_rules.json}"
TABLE="${SCENEBA_STAGE0_FACTOR_TABLE:-$AUDIT_ROOT/stage0_factor_table_blind.json}"
REPORT="${SCENEBA_STAGE0_REPORT:-$AUDIT_ROOT/stage0_marginal_audit.json}"

echo "FIXED_RULES_SHA256=$(sha256sum "$RULES" | awk '{print $1}')"
echo "BUILD_BLIND_FACTOR_TABLE $(date)"

env LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$PY" sceneba_marginal_factor_table.py \
    --saved-results "$ROOT" \
    --blind-manifest "$BLIND" \
    --source-version v4_deepsearch \
    --reference-version v4 \
    --maximum-views 2 \
    --yaw-modes 0,180 \
    --raster-size 128 \
    --out "$TABLE"

echo "OPEN_ANSWER_KEY_ONLY_AFTER_FACTOR_TABLE_IS_FROZEN $(date)"
"$PY" sceneba_marginal_factor_audit.py \
  --factor-table "$TABLE" \
  --fixed-rules "$RULES" \
  --answer-key "$ANSWER" \
  --out "$REPORT"

echo "STAGE0_COMPLETE report=$REPORT $(date)"
