#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENEBA_WITNESS_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENEBA_WITNESS_RESULTS_ROOT:-a10_reusable_results/paper30}"
audit="${SCENEBA_WITNESS_AUDIT_ROOT:-$root/sceneba_audit/moge_noc_witness_v1}"
answer="${SCENEBA_WITNESS_ANSWER_KEY:-$root/sceneba_audit/asset_vlm_recoverable_v1/answer_key.json}"

test -s "$audit/blind_witness_table.json" || {
  echo "Missing blinded witness table; do not open the answer key yet." >&2
  exit 2
}
test -s "$answer" || { echo "Missing answer key: $answer" >&2; exit 2; }

"$PY" sceneba_moge_noc_witness_eval.py \
  --table "$audit/blind_witness_table.json" \
  --answer-key "$answer" \
  --out "$audit/final_gates.json" \
  2>&1 | tee "$audit/final_gates.log"
