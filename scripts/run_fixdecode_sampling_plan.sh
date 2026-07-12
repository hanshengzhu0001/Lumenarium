#!/usr/bin/env bash
set -euo pipefail

cd /ssd/kevinzyz/imaginarium/Imaginarium-repo

PY=/ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python
SENTINEL_SESSION=fixdecode_sentinel4
STRATIFIED_LOG="batch_logs/batch_eval_fixdecode_stratified12_$(date +%Y%m%d_%H%M%S).log"

wait_for_tmux_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    echo "[$(date '+%F %T')] waiting for tmux session: $session"
    sleep 60
  done
}

echo "[$(date '+%F %T')] fixdecode sampling watcher started"
wait_for_tmux_session "$SENTINEL_SESSION"

echo "[$(date '+%F %T')] sentinel4 finished; computing metrics"
"$PY" eval_gt_metrics.py \
  --saved-results saved_results_fixdecode \
  --scenes eval_sample_fixdecode_sentinel4.txt \
  --metrics-out eval_gt_metrics_sentinel4_fixdecode.json \
  --manifest-out eval_freeze_manifest_sentinel4_fixdecode.json

"$PY" eval_matching_diagnostics.py \
  --saved-results saved_results_fixdecode \
  --metrics eval_gt_metrics_sentinel4_fixdecode.json \
  --scenes eval_sample_fixdecode_sentinel4.txt \
  --out eval_matching_diagnostics_sentinel4_fixdecode.json

echo "[$(date '+%F %T')] comparing sentinel4 against baseline"
decision=$("$PY" - <<'PY'
import json

base = json.load(open("eval_matching_diagnostics_sentinel4_baseline.json"))
cand = json.load(open("eval_matching_diagnostics_sentinel4_fixdecode.json"))

def totals(report):
    s1 = sum(report["overall"][v]["s1_bbox_count"] for v in ("v1", "v3"))
    s4 = sum(report["overall"][v]["s4_object_count"] for v in ("v1", "v3"))
    anon = sum(report["overall"][v]["anonymous_s1_count"] for v in ("v1", "v3"))
    matched = sum(report["overall"][v]["matched_object_count"] for v in ("v1", "v3"))
    gt = sum(report["overall"][v]["gt_object_count"] for v in ("v1", "v3"))
    return {
        "s1": s1,
        "s4": s4,
        "anon": anon,
        "matched": matched,
        "gt": gt,
        "s4_per_s1": s4 / s1 if s1 else 0.0,
        "anon_ratio": anon / s1 if s1 else 1.0,
        "matched_per_gt": matched / gt if gt else 0.0,
    }

b = totals(base)
c = totals(cand)
improved_retention = c["s4_per_s1"] >= max(b["s4_per_s1"] * 1.75, b["s4_per_s1"] + 0.15)
reduced_anonymous = c["anon_ratio"] <= b["anon_ratio"] * 0.6
print(json.dumps({"baseline": b, "fixdecode": c, "improved_retention": improved_retention, "reduced_anonymous": reduced_anonymous}, indent=2))
print("PASS" if improved_retention and reduced_anonymous else "STOP")
PY
)
echo "$decision"

if ! grep -q '^PASS$' <<<"$decision"; then
  echo "[$(date '+%F %T')] sentinel4 did not pass gate; not running stratified12"
  exit 0
fi

echo "[$(date '+%F %T')] sentinel4 passed; running stratified12"
env PYTHONUNBUFFERED=1 "$PY" batch_eval.py \
  --run-name fixdecode \
  --scenes eval_sample_fixdecode_stratified12.txt \
  --gpu-count 1 \
  --gpt-max-wait 240 \
  --gpt-max-retries 2 2>&1 | tee -a "$STRATIFIED_LOG"

echo "[$(date '+%F %T')] stratified12 finished; computing metrics"
"$PY" eval_gt_metrics.py \
  --saved-results saved_results_fixdecode \
  --scenes eval_sample_fixdecode_stratified12.txt \
  --metrics-out eval_gt_metrics_stratified12_fixdecode.json \
  --manifest-out eval_freeze_manifest_stratified12_fixdecode.json

"$PY" eval_matching_diagnostics.py \
  --saved-results saved_results_fixdecode \
  --metrics eval_gt_metrics_stratified12_fixdecode.json \
  --scenes eval_sample_fixdecode_stratified12.txt \
  --out eval_matching_diagnostics_stratified12_fixdecode.json

echo "[$(date '+%F %T')] fixdecode sampling plan complete"
