#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
saved="$HOME/Lumenarium/a10_reusable_results/paper30"
discovery="$saved/sceneba_audit/v5_sceneproof_generalized_settle_paper30_fix107/candidate_representatives.tsv"
targets="/tmp/sceneproof_generalized_settle_representatives_fix108.tsv"

test -s "$discovery" || {
  echo "Missing representative discovery: $discovery" >&2
  exit 2
}
tail -n +2 "$discovery" | cut -f1,2 > "$targets"
test -s "$targets" || {
  echo "Representative manifest is empty: $targets" >&2
  exit 2
}

export SCENEPROOF_RIGID_TARGETS_FILE="$targets"
export SCENEPROOF_RIGID_TARGET_VERSION="v5_sceneproof_generalized_settle_representatives_fix108_2"
export SCENEPROOF_ACCEPT_POLICY=relaxed
export SCENEPROOF_FORCE_MEASURED_CANDIDATES=1
export SCENEPROOF_LOCAL_ONLY=1
export IMAGINARIUM_SETTLE_XY_SLIP_TOLERANCE_M="${IMAGINARIUM_SETTLE_XY_SLIP_TOLERANCE_M:-0.005}"
export SCENEPROOF_RIGID_SETTLE_DURATION_SECONDS="${SCENEPROOF_RIGID_SETTLE_DURATION_SECONDS:-1.0}"

bash scripts/run_sceneproof_rigid_only_adaptive_eval_fix84e.sh
