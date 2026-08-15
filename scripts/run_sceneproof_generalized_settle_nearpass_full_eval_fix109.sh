#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
saved="$HOME/Lumenarium/a10_reusable_results/paper30"
previous="$saved/sceneba_audit/v5_sceneproof_generalized_settle_representatives_fix108_2"
shortlist="$previous/nearpass_fix109.tsv"

test -s "$shortlist" || {
  echo "Missing near-pass manifest: $shortlist" >&2
  exit 2
}

export SCENEPROOF_RIGID_TARGETS_FILE="$shortlist"
export SCENEPROOF_RIGID_TARGET_VERSION="v5_sceneproof_generalized_settle_nearpass_fix109"
export SCENEPROOF_ACCEPT_POLICY=relaxed
export SCENEPROOF_FORCE_MEASURED_CANDIDATES=1
export SCENEPROOF_LOCAL_ONLY=0
export IMAGINARIUM_SETTLE_XY_SLIP_TOLERANCE_M="${IMAGINARIUM_SETTLE_XY_SLIP_TOLERANCE_M:-0.005}"
export SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}"

bash scripts/run_sceneproof_rigid_only_adaptive_eval_fix84e.sh
