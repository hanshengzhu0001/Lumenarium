#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
python="$HOME/.venvs/lumenarium-py311/bin/python"
root="$HOME/Lumenarium/a10_reusable_results/paper30"
baseline="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
source="v4_deepsearch"
prefix="v5_sceneproof_generalized_settle_fix110"
previous="$root/sceneba_audit/v5_sceneproof_generalized_settle_nearpass_fix109"
audit="$root/sceneba_audit/$prefix"
subset_dir="$audit/subset_search"
paper_manifest="$root/manifest.txt"
selected="$previous/per_object_trials.tsv"

test -s "$selected" || { echo "Missing Fix109 trials: $selected" >&2; exit 2; }
mkdir -p "$subset_dir"

"$python" sceneproof_generalized_settle_subset_selector_fix110.py prepare \
  --selected "$selected" --out-dir "$subset_dir" --target-prefix "$prefix"

for mask in 00 01 02 03 04 05 06 07; do
  target="${prefix}_subset_${mask}"
  subset="$subset_dir/subset_${mask}.tsv"
  "$python" sceneproof_rigid_settle_adaptive_eval_fix84.py compose \
    --saved-results "$root" --manifest "$paper_manifest" --selected "$subset" \
    --baseline-version "$baseline" --target-version "$target" \
    --target-manifest "$subset_dir/manifest_${mask}.txt" \
    --certificate "$subset_dir/certificate_${mask}.json"
  "$python" eval_physical_realizability.py \
    --saved-results "$root" --scenes "$paper_manifest" \
    --versions "$baseline,$target" --geometry-version "$source" \
    --baseline-version "$baseline" \
    --metrics-out "$subset_dir/physical_${mask}.json" \
    --scene-csv "$subset_dir/physical_scenes_${mask}.csv" \
    --object-csv "$subset_dir/physical_objects_${mask}.csv" \
    --report-out "$subset_dir/physical_${mask}.txt" \
    > "$subset_dir/physical_${mask}.log" 2>&1
done

"$python" sceneproof_generalized_settle_subset_selector_fix110.py select \
  --out-dir "$subset_dir" --baseline-version "$baseline" \
  --final-selected "$audit/winning_trials.tsv" \
  --audit "$audit/subset_selection.json"

awk -F '\t' 'NR > 1 && $6 == "true" { print $1 "\t" $2 }' \
  "$audit/winning_trials.tsv" > "$audit/winning_targets.tsv"

if ! test -s "$audit/winning_targets.tsv"; then
  echo "FIX110_WINNER retained=0 decision=keep_fix61_no_render"
  exit 0
fi

export SCENEPROOF_RIGID_TARGETS_FILE="$audit/winning_targets.tsv"
export SCENEPROOF_RIGID_TARGET_VERSION="$prefix"
export SCENEPROOF_ACCEPT_POLICY=relaxed
export SCENEPROOF_FORCE_MEASURED_CANDIDATES=1
export SCENEPROOF_LOCAL_ONLY=0
export IMAGINARIUM_SETTLE_XY_SLIP_TOLERANCE_M="${IMAGINARIUM_SETTLE_XY_SLIP_TOLERANCE_M:-0.005}"
export SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}"

bash scripts/run_sceneproof_rigid_only_adaptive_eval_fix84e.sh
