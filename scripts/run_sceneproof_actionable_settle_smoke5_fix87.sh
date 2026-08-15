#!/usr/bin/env bash
# SceneProof Fix87 — settle only what gravity can actually fix, then attribute.
#
# Fix86 screened out the objects gravity must not touch (wall and ceiling
# attachments, contained objects, support parents).  Its last rule was still
# wrong: it kept any object whose *total* support term was below one, which on
# Smoke5 admitted 286 targets, including objects already in nanometre-scale
# contact with their parent.
#
# The evaluator's resting-support term is an explicit closed form,
#   s = ( L(gap, 0.05) + L(containment, 0.05) + min(1, overlap/0.9) ) / 3,
# and a vertical drop can only move the first summand.  Fix87 therefore gates on
# the closed-form attainable gain
#   delta_max = min(1, gap / 0.05) / 3,
# so an object is simulated only when a settle can actually pay for itself.
#
# The screen also reports two things worth more than the settle itself:
#   * objects whose declared support parent has a degenerate footprint (fewer
#     than three hull vertices, the only condition that makes the evaluator's
#     containment error infinite).  Two of the three summands are unreachable for
#     them, so their support term is capped at 1/3 no matter what any optimizer
#     or simulator does.  That is a support-graph or asset-footprint defect, not
#     a physics defect.
#   * the split of the scene's residual support deficit into the contact-gap part
#     that gravity can recover and the containment and overlap parts it cannot.
#
# Set SCENEPROOF_SCREEN_ONLY=1 to run the screen alone: no Blender, no GPU,
# seconds per scene.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
BATCH="v5_sceneproof_actionable_settle_batch_fix87"
SELECT="v5_sceneproof_actionable_settle_select_fix87"

SCREEN_ONLY="${SCENEPROOF_SCREEN_ONLY:-0}"
MIN_GAP="${SCENEPROOF_MIN_SETTLE_GAP_M:-0.002}"
MIN_GAIN="${SCENEPROOF_MIN_ATTAINABLE_GAIN:-0.01}"
MAX_TARGETS="${SCENEPROOF_MAX_SETTLE_TARGETS:-0}"
MIN_TRANS="${SCENEPROOF_SETTLE_MIN_TRANSLATION_M:-0.002}"
GATED="${SCENEPROOF_SETTLE_GATED_FAMILIES:-collision,support,plane}"
DURATION="${SCENEPROOF_LOCAL_SETTLE_DURATION:-1.0}"
SETTLE_TIMEOUT="${SCENEPROOF_SETTLE_TIMEOUT:-3600}"
EVAL_TIMEOUT="${SCENEPROOF_EVAL_TIMEOUT:-1800}"

AUDIT_ROOT="$root/sceneba_audit/$SELECT"
mkdir -p "$AUDIT_ROOT"
summary="$AUDIT_ROOT/_summary.txt"
: > "$summary"

echo "=== SCENEPROOF FIX87: ACTIONABLE-GAIN SETTLE + COMPONENT ATTRIBUTION ==="
echo "Scenes:       $SCENES"
echo "Baseline:     $BASELINE"
echo "Geometry:     $SOURCE"
echo "Screen only:  $SCREEN_ONLY"
echo "Min gap:      ${MIN_GAP} m"
echo "Min gain:     $MIN_GAIN (support-term units)"
echo "Gated:        $GATED"
echo "Start:        $(date)"

for scene in $SCENES; do
    echo ""
    echo "--- $scene ($(date +%H:%M:%S)) ---"
    audit="$AUDIT_ROOT/$scene"
    mkdir -p "$audit"

    placement="$root/${scene}_${BASELINE}_result/S4_layout_refinement/${scene}_${BASELINE}_placement_info_s4.json"
    if ! test -s "$placement"; then
        echo "SKIP $scene: missing baseline placement"
        echo "$scene SKIP missing_baseline_placement" >> "$summary"
        continue
    fi
    manifest="$audit/scene_manifest.txt"
    printf '%s\n' "$scene" > "$manifest"

    # ---- stage A: actionable-gain screen -----------------------------------
    baseline_csv="$root/sceneba_audit/v5_sceneproof_settle_component_select_fix85/$scene/batch_physical_objects.csv"
    if ! test -s "$baseline_csv"; then
        baseline_csv="$audit/baseline_physical_objects.csv"
        if ! timeout "$EVAL_TIMEOUT" "$python" eval_physical_realizability.py \
            --saved-results "$root" --scenes "$manifest" \
            --versions "$BASELINE" --geometry-version "$SOURCE" \
            --baseline-version "$BASELINE" \
            --metrics-out "$audit/baseline_physical.json" \
            --scene-csv "$audit/baseline_physical_scenes.csv" \
            --object-csv "$baseline_csv" \
            --report-out "$audit/baseline_physical.txt" \
            > "$audit/baseline_eval.log" 2>&1; then
            echo "  FAIL baseline evaluation, see $audit/baseline_eval.log"
            echo "$scene FAIL baseline_eval" >> "$summary"
            continue
        fi
    fi

    echo "  [A] actionable-gain screen ..."
    screen_rc=0
    "$python" sceneproof_settle_eligibility_screen_fix86.py \
        --scene "$scene" \
        --incumbent "$placement" \
        --baseline-objects-csv "$baseline_csv" \
        --baseline-version "$BASELINE" \
        --saved-results "$root" \
        --geometry-version "$SOURCE" \
        --minimum-gap-m "$MIN_GAP" \
        --minimum-attainable-gain "$MIN_GAIN" \
        --max-targets "$MAX_TARGETS" \
        --out-report "$audit/eligibility.json" \
        --out-ids "$audit/target_ids.txt" > "$audit/screen.log" 2>&1 || screen_rc=$?
    sed -n '/^SCORED=/,$p' "$audit/screen.log" | sed 's/^/       /'
    if test "$screen_rc" -eq 4; then
        echo "  NO-OP $scene: no object has an actionable contact gap"
        echo "$scene NOOP no_actionable_target" >> "$summary"
        continue
    fi
    if test "$screen_rc" -ne 0; then
        echo "  FAIL screen, see $audit/screen.log"
        echo "$scene FAIL screen" >> "$summary"
        continue
    fi
    target_ids=$(cat "$audit/target_ids.txt")
    n_targets=$("$python" -c "import sys; print(len([t for t in sys.argv[1].split(',') if t]))" "$target_ids")
    if test "$SCREEN_ONLY" = "1"; then
        echo "$scene SCREENED $n_targets" >> "$summary"
        rm -f "$manifest"
        continue
    fi

    # ---- stage B: well-posed settle of the screened targets ----------------
    echo "  [B] well-posed settle of $n_targets targets ..."
    probe_dir="$audit/probes"
    mkdir -p "$probe_dir"
    n_probes=$(find "$probe_dir" -maxdepth 1 -name '*.json' ! -name '_*' 2>/dev/null | wc -l)
    if test "$n_probes" -eq 0; then
        source_json=$(find "$root/${scene}_${SOURCE}_result/S3_pose_inference" -maxdepth 1 -name '*_placement_info.json' -print -quit)
        if test -z "$source_json"; then
            echo "  SKIP $scene: no S3 placement for $SOURCE"
            echo "$scene SKIP missing_s3_placement" >> "$summary"
            continue
        fi
        timeout "$SETTLE_TIMEOUT" env \
            CUDA_VISIBLE_DEVICES=0 \
            IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
            IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
            IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_AUDIT_OUTPUT="$probe_dir" \
            IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_OBJECT_IDS="$target_ids" \
            IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS="$DURATION" \
            LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
            "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
              --obj_placement_info_json_path "$source_json" \
              --output_folder "/tmp/fix87_settle_${scene}_$$" \
              > "$audit/settle_blender.log" 2>&1 < /dev/null \
            || echo "  WARN: settle returned non-zero, see $audit/settle_blender.log"
        n_probes=$(find "$probe_dir" -maxdepth 1 -name '*.json' ! -name '_*' 2>/dev/null | wc -l)
    else
        echo "       $n_probes cached probes, reusing"
    fi
    if test "$n_probes" -eq 0; then
        echo "  FAIL $scene: settle produced no probe"
        echo "$scene FAIL no_probe" >> "$summary"
        continue
    fi
    echo "       probes: $n_probes"

    # ---- stage C: attribution, selection, confirmation ---------------------
    echo "  [C1] materialize batch candidate ..."
    batch_placement="$root/${scene}_${BATCH}_result/S4_layout_refinement/${scene}_${BATCH}_placement_info_s4.json"
    if ! "$python" sceneproof_settle_component_select_fix85.py materialize-batch \
        --incumbent "$placement" \
        --probe-dir "$probe_dir" \
        --min-translation-m "$MIN_TRANS" \
        --out "$batch_placement" > "$audit/materialize_batch.log" 2>&1; then
        echo "  NO-OP $scene: no probe moved beyond ${MIN_TRANS} m"
        echo "$scene NOOP no_moved_probe" >> "$summary"
        continue
    fi
    grep -E '^MOVED=' "$audit/materialize_batch.log" | sed 's/^/       /'

    echo "  [C2] batch physical evaluation ..."
    if ! timeout "$EVAL_TIMEOUT" "$python" eval_physical_realizability.py \
        --saved-results "$root" --scenes "$manifest" \
        --versions "$BASELINE,$BATCH" --geometry-version "$SOURCE" \
        --baseline-version "$BASELINE" \
        --metrics-out "$audit/batch_physical.json" \
        --scene-csv "$audit/batch_physical_scenes.csv" \
        --object-csv "$audit/batch_physical_objects.csv" \
        --report-out "$audit/batch_physical.txt" > "$audit/batch_eval.log" 2>&1; then
        echo "  FAIL batch physical evaluation, see $audit/batch_eval.log"
        echo "$scene FAIL batch_physical" >> "$summary"
        continue
    fi

    echo "  [C3] component attribution and selection ..."
    select_placement="$root/${scene}_${SELECT}_result/S4_layout_refinement/${scene}_${SELECT}_placement_info_s4.json"
    select_rc=0
    "$python" sceneproof_settle_component_select_fix85.py select \
        --scene "$scene" \
        --saved-results "$root" \
        --geometry-version "$SOURCE" \
        --incumbent "$placement" \
        --probe-dir "$probe_dir" \
        --min-translation-m "$MIN_TRANS" \
        --batch-candidate "$batch_placement" \
        --batch-physical "$audit/batch_physical.json" \
        --batch-objects-csv "$audit/batch_physical_objects.csv" \
        --baseline-version "$BASELINE" \
        --candidate-version "$BATCH" \
        --gated-families "$GATED" \
        --out-selection "$audit/selection.json" \
        --out-candidate "$select_placement" > "$audit/select.log" 2>&1 || select_rc=$?
    sed -n '/^MOVED=/,$p' "$audit/select.log" | sed 's/^/       /'
    if test "$select_rc" -ne 0; then
        echo "  ABSTAIN $scene: attribution verification failed (rc=$select_rc)"
        echo "$scene ABSTAIN attribution_verification_failed" >> "$summary"
        continue
    fi
    n_accepted=$("$python" -c "import json,sys; print(json.load(open(sys.argv[1]))['selection']['accepted_object_count'])" "$audit/selection.json")
    if test "$n_accepted" -eq 0; then
        echo "  NO-OP $scene: no component is non-inferior in every gated family"
        echo "$scene NOOP no_safe_component" >> "$summary"
        continue
    fi

    echo "  [C4] confirmation evaluation on $n_accepted objects ..."
    if ! timeout "$EVAL_TIMEOUT" "$python" eval_physical_realizability.py \
        --saved-results "$root" --scenes "$manifest" \
        --versions "$BASELINE,$SELECT" --geometry-version "$SOURCE" \
        --baseline-version "$BASELINE" \
        --metrics-out "$audit/confirm_physical.json" \
        --scene-csv "$audit/confirm_physical_scenes.csv" \
        --object-csv "$audit/confirm_physical_objects.csv" \
        --report-out "$audit/confirm_physical.txt" > "$audit/confirm_eval.log" 2>&1; then
        echo "  FAIL confirmation evaluation, see $audit/confirm_eval.log"
        echo "$scene FAIL confirm_physical" >> "$summary"
        continue
    fi
    gt_flag=""
    if timeout "$EVAL_TIMEOUT" "$python" eval_gt_metrics.py \
        --saved-results "$root" --scenes "$manifest" \
        --versions "$BASELINE,$SELECT" \
        --min-visible-mask-area 8000 --min-visible-bbox-size 0 \
        --batch-logs logs \
        --metrics-out "$audit/confirm_gt_8000.json" \
        --manifest-out "$audit/confirm_gt_manifest_8000.json" \
        > "$audit/confirm_gt.log" 2>&1; then
        gt_flag="--confirm-gt $audit/confirm_gt_8000.json"
    else
        echo "       WARN: GT evaluation failed, gating on physical families only"
    fi

    echo "  [C5] gate on measured confirmation ..."
    confirm_rc=0
    "$python" sceneproof_settle_component_select_fix85.py confirm \
        --scene "$scene" \
        --selection "$audit/selection.json" \
        --confirm-physical "$audit/confirm_physical.json" \
        $gt_flag \
        --baseline-version "$BASELINE" \
        --candidate-version "$SELECT" \
        --out "$audit/promotion.json" > "$audit/confirm.log" 2>&1 || confirm_rc=$?
    sed -n '/^ /,$p' "$audit/confirm.log" | sed 's/^/     /'
    if test "$confirm_rc" -eq 0; then
        echo "  PROMOTED $scene: $n_accepted objects"
        echo "$scene PROMOTED $n_accepted" >> "$summary"
    else
        echo "  REJECTED $scene at confirmation (rc=$confirm_rc)"
        echo "$scene REJECTED confirmation" >> "$summary"
    fi
    rm -f "$manifest"
done

echo ""
echo "========================================"
echo "FIX87 COMPLETE  $(date)"
echo "----------------------------------------"
cat "$summary"
echo "----------------------------------------"
if test "$SCREEN_ONLY" = "1"; then
    screened=$(awk '$2=="SCREENED" {sum += $3} END {print sum+0}' "$summary")
    echo "Total actionable targets across scenes: $screened"
    echo "Support deficit attribution per scene:"
    for scene in $SCENES; do
        report="$AUDIT_ROOT/$scene/eligibility.json"
        test -s "$report" || continue
        "$python" - "$report" "$scene" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
d = data["support_deficit"]
print(
    "  {:<16} targets={:<4} deficit={:.4f}  gap={:.4f}  containment={:.4f}  "
    "overlap={:.4f}  recoverable={:.4f}  degenerate_parent={}".format(
        sys.argv[2], data["target_count"], d["total"],
        d["contact_gap_component"], d["containment_component"],
        d["footprint_overlap_component"], d["recoverable_by_settling"],
        data["parent_footprint_degenerate_count"],
    )
)
PY
    done
else
    promoted=$(awk '$2=="PROMOTED" {sum += $3} END {print sum+0}' "$summary")
    echo "Total promoted objects: $promoted"
    echo "Per-scene evidence: \$AUDIT_ROOT/<scene>/{eligibility.json,selection.json,promotion.json}"
fi
echo "AUDIT_ROOT=$AUDIT_ROOT"
