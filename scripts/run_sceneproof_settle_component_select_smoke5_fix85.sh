#!/usr/bin/env bash
# SceneProof Fix85— fast and stable promotion of a bulk gravity settle.
#
# Cost model.  Per scene this runs one bulk Bullet settle (cached across reruns),
# two physical evaluations, and one GT evaluation.  The count is independent of
# how many objects moved, so a scene with 109 moved objects costs the same as a
# scene with 21.
#
# Why that is enough.  Every family score of eval_physical_realizability.py is an
# unweighted mean of per-object terms with a pose-independent denominator, and
# every term reads only the poses inside a bounded dependency set.Grouping the
# moved objects into connected components of that dependency relation makes the
# family deltas exactly additive across components, so the single batch
# evaluation already contains the exact delta vector of every component.  The
# selector keeps the components that are non-inferior in every gated family; the
# union is then non-inferior by construction.
#
# Decision discipline.  The decomposition only proposes a subset.  The promoted
# subset is always re-measured by a confirmation evaluation and gated on that
# measurement, so an incomplete dependency model can cost selectivity but can
# never produce a false promotion.  Nothing in the frozen baseline is modified.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
SOURCE="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
BATCH="v5_sceneproof_settle_batch_probe_fix85"
SELECT="v5_sceneproof_settle_component_select_fix85"

MIN_TRANS="${SCENEPROOF_SETTLE_MIN_TRANSLATION_M:-0.05}"
GATED="${SCENEPROOF_SETTLE_GATED_FAMILIES:-collision,support,plane}"
DURATION="${SCENEPROOF_GLOBAL_SETTLE_DURATION:-2.0}"
SETTLE_TIMEOUT="${SCENEPROOF_GLOBAL_SETTLE_TIMEOUT:-3600}"
EVAL_TIMEOUT="${SCENEPROOF_EVAL_TIMEOUT:-1800}"
SURPLUS="${SCENEPROOF_SETTLE_ALLOW_SURPLUS_TRADING:-0}"

AUDIT_ROOT="$root/sceneba_audit/$SELECT"
PROBE_ROOT="${SCENEPROOF_PROBE_ROOT:-$root/sceneba_audit/v5_sceneproof_global_settle_smoke5}"
mkdir -p "$AUDIT_ROOT"

surplus_flag=""
if test "$SURPLUS" = "1"; then
    surplus_flag="--allow-surplus-trading"
fi

echo "=== SCENEPROOF FIX85: COMPONENT-ATTRIBUTED SETTLE PROMOTION ==="
echo "Scenes:       $SCENES"
echo "Baseline:     $BASELINE"
echo "Geometry:     $SOURCE"
echo "Batchver:    $BATCH"
echo "Select ver:   $SELECT"
echo "Probe root:   $PROBE_ROOT"
echo "Min settle:   ${MIN_TRANS} m"
echo "Gated:        $GATED"
echo "Start:        $(date)"

summary="$AUDIT_ROOT/_summary.txt"
: > "$summary"

for scene in $SCENES; do
    echo ""
    echo "--- $scene ($(date +%H:%M:%S)) ---"
    audit="$AUDIT_ROOT/$scene"
    mkdir -p "$audit"

    # Baseline placement, resolved inside this loop.  A stale placement variable
    # leaking across loops silently evaluates one scene's probes against another
    # scene's geometry, so it is recomputed at every use site.
    placement="$root/${scene}_${BASELINE}_result/S4_layout_refinement/${scene}_${BASELINE}_placement_info_s4.json"
    if ! test -s "$placement"; then
        echo "SKIP $scene: missing baseline placement $placement"
        echo "$scene SKIP missing_baseline_placement" >> "$summary"
        continue
    fi

    probe_dir="$PROBE_ROOT/$scene/probes"
    mkdir -p "$probe_dir"
    n_probes=$(find "$probe_dir" -maxdepth 1 -name '*.json' ! -name '_*' 2>/dev/null | wc -l)

    # ---- stage 1: bulk settle probes (cached) --------------------------------
    if test "$n_probes" -eq 0; then
        source_json=$(find "$root/${scene}_${SOURCE}_result/S3_pose_inference" -maxdepth 1 -name '*_placement_info.json' -print -quit)
        if test -z "$source_json"; then
            echo "SKIP $scene: no S3 placement for geometry version $SOURCE"
            echo "$scene SKIP missing_s3_placement" >> "$summary"
            continue
        fi
        echo "  [1/6] bulk settle (Blender) ..."
        timeout "$SETTLE_TIMEOUT" env \
            CUDA_VISIBLE_DEVICES=0 \
            IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
            IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
            IMAGINARIUM_SCENEPROOF_GLOBAL_SETTLE_AUDIT_OUTPUT="$probe_dir" \
            IMAGINARIUM_SCENEPROOF_GLOBAL_SETTLE_DURATION_SECONDS="$DURATION" \
            LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
            "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
              --obj_placement_info_json_path "$source_json" \
              --output_folder "/tmp/fix85_settle_${scene}_$$" \
              > "$audit/settle_blender.log" 2>&1 < /dev/null \
            || echo "  WARN: bulk settle returned non-zero, see $audit/settle_blender.log"
        n_probes=$(find "$probe_dir" -maxdepth 1 -name '*.json' ! -name '_*' 2>/dev/null | wc -l)
    else
        echo "  [1/6] bulk settle: $n_probes cached probes, reusing"
    fi
    if test "$n_probes" -eq 0; then
        echo "SKIP $scene: no probes"
        echo "$scene SKIP no_probes" >> "$summary"
        continue
    fi

    # ---- stage 2: materialize the all-moved batch candidate ------------------
    echo "  [2/6] materialize batch candidate ..."
    batch_dir="$root/${scene}_${BATCH}_result/S4_layout_refinement"
    batch_placement="$batch_dir/${scene}_${BATCH}_placement_info_s4.json"
    if ! "$python" sceneproof_settle_component_select_fix85.py materialize-batch \
        --incumbent "$placement" \
        --probe-dir "$probe_dir" \
        --min-translation-m "$MIN_TRANS" \
        --out "$batch_placement" > "$audit/materialize_batch.log" 2>&1; then
        echo "  FAIL materialize-batch, see $audit/materialize_batch.log"
        echo "$scene FAIL materialize_batch" >> "$summary"
        continue
    fi
    grep -E '^MOVED=' "$audit/materialize_batch.log" | sed 's/^/       /'

    # ---- stage 3: one batch physical evaluation ------------------------------
    echo "  [3/6] batch physical evaluation ..."
    manifest="$audit/scene_manifest.txt"
    printf '%s\n' "$scene" > "$manifest"
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

    # ---- stage 4: exact component attribution and subset selection -----------
    echo "  [4/6] component attribution and selection ..."
    select_dir="$root/${scene}_${SELECT}_result/S4_layout_refinement"
    select_placement="$select_dir/${scene}_${SELECT}_placement_info_s4.json"
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
        $surplus_flag \
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

    # ---- stage 5: one confirmation evaluation on the selected subset ---------
    echo "  [5/6] confirmation evaluation on $n_accepted objects ..."
    if ! timeout "$EVAL_TIMEOUT" "$python" eval_physical_realizability.py \
        --saved-results "$root" --scenes "$manifest" \
        --versions "$BASELINE,$SELECT" --geometry-version "$SOURCE" \
        --baseline-version "$BASELINE" \
        --metrics-out "$audit/confirm_physical.json" \
        --scene-csv "$audit/confirm_physical_scenes.csv" \
        --object-csv "$audit/confirm_physical_objects.csv" \
        --report-out "$audit/confirm_physical.txt" > "$audit/confirm_eval.log" 2>&1; then
        echo "  FAIL confirmation physical evaluation, see $audit/confirm_eval.log"
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
        echo "  WARN: GT evaluation failed, gating on physical families only"
    fi

    # ---- stage 6: gate on the measurement -----------------------------------
    echo "  [6/6] gate on measured confirmation ..."
    confirm_rc=0
    "$python" sceneproof_settle_component_select_fix85.py confirm \
        --scene "$scene" \
        --selection "$audit/selection.json" \
        --confirm-physical "$audit/confirm_physical.json" \
        $gt_flag \
        --baseline-version "$BASELINE" \
        --candidate-version "$SELECT" \
        --out "$audit/promotion.json" > "$audit/confirm.log" 2>&1 || confirm_rc=$?
    sed -n '/^  /,$p' "$audit/confirm.log" | sed 's/^/     /'
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
echo "FIX85 COMPONENT-ATTRIBUTED SETTLE COMPLETE  $(date)"
echo "----------------------------------------"
cat "$summary"
echo "----------------------------------------"
promoted_total=$(awk '$2=="PROMOTED" {sum += $3} END {print sum+0}' "$summary")
echo "Total promoted objects: $promoted_total"
echo "AUDIT_ROOT=$AUDIT_ROOT"
echo "Per-scene evidence: \$AUDIT_ROOT/<scene>/{selection.json,promotion.json}"
