#!/usr/bin/env bash
# SceneProof: global gravity settle — drop ALL non-structural objects in one
# Bullet sim per scene and record everyone who moved.
#
# SUPERSEDED for promotion decisions by
# scripts/run_sceneproof_settle_component_select_smoke5_fix85.sh.
#
# Two defects made every promotion verdict produced by the original version of
# this script meaningless, and both are fixed below:
#
#   1. The candidate placement wrote the settled pose to "matrix_after_settle".
#      No evaluator or renderer reads that key; both read
#      "pose_matrix_for_blender".  The candidate was therefore semantically
#      identical to the baseline, so every measured delta was zero by
#      construction.
#   2. The "$placement" variable was assigned in the probe loop and reused in the
#      evaluation loop, where it still held the last scene's path.  Four of five
#      scenes were materialized from the wrong scene's baseline.
#
# The gate below is a SCENE-level gate, not a per-object gate: it compares scene
# aggregates, so every object in a scene shares one verdict.  Per-object
# discrimination requires the component attribution in Fix85.
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"

SCENES="${SMOKE5_SCENES:-bedroom_01 livingroom_10 casino_01 official_01 streelitter_01}"
BASELINE="${SCENEPROOF_BASELINE:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
CANDIDATE="v5_sceneproof_global_settle_smoke5"
SOURCE="v4_deepsearch"
DURATION="${SCENEPROOF_GLOBAL_SETTLE_DURATION:-2.0}"
TIMEOUT="${SCENEPROOF_GLOBAL_SETTLE_TIMEOUT:-3600}"

AUDIT_ROOT="$root/sceneba_audit/$CANDIDATE"
mkdir -p "$AUDIT_ROOT"

echo "=== GLOBAL GRAVITY SETTLE: $SCENES ==="
echo "Baseline: $BASELINE"

for scene in $SCENES; do
    echo ""
    echo "--- $scene ---"
    probe_dir="$AUDIT_ROOT/$scene/probes"
    mkdir -p "$probe_dir"

    placement="$root/${scene}_${BASELINE}_result/S4_layout_refinement/${scene}_${BASELINE}_placement_info_s4.json"
    test -s "$placement" || { echo "SKIP: no placement"; continue; }

    # Skip if probes already cached
    n_existing=$(ls "$probe_dir"/*.json 2>/dev/null | grep -v _manifest | wc -l)
    if test "$n_existing" -gt 0; then
        echo "PROBES $scene: $n_existing (cached, skipping Blender)"
        continue
    fi

    source_json="$(find "$root/${scene}_${SOURCE}_result/S3_pose_inference" -maxdepth 1 -name '*_placement_info.json' -print -quit)"
    scene_log="$AUDIT_ROOT/$scene/blender.log"

    echo "START $scene $(date)"
    timeout "$TIMEOUT" env \
        CUDA_VISIBLE_DEVICES=0 \
        IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
        IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
        IMAGINARIUM_SCENEPROOF_GLOBAL_SETTLE_AUDIT_OUTPUT="$probe_dir" \
        IMAGINARIUM_SCENEPROOF_GLOBAL_SETTLE_DURATION_SECONDS="$DURATION" \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
          --obj_placement_info_json_path "$source_json" \
          --output_folder /tmp/global_settle_${scene}_$$ \
          > "$scene_log" 2>&1 < /dev/null || echo "FAIL $scene" >> /tmp/global_settle_failures.txt

    n_probes=$(ls "$probe_dir"/*.json 2>/dev/null | grep -v _manifest | wc -l)
    echo "PROBES $scene: $n_probes"
done

echo ""
echo "=== EVALUATION: physical + GT ==="
for scene in $SCENES; do
    probe_dir="$AUDIT_ROOT/$scene/probes"
    n_probes=$(ls "$probe_dir"/*.json 2>/dev/null | grep -v _manifest | wc -l || echo 0)
    test "$n_probes" -gt 0 || continue

    # Recomputed here on purpose: reusing the probe loop's value evaluates this
    # scene's probes against the last scene's baseline placement.
    placement="$root/${scene}_${BASELINE}_result/S4_layout_refinement/${scene}_${BASELINE}_placement_info_s4.json"
    test -s "$placement" || { echo "SKIP $scene: no baseline placement"; continue; }

    # materialize ONE candidate placement with ALL settled objects
    cand_dir="$root/${scene}_${CANDIDATE}_result/S4_layout_refinement"
    cand_placement="$cand_dir/${scene}_${CANDIDATE}_placement_info_s4.json"
    mkdir -p "$cand_dir"

    # materialize: take baseline placement, apply only probes with translation > 5cm
    "$python" - "$placement" "$probe_dir" "$cand_placement" <<'PY'
import json, sys, os, glob
from pathlib import Path
baseline = json.load(open(sys.argv[1]))
probe_dir = Path(sys.argv[2])
out = sys.argv[3]
MIN_TRANS = 0.05  # only commit objects that moved > 5 cm
committed = 0
for fname in sorted(probe_dir.glob("*.json")):
    if fname.name.startswith("_"):
        continue
    try:
        probe = json.loads(fname.read_text(encoding="utf-8"))
    except Exception:
        continue
    if probe.get("translation_delta_m", 0) < MIN_TRANS:
        continue
    oid = probe.get("object_id", "")
    m = probe.get("settled_pose_matrix")
    if oid in baseline.get("obj_info", {}) and m:
        # "pose_matrix_for_blender" is the only pose key the evaluators and the
        # renderer read.  Writing any other key yields a candidate that differs
        # byte-wise but not semantically from the baseline.
        baseline["obj_info"][oid]["pose_matrix_for_blender"] = m
        committed += 1
json.dump(baseline, open(out, "w"), indent=2)
print(f"  Materialized {committed} objects (>5cm) from {sys.argv[2]}", flush=True)
PY

    manifest="/tmp/gs_${scene}.txt"; printf '%s\n' "$scene" > "$manifest"
    audit="$AUDIT_ROOT/$scene"
    "$python" eval_physical_realizability.py \
      --saved-results "$root" --scenes "$manifest" \
      --versions "$BASELINE,$CANDIDATE" --geometry-version "$SOURCE" \
      --baseline-version "$BASELINE" \
      --metrics-out "$audit/physical.json" \
      --scene-csv "$audit/physical_scenes.csv" \
      --object-csv "$audit/physical_objects.csv" \
      --report-out "$audit/physical.txt" 2>/dev/null || echo "FAIL_PHYS $scene"

    "$python" eval_gt_metrics.py \
      --saved-results "$root" --scenes "$manifest" \
      --versions "$BASELINE,$CANDIDATE" \
      --min-visible-mask-area 8000 --min-visible-bbox-size 0 \
      --batch-logs logs \
      --metrics-out "$audit/gt_8000.json" \
      --manifest-out "$audit/gt_manifest_8000.json" 2>/dev/null || echo "FAIL_GT $scene"
    rm -f "$manifest"
done

echo ""
echo "=== SCENE-LEVEL GATES (one verdict per scene, replicated per object) ==="
PROMOTED=0
for scene in $SCENES; do
    probe_dir="$AUDIT_ROOT/$scene/probes"
    for probe_file in "$probe_dir"/*.json; do
        test -f "$probe_file" || continue
        echo "$probe_file" | grep -q "_manifest" && continue
        oid=$(basename "$probe_file" .json)
        passed=$("$python" - "$probe_file" "$AUDIT_ROOT/$scene/physical.json" "$AUDIT_ROOT/$scene/gt_8000.json" "$BASELINE" "$CANDIDATE" <<'PY'
import json, sys
probe = json.load(open(sys.argv[1]))
physical = json.load(open(sys.argv[2]))
gt = json.load(open(sys.argv[3]))
baseline = sys.argv[4]
candidate = sys.argv[5]

trans = probe.get("translation_delta_m", 0)
moved = trans > 0.05  # 5 cm threshold (eliminates sub-cm jitter)
if not moved:
    print("False"); sys.exit(0)

# physical deltas (scene-level)
b = physical["versions"][baseline]["aggregate"]
c = physical["versions"][candidate]["aggregate"]
for fam in ("collision", "support", "plane"):
    bs = b["families"].get(fam, {}).get("score")
    cs = c["families"].get(fam, {}).get("score")
    if bs is not None and cs is not None and cs < bs - 1e-6:
        print("False"); sys.exit(0)

# GT deltas
v1 = gt["versions"][baseline]
v2 = gt["versions"][candidate]
for key in ("rotation_auc60_aligned", "rotation_auc60"):
    if key in v1 and key in v2:
        if v2[key] < v1[key] - 0.005:
            print("False"); sys.exit(0)
for key in ("translation_auc05_aligned", "translation_auc05"):
    if key in v1 and key in v2:
        if v2[key] < v1[key] - 0.001:
            print("False"); sys.exit(0)

print("True")
PY
)
        echo "  $scene/$oid: passed=$passed"
        if test "$passed" = "True"; then
            PROMOTED=$((PROMOTED + 1))
        fi
    done
done

echo ""
echo "========================================"
echo "GLOBAL SETTLE SMOKE5 COMPLETE"
echo "Promoted objects: $PROMOTED"
echo "Failures: $(cat /tmp/global_settle_failures.txt 2>/dev/null || echo none)"
echo "AUDIT_ROOT=$AUDIT_ROOT"
