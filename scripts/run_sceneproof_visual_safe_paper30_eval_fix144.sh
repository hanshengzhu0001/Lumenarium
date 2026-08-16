#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="${SCENEPROOF_RESULTS_ROOT:-$HOME/Lumenarium/a10_reusable_results/paper30}"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
manifest="${SCENEPROOF_MANIFEST:-$root/manifest.txt}"
source="${SCENEPROOF_VISUAL_SAFE_SOURCE_VERSION:-v4_deepsearch}"
baseline="${SCENEPROOF_VISUAL_SAFE_BASELINE_VERSION:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
target="${SCENEPROOF_VISUAL_SAFE_TARGET_VERSION:-v5_sceneproof_visual_safe_paper30_fix144}"
geometry="${SCENEPROOF_VISUAL_SAFE_GEOMETRY_VERSION:-$source}"
audit="$root/sceneba_audit/$target"
log_root="$HOME/Lumenarium/logs/$target"
gpu0="${IMAGINARIUM_GPU0_ID:-0}"
gpu1="${IMAGINARIUM_GPU1_ID:-1}"
mkdir -p "$audit/scenes" "$log_root"

test -s "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 2; }
test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }

list0="$(mktemp /tmp/sceneproof_visual_safe_gpu0_XXXXXX)"
list1="$(mktemp /tmp/sceneproof_visual_safe_gpu1_XXXXXX)"
trap 'rm -f -- "$list0" "$list1"' EXIT
awk 'NR % 2 == 1' "$manifest" > "$list0"
awk 'NR % 2 == 0' "$manifest" > "$list1"

worker() {
  local gpu="$1" list="$2" scene source_json baseline_pose rollback_pose
  local output_dir placement scene_audit tmp start_ns end_ns elapsed rc worker_status=0
  local runtime="$log_root/runtime_gpu${gpu}.jsonl"
  : > "$runtime"
  while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"
    test -n "$scene" || continue
    source_json="$(find "$root/${scene}_${source}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null || true)"
    rollback_pose="$source_json"
    baseline_pose="$root/${scene}_${baseline}_result/S4_layout_refinement/${scene}_${baseline}_placement_info_s4.json"
    output_dir="$root/${scene}_${target}_result/S4_layout_refinement"
    placement="$output_dir/${scene}_${target}_placement_info_s4.json"
    scene_audit="$audit/scenes/${scene}.json"
    mkdir -p "$output_dir"
    if test -s "$placement" && test -s "$scene_audit" && [[ "${SCENEPROOF_VISUAL_SAFE_FORCE:-0}" != "1" ]]; then
      echo "CACHED scene=$scene gpu=$gpu"
      continue
    fi
    if ! test -s "$source_json" || ! test -s "$baseline_pose"; then
      echo "FAIL_INPUT scene=$scene gpu=$gpu source=$source_json baseline=$baseline_pose" >&2
      worker_status=1
      continue
    fi
    cp "$baseline_pose" "$placement"
    tmp="$(mktemp -d "/tmp/sceneproof_visual_safe_${scene}_XXXXXX")"
    echo "START scene=$scene gpu=$gpu $(date)"
    start_ns="$(date +%s%N)"
    set +e
    timeout 3600 env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
      IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
      IMAGINARIUM_SCENEPROOF_SPARSE_VERTICAL_CONTACT_AUDIT_OUTPUT="$scene_audit" \
      IMAGINARIUM_SCENEPROOF_SPARSE_VERTICAL_CONTACT_PLACEMENT_OUTPUT="$placement" \
      IMAGINARIUM_SCENEPROOF_SPARSE_ROLLBACK_PLACEMENT="$rollback_pose" \
      IMAGINARIUM_SCENEPROOF_SPARSE_CONTACT_TOLERANCE_M=0.02 \
      IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_SHIFT_M=0.5 \
      IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_TANGENT_SHIFT_M=0.15 \
      IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_PROGRAM_TANGENT_SHIFT_M=0.50 \
      IMAGINARIUM_SCENEPROOF_SPARSE_MINIMUM_HIT_FRACTION=0.10 \
      IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_SALVAGE=1 \
      IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_MAX_FLOOR_SHIFT_M=0.60 \
      IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_MAX_SUPPRESSED=4 \
      PYTHONUNBUFFERED=1 \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
        --obj_placement_info_json_path "$source_json" --output_folder "$tmp" \
        > "$log_root/${scene}_gpu${gpu}.log" 2>&1 < /dev/null
    rc=$?
    set -e
    end_ns="$(date +%s%N)"
    elapsed="$(awk -v a="$start_ns" -v b="$end_ns" 'BEGIN {printf "%.6f",(b-a)/1e9}')"
    rm -rf -- "$tmp"
    if test "$rc" -eq 0 && test -s "$placement" && test -s "$scene_audit"; then
      echo "DONE scene=$scene gpu=$gpu elapsed_s=$elapsed status=ok"
      status_text=ok
    else
      echo "DONE scene=$scene gpu=$gpu elapsed_s=$elapsed status=fail rc=$rc" >&2
      status_text=fail
      worker_status=1
    fi
    printf '{"scene":"%s","gpu":%s,"elapsed_seconds":%s,"status":"%s","return_code":%s}\n' \
      "$scene" "$gpu" "$elapsed" "$status_text" "$rc" >> "$runtime"
  done < "$list"
  return "$worker_status"
}

wall_start="$(date +%s%N)"
echo "VISUAL_SAFE_PAPER30_START baseline=$baseline target=$target source=$source"
worker "$gpu0" "$list0" > "$log_root/gpu0.log" 2>&1 & p0=$!
worker "$gpu1" "$list1" > "$log_root/gpu1.log" 2>&1 & p1=$!
echo "VISUAL_SAFE_GPU0_PID=$p0"
echo "VISUAL_SAFE_GPU1_PID=$p1"
status=0
wait "$p0" || status=1
wait "$p1" || status=1
wall_end="$(date +%s%N)"

"$python" eval_gt_metrics.py \
  --saved-results "$root" --scenes "$manifest" --versions "$baseline,$target" \
  --min-visible-mask-area 8000 --min-visible-bbox-size 0 --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" --manifest-out "$audit/gt_manifest_8000.json" \
  > "$audit/gt.log" 2>&1 || status=1

"$python" eval_physical_realizability.py \
  --saved-results "$root" --scenes "$manifest" --versions "$baseline,$target" \
  --geometry-version "$geometry" --baseline-version "$baseline" --collision-policy legacy \
  --metrics-out "$audit/physical.json" --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" --report-out "$audit/physical.txt" \
  > "$audit/physical.log" 2>&1 || status=1

"$python" - "$manifest" "$audit" "$log_root" "$baseline" "$target" "$wall_start" "$wall_end" <<'PY'
import json, sys
from pathlib import Path

manifest, audit, logs = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
baseline, target = sys.argv[4:6]
wall_s = (int(sys.argv[7]) - int(sys.argv[6])) / 1e9
scenes = [x.strip() for x in manifest.read_text().splitlines() if x.strip()]

runtime_rows = []
for path in logs.glob("runtime_gpu*.jsonl"):
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("status") == "ok":
            runtime_rows.append(row)

repaired = relocated = suppressed = unresolved = 0
failed_scenes = []
for scene in scenes:
    path = audit / "scenes" / f"{scene}.json"
    if not path.is_file():
        failed_scenes.append(scene)
        continue
    row = json.load(open(path))
    visual = row.get("visual_safe_salvage", {})
    repaired += len(row.get("repaired_object_ids", []))
    relocated += len(visual.get("floor_relocated_object_ids", []))
    suppressed += len(visual.get("render_suppressed_object_ids", []))
    unresolved += len(visual.get("unresolved_object_ids", row.get("unresolved_object_ids", [])))

gt = json.load(open(audit / "gt_8000.json"))["versions"]
physical = json.load(open(audit / "physical.json"))["versions"]

def pose(version):
    row = gt[version]
    primary = row.get("primary_metrics", {})
    return {
        "primary_recovery": primary.get("object_recovery"),
        "primary_parent": primary.get("parent_accuracy"),
        "rotation_auc60": primary.get("rotation_auc60_aligned"),
        "translation_auc05": primary.get("translation_auc05_aligned"),
    }

def phys(version):
    agg = physical[version]["aggregate"]
    families = agg.get("families", {})
    return {
        "physical_macro": agg.get("headline_macro_realizability"),
        "collision": families.get("collision", {}).get("score"),
        "support": families.get("support", {}).get("score"),
        "plane": families.get("plane", {}).get("score"),
        "semantic": families.get("semantic", {}).get("score"),
    }

incremental_mean = (
    sum(float(x["elapsed_seconds"]) for x in runtime_rows) / len(runtime_rows)
    if runtime_rows else None
)
s03_mean = 636.949
fix61_mean = 192.930
full_mean = None if incremental_mean is None else s03_mean + fix61_mean + incremental_mean
out = {
    "schema_version": "sceneproof_visual_safe_paper30_eval_v1",
    "protocol": "Paper30 Primary objects >=8000 visible pixels; common legacy physical evaluator",
    "baseline": baseline,
    "target": target,
    "scenes": len(scenes),
    "failed_scenes": failed_scenes,
    "baseline_metrics": {**pose(baseline), **phys(baseline)},
    "visual_safe_metrics": {**pose(target), **phys(target)},
    "visual_safe_actions": {
        "repaired_objects": repaired,
        "floor_relocated_objects": relocated,
        "render_suppressed_objects": suppressed,
        "unresolved_objects": unresolved,
    },
    "timing_seconds": {
        "s0_s3_mean": s03_mean,
        "fix61_mean": fix61_mean,
        "visual_safe_incremental_mean": incremental_mean,
        "visual_safe_two_a10_wall": wall_s,
        "visual_safe_full_s0_s4_mean": full_mean,
    },
    "reporting_note": (
        "Recovery and pose are placement-level. Render suppression changes presentation visibility "
        "and therefore remains ineligible for the main paper metric table."
    ),
}
(audit / "final_eval.json").write_text(json.dumps(out, indent=2) + "\n")

def pct(value):
    return "n/a" if value is None else f"{100.0 * value:.2f}%"

print(f"SCENES={len(scenes)-len(failed_scenes)}/{len(scenes)} FAILURES={len(failed_scenes)}")
print(f"VISUAL_SAFE_PRIMARY_RECOVERY={pct(out['visual_safe_metrics']['primary_recovery'])}")
print(f"VISUAL_SAFE_PRIMARY_PARENT={pct(out['visual_safe_metrics']['primary_parent'])}")
print(f"VISUAL_SAFE_PHYSICAL_MACRO={pct(out['visual_safe_metrics']['physical_macro'])}")
print(f"VISUAL_SAFE_ROTATION_AUC60={pct(out['visual_safe_metrics']['rotation_auc60'])}")
print(f"VISUAL_SAFE_TRANSLATION_AUC05={pct(out['visual_safe_metrics']['translation_auc05'])}")
print(f"VISUAL_SAFE_ACTIONS={json.dumps(out['visual_safe_actions'], sort_keys=True)}")
print(f"VISUAL_SAFE_INCREMENTAL_MEAN_SECONDS={incremental_mean:.3f}")
print(f"VISUAL_SAFE_FULL_S0_S4_MEAN_SECONDS={full_mean:.3f}")
print(f"VISUAL_SAFE_TWO_A10_WALL_SECONDS={wall_s:.3f}")
print(f"VISUAL_SAFE_FINAL_EVAL={(audit/'final_eval.json').resolve()}")
PY

echo "VISUAL_SAFE_PAPER30_FINISHED status=$status"
exit "$status"
