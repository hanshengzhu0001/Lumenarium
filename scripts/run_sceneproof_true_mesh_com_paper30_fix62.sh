#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

manifest="${SCENEPROOF_COM_MANIFEST:-a10_reusable_results/paper30/manifest.txt}"
results_root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
source_version="${SCENEPROOF_COM_SOURCE_VERSION:-v4_deepsearch}"
smooth="${SCENEPROOF_COM_SMOOTH_VERSION:-v5_sceneproof_fix43_smooth_paper30_fix61}"
final="${SCENEPROOF_COM_FINAL_VERSION:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
audit_root="${SCENEPROOF_COM_AUDIT_ROOT:-$results_root/sceneba_audit/${final}/true_mesh_com_fix62}"
physical_objects="${SCENEPROOF_COM_PHYSICAL_OBJECTS:-$results_root/sceneba_audit/${final}/physical_objects.csv}"
log_root="${SCENEPROOF_COM_LOG_ROOT:-logs/sceneproof_true_mesh_com_paper30_fix62}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
timeout_seconds="${SCENEPROOF_COM_TIMEOUT:-1800}"

test -f "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 2; }
test -f "$physical_objects" || { echo "Missing physical objects: $physical_objects" >&2; exit 2; }
test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }
mkdir -p "$audit_root" "$log_root"

list0="/tmp/sceneproof_com_gpu0.$$.txt"
list1="/tmp/sceneproof_com_gpu1.$$.txt"
awk 'NR % 2 == 1' "$manifest" > "$list0"
awk 'NR % 2 == 0' "$manifest" > "$list1"
worker_pids=()

cleanup() { rm -f -- "$list0" "$list1"; }
stop_workers() {
  if ((${#worker_pids[@]})); then kill "${worker_pids[@]}" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'stop_workers; cleanup; exit 130' INT TERM

source_pose() {
  local scene="$1"
  find "$results_root/${scene}_${source_version}_result/S3_pose_inference" \
    -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null
}

placement() {
  local scene="$1" version="$2"
  printf '%s/%s_%s_result/S4_layout_refinement/%s_%s_placement_info_s4.json' \
    "$results_root" "$scene" "$version" "$scene" "$version"
}

run_worker() {
  local gpu="$1" scene_list="$2"
  local scene version source_json target_json audit_json scene_log tmp_output rc
  local worker_status=0
  while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"; test -n "$scene" || continue
    source_json="$(source_pose "$scene")"
    for version in "$smooth" "$final"; do
      target_json="$(placement "$scene" "$version")"
      audit_json="$audit_root/${scene}__${version}.json"
      scene_log="$log_root/${scene}_${version}_gpu${gpu}.log"
      if test -s "$audit_json"; then
        echo "CACHED_COM_AUDIT scene=$scene version=$version gpu=$gpu"
        continue
      fi
      if ! test -s "$source_json" || ! test -s "$target_json"; then
        echo "FAIL_COM_AUDIT scene=$scene version=$version reason=missing_input" >&2
        worker_status=1
        continue
      fi
      tmp_output="/tmp/sceneproof_com_${scene}_${version}_gpu${gpu}_$$"
      mkdir -p "$tmp_output"
      echo "START_COM_AUDIT scene=$scene version=$version gpu=$gpu $(date)"
      timeout "$timeout_seconds" env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$target_json" \
        IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
        IMAGINARIUM_SCENEPROOF_TRUE_MESH_COM_AUDIT_OUTPUT="$audit_json" \
        PYTHONUNBUFFERED=1 \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$blender" --background \
          --python modules/S4_blender_layout_and_corr.py -- \
          --obj_placement_info_json_path "$source_json" \
          --output_folder "$tmp_output" \
          > "$scene_log" 2>&1 < /dev/null
      rc=$?
      rm -rf -- "$tmp_output"
      if test "$rc" -eq 0 && test -s "$audit_json"; then
        echo "OK_COM_AUDIT scene=$scene version=$version gpu=$gpu"
      else
        worker_status=1
        echo "FAIL_COM_AUDIT scene=$scene version=$version gpu=$gpu rc=$rc log=$scene_log" >&2
        grep -E 'True-mesh COM|Traceback|RuntimeError|ValueError|Error:' "$scene_log" | tail -30 || true
      fi
    done
  done < "$scene_list"
  return "$worker_status"
}

run_worker 0 "$list0" > "$log_root/gpu0.log" 2>&1 & worker_pids+=("$!")
sleep 5
run_worker 1 "$list1" > "$log_root/gpu1.log" 2>&1 & worker_pids+=("$!")
echo "COM_GPU0_PID=${worker_pids[0]}"
echo "COM_GPU1_PID=${worker_pids[1]}"
echo "COM_AUDIT smooth=$smooth final=$final policy=audit_only_true_mesh"

status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || status=1; done

out="$audit_root/responsibility.json"
report="$audit_root/responsibility.txt"
"$python" sceneproof_true_mesh_com_responsibility_audit.py \
  --manifest "$manifest" \
  --audit-root "$audit_root" \
  --smooth-version "$smooth" \
  --final-version "$final" \
  --physical-objects "$physical_objects" \
  --out "$out" --report "$report" || status=1

cat "$report" 2>/dev/null || true
echo "TRUE_MESH_COM_AUDIT=$(readlink -f "$out")"
echo "TRUE_MESH_COM_REPORT=$(readlink -f "$report")"
exit "$status"
