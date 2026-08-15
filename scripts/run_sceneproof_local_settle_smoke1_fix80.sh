#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
scene="bedroom_01"
source_version="v4_deepsearch"
candidate="v5_sceneproof_pose_serialization_smoke1_fix76"
factor_root="$root/sceneba_audit/$candidate/true_mesh_com_smoke1_fix79"
action_audit="$factor_root/com_factorization_audit.json"
probe_root="$factor_root/local_settle_oracle_fix80"
log_root="logs/sceneproof_local_settle_smoke1_fix80"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
duration="${SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS:-1.0}"
timeout_seconds="${SCENEPROOF_LOCAL_SETTLE_TIMEOUT:-1800}"
placement="$root/${scene}_${candidate}_result/S4_layout_refinement/${scene}_${candidate}_placement_info_s4.json"
source_json="$(find "$root/${scene}_${source_version}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit)"

test -s "$action_audit" || { echo "Missing Fix79 action audit: $action_audit" >&2; exit 2; }
test -s "$placement" || { echo "Missing Fix76 placement: $placement" >&2; exit 2; }
test -s "$source_json" || { echo "Missing source S3 placement" >&2; exit 2; }
test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }
mkdir -p "$probe_root" "$log_root"

objects="/tmp/sceneproof_local_settle_fix80_objects.$$.txt"
list0="/tmp/sceneproof_local_settle_fix80_gpu0.$$.txt"
list1="/tmp/sceneproof_local_settle_fix80_gpu1.$$.txt"
cleanup() { rm -f -- "$objects" "$list0" "$list1"; }
trap cleanup EXIT

"$python" - "$action_audit" > "$objects" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1]))
for row in data.get("actionable_objects", []):
    if row.get("action") == "local_gravity_settle_probe_candidate":
        print(row["object_id"])
PY
awk 'NR % 2 == 1' "$objects" > "$list0"
awk 'NR % 2 == 0' "$objects" > "$list1"

run_worker() {
  local gpu="$1" object_list="$2"
  local object_id output scene_log tmp_output
  local worker_status=0
  while IFS= read -r object_id || test -n "$object_id"; do
    object_id="${object_id%$'\r'}"; test -n "$object_id" || continue
    output="$probe_root/${object_id}.json"
    scene_log="$log_root/${object_id}_gpu${gpu}.log"
    if test -s "$output"; then
      echo "CACHED_LOCAL_SETTLE object=$object_id gpu=$gpu"
      continue
    fi
    tmp_output="/tmp/sceneproof_local_settle_${object_id}_gpu${gpu}_$$"
    mkdir -p "$tmp_output"
    echo "START_LOCAL_SETTLE object=$object_id gpu=$gpu duration=$duration $(date)"
    if timeout "$timeout_seconds" env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
      IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
      IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_OBJECT_ID="$object_id" \
      IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS="$duration" \
      IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_AUDIT_OUTPUT="$output" \
      PYTHONUNBUFFERED=1 \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$blender" --background \
        --python modules/S4_blender_layout_and_corr.py -- \
        --obj_placement_info_json_path "$source_json" \
        --output_folder "$tmp_output" \
        > "$scene_log" 2>&1 < /dev/null; then
      test -s "$output" || worker_status=1
    else
      worker_status=1
      echo "FAIL_LOCAL_SETTLE object=$object_id gpu=$gpu log=$(readlink -f "$scene_log")" >&2
      grep -E 'Local gravity settle|Traceback|RuntimeError|ValueError|Error:' "$scene_log" | tail -40 || true
    fi
    rm -rf -- "$tmp_output"
  done < "$object_list"
  return "$worker_status"
}

worker_pids=()
run_worker 0 "$list0" > "$log_root/gpu0.log" 2>&1 & worker_pids+=("$!")
sleep 3
run_worker 1 "$list1" > "$log_root/gpu1.log" 2>&1 & worker_pids+=("$!")
echo "SETTLE_GPU0_PID=${worker_pids[0]}"
echo "SETTLE_GPU1_PID=${worker_pids[1]}"
echo "LOCAL_SETTLE_ORACLE candidate=$candidate policy=audit_only_process_isolated_full_so3"

status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || status=1; done

audit="$probe_root/oracle.json"
report="$probe_root/oracle.txt"
"$python" sceneproof_local_settle_oracle_fix80.py \
  --action-audit "$action_audit" \
  --probe-root "$probe_root" \
  --out "$audit" \
  --report "$report" || status=1

echo "FIX80_LOCAL_SETTLE_AUDIT=$(readlink -f "$audit")"
echo "FIX80_LOCAL_SETTLE_REPORT=$(readlink -f "$report")"
echo "FIX80_LOCAL_SETTLE_LOG_ROOT=$(readlink -f "$log_root")"
exit "$status"
