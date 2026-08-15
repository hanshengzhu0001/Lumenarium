#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

manifest="${SCENEPROOF_MANIFEST:-a10_reusable_results/paper30/manifest.txt}"
root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
source_version="${SCENEPROOF_RENDER_SOURCE_VERSION:-v4_deepsearch}"
target="${SCENEPROOF_CERTIFIED_VERSION:-v5_sceneproof_postsim_component_certified_paper30_fix25}"
log_root="${SCENEPROOF_RENDER_LOG_ROOT:-logs/${target}_locked_camera_render}"
gpu0_id="${IMAGINARIUM_GPU0_ID:-0}"
gpu1_id="${IMAGINARIUM_GPU1_ID:-1}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
samples="${SCENEPROOF_RENDER_SAMPLES:-256}"
timeout_seconds="${SCENEPROOF_RENDER_TIMEOUT:-1800}"
gpu_floor="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}"
force_render="${SCENEPROOF_RENDER_FORCE:-0}"

test -f "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 2; }
test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }
mkdir -p "$log_root"

list0="/tmp/sceneproof_render_gpu0.$$.txt"
list1="/tmp/sceneproof_render_gpu1.$$.txt"
awk 'NR % 2 == 1' "$manifest" > "$list0"
awk 'NR % 2 == 0' "$manifest" > "$list1"
worker_pids=()

cleanup() {
  rm -f -- "$list0" "$list1"
}
stop_workers() {
  if ((${#worker_pids[@]})); then
    kill "${worker_pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'stop_workers; cleanup; exit 130' INT TERM

source_pose() {
  local scene="$1"
  find "$root/${scene}_${source_version}_result/S3_pose_inference" \
    -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null
}

run_worker() {
  local gpu="$1" scene_list="$2"
  local scene source_json target_dir placement render audit scene_log runtime_log
  local free start_ns end_ns elapsed rc status_text tmp_output worker_status=0
  runtime_log="$log_root/runtime_gpu${gpu}.jsonl"
  touch "$runtime_log"
  while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"
    test -n "$scene" || continue
    source_json="$(source_pose "$scene")"
    target_dir="$root/${scene}_${target}_result/S4_layout_refinement"
    placement="$target_dir/${scene}_${target}_placement_info_s4.json"
    render="$target_dir/${scene}_${target}_render_simu.png"
    audit="$target_dir/${scene}_${target}_render_camera.json"
    scene_log="$log_root/${scene}_gpu${gpu}.log"

    if test "$force_render" != "1" \
      && test -s "$render" && test -s "$audit" \
      && test "$render" -nt "$placement" \
      && test "$audit" -nt "$placement"; then
      echo "CACHED_RENDER scene=$scene gpu=$gpu"
      continue
    fi
    if test -s "$render" || test -s "$audit"; then
      echo "STALE_OR_FORCED_RENDER scene=$scene gpu=$gpu force=$force_render"
    fi
    if ! test -s "$source_json" || ! test -s "$placement"; then
      echo "FAIL_RENDER scene=$scene gpu=$gpu reason=missing_source_or_certified_pose" >&2
      worker_status=1
      continue
    fi
    while true; do
      free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
      if test -n "$free" && test "$free" -ge "$gpu_floor"; then break; fi
      echo "WAIT_GPU_RENDER scene=$scene gpu=$gpu free=${free:-unknown}MiB $(date)"
      sleep 60
    done

    tmp_output="/tmp/sceneproof_locked_render_${scene}_gpu${gpu}_$$"
    mkdir -p "$tmp_output"
    echo "START_RENDER scene=$scene gpu=$gpu samples=$samples $(date)"
    start_ns="$(date +%s%N)"
    timeout "$timeout_seconds" env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
      IMAGINARIUM_S4_RENDER_ONLY_OUTPUT="$render" \
      IMAGINARIUM_S4_RENDER_ONLY_AUDIT="$audit" \
      IMAGINARIUM_S4_RENDER_ONLY_SAMPLES="$samples" \
      PYTHONUNBUFFERED=1 \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$blender" --background \
        --python modules/S4_blender_layout_and_corr.py -- \
        --obj_placement_info_json_path "$source_json" \
        --output_folder "$tmp_output" \
        > "$scene_log" 2>&1 < /dev/null
    rc=$?
    end_ns="$(date +%s%N)"
    elapsed="$(awk -v start="$start_ns" -v end="$end_ns" 'BEGIN { printf "%.6f", (end-start)/1000000000.0 }')"
    rm -rf -- "$tmp_output"
    if test -s "$render" && test -s "$audit"; then
      status_text=ok
      echo "OK_RENDER scene=$scene gpu=$gpu elapsed_seconds=$elapsed $(date)"
    else
      status_text=fail
      worker_status=1
      echo "FAIL_RENDER scene=$scene gpu=$gpu rc=$rc log=$scene_log $(date)"
      grep -E \
        'SceneProof render-only|Traceback|RuntimeError|ValueError|FileNotFoundError|Error:' \
        "$scene_log" | tail -20 || true
    fi
    printf '{"scene":"%s","version":"%s_render","engine":"blender","stage":"locked_camera_render","gpu":%s,"elapsed_seconds":%s,"status":"%s","return_code":%s}\n' \
      "$scene" "$target" "$gpu" "$elapsed" "$status_text" "$rc" >> "$runtime_log"
  done < "$scene_list"
  return "$worker_status"
}

run_worker "$gpu0_id" "$list0" > "$log_root/gpu0.log" 2>&1 & worker_pids+=("$!")
sleep 10
run_worker "$gpu1_id" "$list1" > "$log_root/gpu1.log" 2>&1 & worker_pids+=("$!")
echo "RENDER_GPU0_PID=${worker_pids[0]}"
echo "RENDER_GPU1_PID=${worker_pids[1]}"
echo "RENDER_TARGET=$target CAMERA_POLICY=source_s3_scene_camera_locked SAMPLES=$samples"

status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || status=1; done
expected="$(grep -cve '^[[:space:]]*$' "$manifest")"
completed=0
while IFS= read -r scene || test -n "$scene"; do
  scene="${scene%$'\r'}"; test -n "$scene" || continue
  test -s "$root/${scene}_${target}_result/S4_layout_refinement/${scene}_${target}_render_simu.png" \
    && completed=$((completed + 1))
done < "$manifest"
test "$completed" -eq "$expected" || status=1
echo "RENDER_FINISHED completed=$completed/$expected status=$status"
exit "$status"
