#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
baseline="${SCENEPROOF_BASELINE_VERSION:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
source="${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
target="${SCENEPROOF_RIGID_TARGET_VERSION:-v5_sceneproof_rigid_only_adaptive_paper30_fix84e}"
audit="$root/sceneba_audit/$target"
log_root="logs/$target"
paper_manifest="$root/manifest.txt"
target_manifest="$audit/rigid_targets.tsv"
scene_manifest="$audit/affected_scenes.txt"
selected_tsv="$audit/per_object_trials.tsv"
certificate="$audit/certificate.json"
timeout_seconds="${SCENEPROOF_RIGID_SETTLE_TIMEOUT:-1800}"
duration="${SCENEPROOF_RIGID_SETTLE_DURATION_SECONDS:-1.0}"

test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }
test -s "$paper_manifest" || { echo "Missing Paper30 manifest: $paper_manifest" >&2; exit 2; }
mkdir -p "$audit" "$log_root"
if test -n "${SCENEPROOF_RIGID_TARGETS_FILE:-}"; then
  cp "$SCENEPROOF_RIGID_TARGETS_FILE" "$target_manifest"
else
cat > "$target_manifest" <<'EOF'
bedroom_01	single_sofa_chair_1
livingroom_10	single_sofa_chair_0
casino_02	casino_chair_0
official_01	office_chair_5
EOF
fi
cut -f1 "$target_manifest" | awk '!seen[$0]++' > "$scene_manifest"

list0="/tmp/sceneproof_rigid_fix84e_gpu0.$$.tsv"
list1="/tmp/sceneproof_rigid_fix84e_gpu1.$$.tsv"
awk 'NR % 2 == 1' "$target_manifest" > "$list0"
awk 'NR % 2 == 0' "$target_manifest" > "$list1"
cleanup() { rm -f -- "$list0" "$list1"; }
trap cleanup EXIT

run_worker() {
  local gpu="$1"
  local input="$2"
  local output="$audit/worker${gpu}.tsv"
  : > "$output"
  while IFS=$'\t' read -r scene object_id || test -n "${scene:-}"; do
    test -n "${scene:-}" || continue
    local placement source_json object_audit probe profile route elapsed_total=0
    placement="$root/${scene}_${baseline}_result/S4_layout_refinement/${scene}_${baseline}_placement_info_s4.json"
    source_json="$(find "$root/${scene}_${source}_result/S3_pose_inference" -maxdepth 1 -type f -name '*_placement_info.json' -print -quit 2>/dev/null || true)"
    object_audit="$audit/$scene/$object_id"
    mkdir -p "$object_audit/probes"
    if ! test -s "$placement" || ! test -s "$source_json"; then
      printf '%s\t%s\tmissing_input\tfalse\tfalse\tfalse\t0\t-\t-\t-\n' "$scene" "$object_id" >> "$output"
      continue
    fi

    run_profile() {
      local name="$1" linear="$2" angular="$3" active_friction="$4" passive_friction="$5"
      local start_ns end_ns tmp rc=0
      probe="$object_audit/probes/${name}.json"
      if test -s "$probe"; then return 0; fi
      tmp="/tmp/sceneproof_rigid_${scene}_${object_id}_${name}_gpu${gpu}_$$"
      mkdir -p "$tmp"
      start_ns="$(date +%s%N)"
      timeout "$timeout_seconds" env -u IMAGINARIUM_S4_RENDER_ONLY_CAMERA_TARGET \
        CUDA_VISIBLE_DEVICES="$gpu" \
        IMAGINARIUM_S4_RENDER_ONLY_PLACEMENT="$placement" \
        IMAGINARIUM_S4_RENDER_ONLY_SKIP_RENDER=1 \
        IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_OBJECT_ID="$object_id" \
        IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_DURATION_SECONDS="$duration" \
        IMAGINARIUM_SCENEPROOF_LOCAL_SETTLE_AUDIT_OUTPUT="$probe" \
        IMAGINARIUM_SETTLE_LINEAR_DAMPING="$linear" \
        IMAGINARIUM_SETTLE_ANGULAR_DAMPING="$angular" \
        IMAGINARIUM_SETTLE_FRICTION="$active_friction" \
        IMAGINARIUM_SETTLE_PASSIVE_FRICTION="$passive_friction" \
        IMAGINARIUM_SETTLE_SUBSTEPS=10 \
        IMAGINARIUM_SETTLE_SOLVER_ITERATIONS=10 \
        PYTHONUNBUFFERED=1 \
        LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "$blender" --background --python modules/S4_blender_layout_and_corr.py -- \
          --obj_placement_info_json_path "$source_json" \
          --output_folder "$tmp" \
          > "$object_audit/${name}_gpu${gpu}.log" 2>&1 < /dev/null || rc=$?
      end_ns="$(date +%s%N)"
      elapsed_total="$(awk -v total="$elapsed_total" -v a="$start_ns" -v b="$end_ns" 'BEGIN {printf "%.6f", total+(b-a)/1e9}')"
      rm -rf -- "$tmp"
      test "$rc" -eq 0 && test -s "$probe"
    }

    profile="primary_08_f100"
    if ! run_profile "$profile" 0.8 0.8 100 100; then
      printf '%s\t%s\t%s\tfalse\tfalse\tfalse\t%s\t-\t-\t-\n' "$scene" "$object_id" "$profile" "$elapsed_total" >> "$output"
      continue
    fi
    route="$($python sceneproof_rigid_settle_adaptive_eval_fix84.py route --probe "$probe" --stage primary)"
    if test "$route" = "retry_damping"; then
      profile="damping_05_f100"
      if run_profile "$profile" 0.5 0.5 100 100; then
        route="$($python sceneproof_rigid_settle_adaptive_eval_fix84.py route --probe "$probe" --stage damping)"
      else
        route=reject
      fi
    fi
    if test "$route" = "retry_friction"; then
      profile="damping_05_f05"
      if run_profile "$profile" 0.5 0.5 0.5 0.5; then
        route="$($python sceneproof_rigid_settle_adaptive_eval_fix84.py route --probe "$probe" --stage friction)"
      else
        route=reject
      fi
    fi
    if test "$route" != "select" && test "${SCENEPROOF_FORCE_MEASURED_CANDIDATES:-0}" != "1"; then
      printf '%s\t%s\t%s\tfalse\tfalse\tfalse\t%s\t%s\t-\t-\n' "$scene" "$object_id" "$profile" "$elapsed_total" "$(readlink -f "$probe")" >> "$output"
      continue
    fi

    local candidate="v5_sceneproof_rigid_${scene}_${object_id}_fix84e"
    local candidate_dir="$root/${scene}_${candidate}_result/S4_layout_refinement"
    local candidate_json="$candidate_dir/${scene}_${candidate}_placement_info_s4.json"
    local one_manifest="$object_audit/scene.txt"
    printf '%s\n' "$scene" > "$one_manifest"
    local materialize_extra=()
    if test "${SCENEPROOF_FORCE_MEASURED_CANDIDATES:-0}" = "1"; then
      materialize_extra+=(--allow-measured-candidate)
    fi
    if ! "$python" sceneproof_local_settle_materialize_fix82.py \
      --incumbent "$placement" --probe "$probe" --out "$candidate_json" \
      "${materialize_extra[@]}" \
      > "$object_audit/materialize.log" 2>&1; then
      printf '%s\t%s\t%s\tfalse\tfalse\tfalse\t%s\t%s\t-\t%s\n' \
        "$scene" "$object_id" "$profile" "$elapsed_total" \
        "$(readlink -f "$probe")" "$(readlink -f "$object_audit/materialize.log")" >> "$output"
      continue
    fi
    if ! "$python" eval_physical_realizability.py \
      --saved-results "$root" --scenes "$one_manifest" \
      --versions "$baseline,$candidate" --geometry-version "$source" \
      --baseline-version "$baseline" \
      --metrics-out "$object_audit/physical.json" \
      --scene-csv "$object_audit/physical_scenes.csv" \
      --object-csv "$object_audit/physical_objects.csv" \
      --report-out "$object_audit/physical.txt" \
      > "$object_audit/physical.log" 2>&1; then
      printf '%s\t%s\t%s\tfalse\tfalse\tfalse\t%s\t%s\t%s\t%s\n' \
        "$scene" "$object_id" "$profile" "$elapsed_total" \
        "$(readlink -f "$probe")" "$candidate" "$(readlink -f "$object_audit/physical.log")" >> "$output"
      continue
    fi
    if ! "$python" eval_gt_metrics.py \
      --saved-results "$root" --scenes "$one_manifest" \
      --versions "$baseline,$candidate" --min-visible-mask-area 8000 \
      --min-visible-bbox-size 0 --batch-logs logs \
      --metrics-out "$object_audit/gt_8000.json" \
      --manifest-out "$object_audit/gt_manifest_8000.json" \
      > "$object_audit/gt.log" 2>&1; then
      printf '%s\t%s\t%s\tfalse\tfalse\tfalse\t%s\t%s\t%s\t%s\n' \
        "$scene" "$object_id" "$profile" "$elapsed_total" \
        "$(readlink -f "$probe")" "$candidate" "$(readlink -f "$object_audit/gt.log")" >> "$output"
      continue
    fi
    "$python" sceneproof_local_settle_component_gate_fix84.py \
      --probe "$probe" --physical "$object_audit/physical.json" \
      --physical-objects "$object_audit/physical_objects.csv" \
      --gt "$object_audit/gt_8000.json" \
      --incumbent-version "$baseline" --candidate-version "$candidate" \
      --out "$object_audit/gates_strict.json" > "$object_audit/gates_strict.log" 2>&1 || true
    "$python" sceneproof_local_settle_component_gate_fix84.py \
      --probe "$probe" --physical "$object_audit/physical.json" \
      --physical-objects "$object_audit/physical_objects.csv" \
      --gt "$object_audit/gt_8000.json" \
      --incumbent-version "$baseline" --candidate-version "$candidate" \
      --allow-support-proxy-exemption \
      --out "$object_audit/gates_relaxed.json" > "$object_audit/gates_relaxed.log" 2>&1 || true
    local strict relaxed accepted
    if ! test -s "$object_audit/gates_strict.json" || ! test -s "$object_audit/gates_relaxed.json"; then
      printf '%s\t%s\t%s\tfalse\tfalse\tfalse\t%s\t%s\t%s\t%s\n' \
        "$scene" "$object_id" "$profile" "$elapsed_total" \
        "$(readlink -f "$probe")" "$candidate" "$(readlink -f "$object_audit/gates_strict.log")" >> "$output"
      continue
    fi
    strict="$($python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1])).get("passed"))).lower())' "$object_audit/gates_strict.json")"
    relaxed="$($python -c 'import json,sys; print(str(bool(json.load(open(sys.argv[1])).get("passed"))).lower())' "$object_audit/gates_relaxed.json")"
    if test "${SCENEPROOF_ACCEPT_POLICY:-relaxed}" = "strict"; then
      accepted="$strict"
    else
      accepted="$relaxed"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$scene" "$object_id" "$profile" "$strict" "$relaxed" "$accepted" "$elapsed_total" \
      "$(readlink -f "$probe")" "$candidate" "$(readlink -f "$object_audit/gates_relaxed.json")" >> "$output"
  done < "$input"
}

target_count="$(wc -l < "$target_manifest")"
echo "RIGID_ONLY_TARGETS=$target_count BASELINE=$baseline DURATION=${duration}s WORLD=10/10 ACCEPT_POLICY=${SCENEPROOF_ACCEPT_POLICY:-relaxed}"
run_worker 0 "$list0" > "$log_root/gpu0.log" 2>&1 & p0=$!
sleep 3
run_worker 1 "$list1" > "$log_root/gpu1.log" 2>&1 & p1=$!
echo "RIGID_GPU0_PID=$p0"
echo "RIGID_GPU1_PID=$p1"
status=0
wait "$p0" || status=1
wait "$p1" || status=1
printf 'scene\tobject_id\tprofile\tstrict_passed\trelaxed_passed\taccepted\telapsed_seconds\tprobe\tcandidate_version\tgates\n' > "$selected_tsv"
for worker_output in "$audit/worker0.tsv" "$audit/worker1.tsv"; do
  if ! test -s "$worker_output"; then
    echo "FAIL missing worker output: $worker_output" >&2
    status=1
  fi
done
if test "$status" -ne 0; then
  echo "RIGID_ONLY_EVAL_STOP reason=worker_failure" >&2
  exit "$status"
fi
sort "$audit/worker0.tsv" "$audit/worker1.tsv" >> "$selected_tsv"
test "$(($(wc -l < "$selected_tsv") - 1))" -eq "$target_count" || status=1

if test "${SCENEPROOF_LOCAL_ONLY:-0}" = "1"; then
  echo "RIGID_ONLY_LOCAL_FINISHED status=$status targets=$target_count"
  echo "PER_OBJECT_TRIALS=$(readlink -f "$selected_tsv")"
  exit "$status"
fi

"$python" sceneproof_rigid_settle_adaptive_eval_fix84.py compose \
  --saved-results "$root" --manifest "$paper_manifest" --selected "$selected_tsv" \
  --baseline-version "$baseline" --target-version "$target" \
  --target-manifest "$audit/paper30_manifest.txt" --certificate "$certificate" || status=1

"$python" eval_physical_realizability.py \
  --saved-results "$root" --scenes "$paper_manifest" \
  --versions "$baseline,$target" --geometry-version "$source" \
  --baseline-version "$baseline" --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.txt" > "$audit/physical.log" 2>&1 || status=1
"$python" eval_gt_metrics.py \
  --saved-results "$root" --scenes "$paper_manifest" \
  --versions "$baseline,$target" --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" --manifest-out "$audit/gt_manifest_8000.json" \
  > "$audit/gt.log" 2>&1 || status=1
"$python" sceneproof_rigid_settle_adaptive_eval_fix84.py summarize \
  --selected "$selected_tsv" --physical "$audit/physical.json" \
  --gt "$audit/gt_8000.json" --certificate "$certificate" \
  --baseline-version "$baseline" --target-version "$target" \
  --out "$audit/final_eval.json" --report "$audit/final_eval.txt" || status=1

for render_version in "$baseline" "$target"; do
  env -u IMAGINARIUM_S4_RENDER_ONLY_CAMERA_TARGET \
    SCENEPROOF_MANIFEST="$scene_manifest" \
    SCENEPROOF_CERTIFIED_VERSION="$render_version" \
    SCENEPROOF_RENDER_LOG_ROOT="$log_root/render_$render_version" \
    SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
    bash scripts/render_sceneproof_certified_paper30.sh \
    > "$log_root/render_${render_version}.log" 2>&1 || status=1
done

collection="$root/sceneproof_${target}_comparison"
archive="$HOME/sceneproof_${target}_comparison.tar.gz"
mkdir -p "$collection"
while IFS=$'\t' read -r scene object_id; do
  mkdir -p "$collection/$scene"
  cp "$root/${scene}_${baseline}_result/S4_layout_refinement/${scene}_${baseline}_render_simu.png" \
    "$collection/$scene/00_fix61_baseline.png"
  cp "$root/${scene}_${target}_result/S4_layout_refinement/${scene}_${target}_render_simu.png" \
    "$collection/$scene/01_rigid_settle_final.png"
done < "$target_manifest"
cp "$audit/final_eval.txt" "$collection/final_eval.txt"
tar -czf "$archive" -C "$root" "$(basename "$collection")"

echo "RIGID_ONLY_EVAL_FINISHED status=$status"
echo "FINAL_EVAL=$(readlink -f "$audit/final_eval.json")"
echo "FINAL_REPORT=$(readlink -f "$audit/final_eval.txt")"
echo "RENDER_ARCHIVE=$(readlink -f "$archive")"
exit "$status"
