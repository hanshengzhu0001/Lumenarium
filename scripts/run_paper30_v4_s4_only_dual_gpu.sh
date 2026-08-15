#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

manifest="${IMAGINARIUM_PAPER30_MANIFEST:-a10_reusable_results/paper30/manifest.txt}"
results_root="${IMAGINARIUM_PAPER30_RESULTS_ROOT:-a10_reusable_results/paper30}"
blender="${IMAGINARIUM_BLENDER:-$HOME/Lumenarium/third_party/blender-4.3.2-linux-x64/blender}"
scene_timeout="${IMAGINARIUM_S4_SCENE_TIMEOUT:-3600}"
gpu_free_floor_mb="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-8000}"
iterations="${IMAGINARIUM_LAYOUTVLM_ITERATIONS:-400}"
layoutvlm_stage="${IMAGINARIUM_LAYOUTVLM_STAGE:-full}"
s4_engine="${IMAGINARIUM_S4_ENGINE:-layoutvlm}"
target_version="${IMAGINARIUM_S4_TARGET_VERSION:-v4}"
max_contact_gap="${IMAGINARIUM_LAYOUTVLM_MAX_CONTACT_GAP:-0.5}"
max_containment_error="${IMAGINARIUM_LAYOUTVLM_MAX_CONTAINMENT_ERROR:-0.5}"
depth_weight="${IMAGINARIUM_LAYOUTVLM_DEPTH_WEIGHT:-1.0}"
depth_center_weight="${IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_WEIGHT:-1.0}"
depth_size_weight="${IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_WEIGHT:-0.25}"
depth_metric_weight="${IMAGINARIUM_LAYOUTVLM_DEPTH_METRIC_WEIGHT:-1.0}"
depth_freeze_yaw="${IMAGINARIUM_LAYOUTVLM_DEPTH_FREEZE_YAW:-0}"
depth_min_pixels="${IMAGINARIUM_LAYOUTVLM_DEPTH_MIN_PIXELS:-800}"
depth_trust_weight="${IMAGINARIUM_LAYOUTVLM_DEPTH_TRUST_WEIGHT:-1.0}"
depth_center_margin_px="${IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_MARGIN_PX:-2.0}"
depth_size_margin_log="${IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_MARGIN_LOG:-0.02}"
depth_relative_margin_log="${IMAGINARIUM_LAYOUTVLM_DEPTH_RELATIVE_MARGIN_LOG:-0.01}"
active_set_router="${IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER:-0}"
router_checkpoints="${IMAGINARIUM_LAYOUTVLM_ROUTER_CHECKPOINTS:-30,100}"
router_high_degree="${IMAGINARIUM_LAYOUTVLM_ROUTER_HIGH_DEGREE:-6}"
router_wake_multiplier="${IMAGINARIUM_LAYOUTVLM_ROUTER_WAKE_MULTIPLIER:-1.5}"
layoutvlm_solver="${IMAGINARIUM_LAYOUTVLM_SOLVER:-adam}"
scenelm_initial_damping="${IMAGINARIUM_SCENELM_INITIAL_DAMPING:-0.01}"
scenelm_pcg_iterations="${IMAGINARIUM_SCENELM_PCG_ITERATIONS:-12}"
scenelm_pcg_tolerance="${IMAGINARIUM_SCENELM_PCG_TOLERANCE:-0.001}"
scenelm_acceptance_threshold="${IMAGINARIUM_SCENELM_ACCEPTANCE_THRESHOLD:-0.1}"
scenelm_gradient_tolerance="${IMAGINARIUM_SCENELM_GRADIENT_TOLERANCE:-0.00001}"
scenelm_relative_energy_tolerance="${IMAGINARIUM_SCENELM_RELATIVE_ENERGY_TOLERANCE:-0.0001}"
scenelm_patience="${IMAGINARIUM_SCENELM_PATIENCE:-3}"
scenelm_max_translation_step="${IMAGINARIUM_SCENELM_MAX_TRANSLATION_STEP:-0.2}"
scenelm_max_yaw_step_deg="${IMAGINARIUM_SCENELM_MAX_YAW_STEP_DEG:-15}"
scenelm_max_relation_releases="${IMAGINARIUM_SCENELM_MAX_RELATION_RELEASES:-1}"
scenelm_collision_witness_weight="${IMAGINARIUM_SCENELM_COLLISION_WITNESS_WEIGHT:-25}"
sceneba_discrete_repair="${IMAGINARIUM_SCENEBA_DISCRETE_REPAIR:-0}"
sceneba_repair_yaws="${IMAGINARIUM_SCENEBA_REPAIR_YAWS:-0,90,180,270}"
sceneba_repair_max_translation="${IMAGINARIUM_SCENEBA_REPAIR_MAX_TRANSLATION:-0.5}"
sceneba_repair_min_relative_gain="${IMAGINARIUM_SCENEBA_REPAIR_MIN_RELATIVE_GAIN:-0.08}"
sceneba_repair_min_absolute_gain="${IMAGINARIUM_SCENEBA_REPAIR_MIN_ABSOLUTE_GAIN:-0.001}"
sceneba_repair_min_margin="${IMAGINARIUM_SCENEBA_REPAIR_MIN_MARGIN:-0.0002}"
sceneba_asset_center_candidates="${IMAGINARIUM_SCENEBA_ASSET_CENTER_CANDIDATES:-0}"
sceneba_asset_center_scales="${IMAGINARIUM_SCENEBA_ASSET_CENTER_SCALES:-0.5,1.0,1.5}"
sceneba_support_surface_candidates="${IMAGINARIUM_SCENEBA_SUPPORT_SURFACE_CANDIDATES:-0}"
worker_log_root="${IMAGINARIUM_S4_WORKER_LOG_ROOT:-logs/paper30_v4_s4_only}"
gpu0_id="${IMAGINARIUM_GPU0_ID:-0}"
gpu1_id="${IMAGINARIUM_GPU1_ID:-1}"

# Every S4 invocation must use the original S3 placement to initialize assets,
# ground, and camera exactly once. Depth-aware S4 additionally loads the frozen
# v4 full-400 S4 JSON as a pose-only reference after scene initialization.
source_version="${IMAGINARIUM_S4_SOURCE_VERSION:-v4_deepsearch}"
source_stage="${IMAGINARIUM_S4_SOURCE_STAGE:-S3_pose_inference}"
source_pattern="${IMAGINARIUM_S4_SOURCE_PATTERN:-*_placement_info.json}"
case "$s4_engine" in
    legacy|layoutvlm) ;;
    *) echo "Unsupported IMAGINARIUM_S4_ENGINE=$s4_engine (expected legacy or layoutvlm)" >&2; exit 2 ;;
esac
if test "$s4_engine" = "layoutvlm" && test "$layoutvlm_stage" = "depth"; then
    reference_version="${IMAGINARIUM_S4_REFERENCE_VERSION:-v4}"
    reference_stage="${IMAGINARIUM_S4_REFERENCE_STAGE:-S4_layout_refinement}"
    reference_pattern="${IMAGINARIUM_S4_REFERENCE_PATTERN:-*_placement_info_s4.json}"
else
    reference_version=""
    reference_stage=""
    reference_pattern=""
fi

test -f "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 2; }
test -x "$blender" || { echo "Missing Blender: $blender" >&2; exit 2; }
mkdir -p "$results_root" "$worker_log_root"

list0="/tmp/lumenarium_paper30_v4_s4_gpu0.$$.txt"
list1="/tmp/lumenarium_paper30_v4_s4_gpu1.$$.txt"
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

source_pose_for_scene() {
    local scene="$1"
    local source_dir="${results_root}/${scene}_${source_version}_result/${source_stage}"
    find "$source_dir" -maxdepth 1 -type f -name "$source_pattern" \
        -print -quit 2>/dev/null
}

reference_pose_for_scene() {
    local scene="$1"
    local reference_dir
    test "$s4_engine" = "layoutvlm" || return 0
    test "$layoutvlm_stage" = "depth" || return 0
    reference_dir="${results_root}/${scene}_${reference_version}_result/${reference_stage}"
    find "$reference_dir" -maxdepth 1 -type f -name "$reference_pattern" \
        -print -quit 2>/dev/null
}

prepare_reused_result_tree() {
    local scene="$1"
    local source_result="${results_root}/${scene}_${source_version}_result"
    local target_result="${results_root}/${scene}_${target_version}_result"
    local source_s3="$2"
    local stage source_stage

    mkdir -p "$target_result/S3_pose_inference" \
        "$target_result/S4_layout_refinement"
    for stage in \
        S0_geometry_pred_results \
        S1_scene_parsing_results \
        S2_3d_retrieval_results
    do
        source_stage="$source_result/$stage"
        if test -d "$source_stage" && ! test -e "$target_result/$stage"; then
            ln -s "$(realpath "$source_stage")" "$target_result/$stage"
        fi
    done
    ln -sfn \
        "$(realpath "$source_s3")" \
        "$target_result/S3_pose_inference/${scene}_${target_version}_placement_info.json"
}

run_worker() {
    local gpu="$1"
    local scene_list="$2"
    local scene source_s3 reference_s4 target_result target_s3 target_s4 scene_log
    local free rc worker_status=0 start_ns end_ns elapsed status_text runtime_log
    local -a blender_args

    runtime_log="$worker_log_root/runtime_gpu${gpu}.jsonl"
    # Keep successful timings across resumptions. parse_runtime_jsonl uses the
    # last successful record per scene, so retry records can safely append.
    touch "$runtime_log"

    while IFS= read -r scene || test -n "$scene"; do
        scene="${scene%$'\r'}"
        test -n "$scene" || continue
        target_result="${results_root}/${scene}_${target_version}_result"
        target_s4="$target_result/S4_layout_refinement/${scene}_${target_version}_placement_info_s4.json"
        scene_log="$worker_log_root/${scene}_gpu${gpu}.log"

        if test -s "$target_s4"; then
            echo "CACHED scene=$scene gpu=$gpu"
            continue
        fi

        source_s3="$(source_pose_for_scene "$scene")"
        if ! test -s "$source_s3"; then
            echo "FAIL scene=$scene gpu=$gpu reason=missing_source_pose source_version=$source_version source_stage=$source_stage"
            worker_status=1
            continue
        fi
        reference_s4=""
        if test "$s4_engine" = "layoutvlm" && test "$layoutvlm_stage" = "depth"; then
            reference_s4="$(reference_pose_for_scene "$scene")"
            if ! test -s "$reference_s4"; then
                echo "FAIL scene=$scene gpu=$gpu reason=missing_reference_pose reference_version=$reference_version reference_stage=$reference_stage"
                worker_status=1
                continue
            fi
        fi
        prepare_reused_result_tree "$scene" "$source_s3"
        target_s3="$target_result/S3_pose_inference/${scene}_${target_version}_placement_info.json"

        while true; do
            free="$(
                nvidia-smi -i "$gpu" \
                    --query-gpu=memory.free \
                    --format=csv,noheader,nounits 2>/dev/null |
                    head -1 | tr -d ' '
            )"
            if test -n "$free" && test "$free" -ge "$gpu_free_floor_mb"; then
                break
            fi
            echo "WAIT_GPU scene=$scene gpu=$gpu free=${free:-unknown}MiB $(date)"
            sleep 60
        done

        blender_args=(
            --background
            --python modules/S4_blender_layout_and_corr.py
            --
            --obj_placement_info_json_path "$target_s3"
            --output_folder "$target_result/S4_layout_refinement"
        )
        if test "$s4_engine" = "layoutvlm"; then
            blender_args+=(--use_layoutvlm --layoutvlm_stage "$layoutvlm_stage")
        fi
        echo "START scene=$scene gpu=$gpu free=${free}MiB engine=$s4_engine stage=$layoutvlm_stage iterations=$iterations $(date)"
        start_ns="$(date +%s%N)"
        timeout "$scene_timeout" env \
            CUDA_VISIBLE_DEVICES="$gpu" \
            IMAGINARIUM_LAYOUTVLM_ITERATIONS="$iterations" \
            IMAGINARIUM_LAYOUTVLM_MAX_CONTACT_GAP="$max_contact_gap" \
            IMAGINARIUM_LAYOUTVLM_MAX_CONTAINMENT_ERROR="$max_containment_error" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_WEIGHT="$depth_weight" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_WEIGHT="$depth_center_weight" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_WEIGHT="$depth_size_weight" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_METRIC_WEIGHT="$depth_metric_weight" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_FREEZE_YAW="$depth_freeze_yaw" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_MIN_PIXELS="$depth_min_pixels" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_TRUST_WEIGHT="$depth_trust_weight" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_CENTER_MARGIN_PX="$depth_center_margin_px" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_SIZE_MARGIN_LOG="$depth_size_margin_log" \
            IMAGINARIUM_LAYOUTVLM_DEPTH_RELATIVE_MARGIN_LOG="$depth_relative_margin_log" \
            IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER="$active_set_router" \
            IMAGINARIUM_LAYOUTVLM_ROUTER_CHECKPOINTS="$router_checkpoints" \
            IMAGINARIUM_LAYOUTVLM_ROUTER_HIGH_DEGREE="$router_high_degree" \
            IMAGINARIUM_LAYOUTVLM_ROUTER_WAKE_MULTIPLIER="$router_wake_multiplier" \
            IMAGINARIUM_LAYOUTVLM_SOLVER="$layoutvlm_solver" \
            IMAGINARIUM_SCENELM_INITIAL_DAMPING="$scenelm_initial_damping" \
            IMAGINARIUM_SCENELM_PCG_ITERATIONS="$scenelm_pcg_iterations" \
            IMAGINARIUM_SCENELM_PCG_TOLERANCE="$scenelm_pcg_tolerance" \
            IMAGINARIUM_SCENELM_ACCEPTANCE_THRESHOLD="$scenelm_acceptance_threshold" \
            IMAGINARIUM_SCENELM_GRADIENT_TOLERANCE="$scenelm_gradient_tolerance" \
            IMAGINARIUM_SCENELM_RELATIVE_ENERGY_TOLERANCE="$scenelm_relative_energy_tolerance" \
            IMAGINARIUM_SCENELM_PATIENCE="$scenelm_patience" \
            IMAGINARIUM_SCENELM_MAX_TRANSLATION_STEP="$scenelm_max_translation_step" \
            IMAGINARIUM_SCENELM_MAX_YAW_STEP_DEG="$scenelm_max_yaw_step_deg" \
            IMAGINARIUM_SCENELM_MAX_RELATION_RELEASES="$scenelm_max_relation_releases" \
            IMAGINARIUM_SCENELM_COLLISION_WITNESS_WEIGHT="$scenelm_collision_witness_weight" \
            IMAGINARIUM_SCENEBA_DISCRETE_REPAIR="$sceneba_discrete_repair" \
            IMAGINARIUM_SCENEBA_REPAIR_YAWS="$sceneba_repair_yaws" \
            IMAGINARIUM_SCENEBA_REPAIR_MAX_TRANSLATION="$sceneba_repair_max_translation" \
            IMAGINARIUM_SCENEBA_REPAIR_MIN_RELATIVE_GAIN="$sceneba_repair_min_relative_gain" \
            IMAGINARIUM_SCENEBA_REPAIR_MIN_ABSOLUTE_GAIN="$sceneba_repair_min_absolute_gain" \
            IMAGINARIUM_SCENEBA_REPAIR_MIN_MARGIN="$sceneba_repair_min_margin" \
            IMAGINARIUM_SCENEBA_ASSET_CENTER_CANDIDATES="$sceneba_asset_center_candidates" \
            IMAGINARIUM_SCENEBA_ASSET_CENTER_SCALES="$sceneba_asset_center_scales" \
            IMAGINARIUM_SCENEBA_SUPPORT_SURFACE_CANDIDATES="$sceneba_support_surface_candidates" \
            IMAGINARIUM_LAYOUTVLM_REFERENCE_JSON="$reference_s4" \
            PYTHONUNBUFFERED=1 \
            LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
            "$blender" "${blender_args[@]}" \
            > "$scene_log" 2>&1 < /dev/null
        rc=$?
        end_ns="$(date +%s%N)"
        elapsed="$(
            awk -v start="$start_ns" -v end="$end_ns" \
                'BEGIN { printf "%.6f", (end - start) / 1000000000.0 }'
        )"

        if test -s "$target_s4"; then
            status_text="ok"
            echo "OK scene=$scene gpu=$gpu rc=$rc elapsed_seconds=$elapsed $(date)"
        else
            status_text="fail"
            echo "FAIL scene=$scene gpu=$gpu rc=$rc elapsed_seconds=$elapsed log=$scene_log $(date)"
            worker_status=1
        fi
        printf '{"scene":"%s","version":"%s","engine":"%s","stage":"%s","gpu":%s,"elapsed_seconds":%s,"status":"%s","return_code":%s}\n' \
            "$scene" "$target_version" "$s4_engine" "$layoutvlm_stage" \
            "$gpu" "$elapsed" "$status_text" "$rc" >> "$runtime_log"
        sleep 10
    done < "$scene_list"
    return "$worker_status"
}

source_ready=0
reference_ready=0
while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"
    test -n "$scene" || continue
    if test -s "$(source_pose_for_scene "$scene")"; then
        source_ready=$((source_ready + 1))
    else
        echo "PREFLIGHT_MISSING scene=$scene"
    fi
    if test "$s4_engine" != "layoutvlm" \
        || test "$layoutvlm_stage" != "depth" \
        || test -s "$(reference_pose_for_scene "$scene")"
    then
        reference_ready=$((reference_ready + 1))
    else
        echo "PREFLIGHT_REFERENCE_MISSING scene=$scene"
    fi
done < "$manifest"
preflight_expected="$(grep -cve '^[[:space:]]*$' "$manifest")"
echo "PREFLIGHT source_s3_ready=$source_ready/$preflight_expected"
echo "PREFLIGHT reference_s4_ready=$reference_ready/$preflight_expected"
if test "$source_ready" -ne "$preflight_expected" \
    || test "$reference_ready" -ne "$preflight_expected"
then
    echo "No reusable source poses were found: version=$source_version stage=$source_stage pattern=$source_pattern" >&2
    exit 2
fi
if test "${IMAGINARIUM_S4_PREFLIGHT_ONLY:-0}" = "1"; then
    exit 0
fi

run_worker "$gpu0_id" "$list0" > "$worker_log_root/gpu0.log" 2>&1 &
worker_pids+=("$!")
sleep 20
run_worker "$gpu1_id" "$list1" > "$worker_log_root/gpu1.log" 2>&1 &
worker_pids+=("$!")

echo "GPU0_WORKER_PID=${worker_pids[0]}"
echo "GPU1_WORKER_PID=${worker_pids[1]}"
echo "RESULTS_ROOT=$results_root"
echo "TARGET_VERSION=$target_version ENGINE=$s4_engine STAGE=$layoutvlm_stage SOLVER=$layoutvlm_solver"
echo "SOURCE_VERSION=$source_version SOURCE_STAGE=$source_stage SOURCE_PATTERN=$source_pattern"
echo "REFERENCE_VERSION=$reference_version REFERENCE_STAGE=$reference_stage REFERENCE_PATTERN=$reference_pattern"
echo "ITERATIONS=$iterations GPU_FREE_FLOOR_MB=$gpu_free_floor_mb"

status=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || status=1
done

expected="$(grep -cve '^[[:space:]]*$' "$manifest")"
completed=0
while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"
    test -n "$scene" || continue
    target_s4="${results_root}/${scene}_${target_version}_result/S4_layout_refinement/${scene}_${target_version}_placement_info_s4.json"
    if test -s "$target_s4"; then
        completed=$((completed + 1))
    fi
done < "$manifest"
if test "$completed" -ne "$expected"; then
    status=1
fi
echo "FINISHED completed=$completed/$expected status=$status"
exit "$status"
