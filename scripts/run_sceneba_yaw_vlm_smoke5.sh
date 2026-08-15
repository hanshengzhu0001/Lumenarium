#!/usr/bin/env bash
set -euo pipefail

cd "${SCENEBA_REPO_ROOT:-$HOME/Lumenarium}"

python_bin="${SCENEBA_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
manifest="${SCENEBA_YAW_VLM_MANIFEST:-/tmp/sceneba_yaw_vlm_smoke5.txt}"
config="${SCENEBA_YAW_VLM_CONFIG:-config/config_a10_paper30.yaml}"
dotenv="${SCENEBA_YAW_VLM_DOTENV:-.env}"
source_version="${SCENEBA_YAW_VLM_SOURCE_VERSION:-v4_deepsearch}"
reference_version="${SCENEBA_YAW_VLM_REFERENCE_VERSION:-v4}"
candidate_version="${SCENEBA_YAW_VLM_CANDIDATE_VERSION:-v4_yaw_dense_reproj_audit_v1}"
target_version="${SCENEBA_YAW_VLM_TARGET_VERSION:-v4_yaw_vlm_contact_v1}"
results_root="${SCENEBA_RESULTS_ROOT:-a10_reusable_results/paper30}"
log_root="${SCENEBA_YAW_VLM_LOG_ROOT:-logs/sceneba_yaw_vlm_contact_v1}"
minimum_confidence="${SCENEBA_YAW_VLM_MIN_CONFIDENCE:-0.80}"
symmetry_threshold="${SCENEBA_YAW_VLM_SYMMETRY_THRESHOLD:-0.02}"
minimum_mask_pixels="${SCENEBA_YAW_VLM_MIN_MASK_PIXELS:-8000}"

mkdir -p "$log_root"

completed=0
failed=0
while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"
    test -n "$scene" || continue
    output_dir="${results_root}/${scene}_${target_version}_result/S4_layout_refinement"
    if find "$output_dir" -maxdepth 1 -type f -name '*_placement_info_s4.json' -print -quit 2>/dev/null | grep -q .; then
        echo "CACHED scene=$scene"
        completed=$((completed + 1))
        continue
    fi
    log="${log_root}/${scene}.log"
    start=$(date +%s)
    echo "START scene=$scene $(date)"
    if env \
      IMAGINARIUM_GPT_LOCK_FILE="${IMAGINARIUM_GPT_LOCK_FILE:-/tmp/lumenarium_gemini.lock}" \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$python_bin" -u sceneba_yaw_vlm_verifier.py \
        --saved-results "$results_root" \
        --scene "$scene" \
        --config "$config" \
        --dotenv "$dotenv" \
        --source-version "$source_version" \
        --reference-version "$reference_version" \
        --candidate-version "$candidate_version" \
        --target-version "$target_version" \
        --minimum-confidence "$minimum_confidence" \
        --minimum-mask-pixels "$minimum_mask_pixels" \
        --symmetry-threshold "$symmetry_threshold" \
        > "$log" 2>&1
    then
        elapsed=$(( $(date +%s) - start ))
        echo "OK scene=$scene wall_seconds=$elapsed $(date)"
        completed=$((completed + 1))
    else
        status=$?
        echo "FAIL scene=$scene status=$status log=$log $(date)"
        failed=$((failed + 1))
    fi
done < "$manifest"

echo "FINISHED completed=$completed failed=$failed"
test "$failed" -eq 0
