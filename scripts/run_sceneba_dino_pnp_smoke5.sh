#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"

PY="${SCENEBA_PYTHON:-${HOME}/.venvs/lumenarium-py311/bin/python}"
MANIFEST="${SCENEBA_DINO_PNP_MANIFEST:-/tmp/sceneba_repair_smoke5.txt}"
GPU_ID="${SCENEBA_DINO_PNP_GPU_ID:-1}"
GPU_FREE_FLOOR_MB="${SCENEBA_DINO_PNP_GPU_FREE_FLOOR_MB:-16000}"
ROOT="a10_reusable_results/paper30"
LOG_ROOT="${SCENEBA_DINO_PNP_LOG_ROOT:-logs/sceneba_dino_pnp_smoke5}"
TARGET_VERSION="${SCENEBA_DINO_PNP_TARGET_VERSION:-v4_dino_pnp_oracle}"

test -s "${MANIFEST}" || {
  echo "Missing Smoke5 manifest: ${MANIFEST}" >&2
  exit 2
}
mkdir -p "${LOG_ROOT}"

wait_for_gpu() {
  local free
  while true; do
    free="$(
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | sed -n "$((GPU_ID + 1))p" | tr -d ' '
    )"
    if test -n "${free}" && test "${free}" -ge "${GPU_FREE_FLOOR_MB}"; then
      echo "GPU_READY id=${GPU_ID} free=${free}MiB floor=${GPU_FREE_FLOOR_MB}MiB"
      return
    fi
    echo "GPU_WAIT id=${GPU_ID} free=${free:-unknown}MiB floor=${GPU_FREE_FLOOR_MB}MiB"
    sleep 30
  done
}

completed=0
failed=0
while IFS= read -r scene || test -n "${scene}"; do
  test -n "${scene}" || continue
  source_s3="${ROOT}/${scene}_v4_deepsearch_result/S3_pose_inference"
  bank="${source_s3}/${scene}_sceneba_dino_correspondence_bank.json"
  target="${ROOT}/${scene}_${TARGET_VERSION}_result/S4_layout_refinement/${scene}_${TARGET_VERSION}_placement_info_s4.json"
  scene_log="${LOG_ROOT}/${scene}.log"
  if test -s "${bank}" && test -s "${target}"; then
    echo "CACHED scene=${scene}"
    completed=$((completed + 1))
    continue
  fi
  wait_for_gpu
  echo "START scene=${scene} gpu=${GPU_ID} $(date)"
  started="$(date +%s)"
  set +e
  {
    if ! test -s "${bank}"; then
      env \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH=1 \
        PYTHONUNBUFFERED=1 \
        LD_LIBRARY_PATH="${HOME}/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
        "${PY}" -u sceneba_build_pose_bank.py \
          --saved-results "${ROOT}" \
          --scene "${scene}" \
          --source-version v4_deepsearch \
          --top-k-assets 1 \
          --top-k-views 5 \
          --yaw-offsets 0 \
          --capture-dense-correspondences \
          --out "${bank}"
    fi
    env \
      LD_LIBRARY_PATH="${HOME}/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "${PY}" -u sceneba_dino_pnp_candidates.py \
        --saved-results "${ROOT}" \
        --scene "${scene}" \
        --source-version v4_deepsearch \
        --reference-version v4 \
        --target-version "${TARGET_VERSION}" \
        --minimum-matches 8 \
        --reprojection-threshold-px 8 \
        --max-shift-m 1.0
  } > "${scene_log}" 2>&1
  rc=$?
  set -e
  elapsed=$(( $(date +%s) - started ))
  if test "${rc}" -eq 0 && test -s "${target}"; then
    echo "OK scene=${scene} wall_seconds=${elapsed} $(date)"
    completed=$((completed + 1))
  else
    echo "FAIL scene=${scene} rc=${rc} log=${scene_log}"
    failed=$((failed + 1))
  fi
done < "${MANIFEST}"

echo "FINISHED completed=${completed} failed=${failed}"
test "${failed}" -eq 0
