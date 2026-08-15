#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"

PY="${SCENEBA_PYTHON:-${HOME}/.venvs/lumenarium-py311/bin/python}"
MANIFEST="${SCENEBA_MANIFEST:-/tmp/sceneba_pose_bank_smoke5.txt}"
GPU_ID="${SCENEBA_GPU_ID:-1}"
GPU_FREE_FLOOR_MB="${SCENEBA_GPU_FREE_FLOOR_MB:-18000}"
SOURCE_VERSION="${SCENEBA_SOURCE_VERSION:-v4_deepsearch}"
TOP_K_ASSETS="${SCENEBA_TOP_K_ASSETS:-3}"
TOP_K_VIEWS="${SCENEBA_TOP_K_VIEWS:-3}"
YAW_OFFSETS="${SCENEBA_YAW_OFFSETS:-0,90,180,270}"
RESULTS_ROOT="${SCENEBA_RESULTS_ROOT:-a10_reusable_results/paper30}"
LOG_ROOT="${SCENEBA_LOG_ROOT:-logs/sceneba_pose_bank_smoke5}"
OVERWRITE="${SCENEBA_OVERWRITE:-0}"

if [[ ! -s "${MANIFEST}" ]]; then
  printf '%s\n' \
    bedroom_01 \
    livingroom_10 \
    casino_01 \
    official_01 \
    streelitter_01 > "${MANIFEST}"
fi

mkdir -p "${LOG_ROOT}"

wait_for_gpu() {
  local free
  while true; do
    free="$(
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | sed -n "$((GPU_ID + 1))p" | tr -d ' '
    )"
    if [[ "${free:-0}" -ge "${GPU_FREE_FLOOR_MB}" ]]; then
      echo "GPU_READY id=${GPU_ID} free=${free}MiB floor=${GPU_FREE_FLOOR_MB}MiB"
      return
    fi
    echo "GPU_WAIT id=${GPU_ID} free=${free:-unknown}MiB floor=${GPU_FREE_FLOOR_MB}MiB $(date)"
    sleep 30
  done
}

total=0
completed=0
failed=0
while IFS= read -r scene || [[ -n "${scene}" ]]; do
  [[ -z "${scene}" ]] && continue
  total=$((total + 1))
  out="${RESULTS_ROOT}/${scene}_${SOURCE_VERSION}_result/S3_pose_inference/${scene}_sceneba_pose_bank.json"
  log="${LOG_ROOT}/${scene}_gpu${GPU_ID}.log"
  if [[ -s "${out}" && "${OVERWRITE}" != "1" ]]; then
    echo "CACHED scene=${scene} path=${out}"
    completed=$((completed + 1))
    continue
  fi
  wait_for_gpu
  echo "START scene=${scene} gpu=${GPU_ID} $(date)"
  started="$(date +%s)"
  if env \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH="${IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH:-1}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
      PYTHONUNBUFFERED=1 \
      "${PY}" -u sceneba_build_pose_bank.py \
        --saved-results "${RESULTS_ROOT}" \
        --scene "${scene}" \
        --source-version "${SOURCE_VERSION}" \
        --top-k-assets "${TOP_K_ASSETS}" \
        --top-k-views "${TOP_K_VIEWS}" \
        --yaw-offsets "${YAW_OFFSETS}" \
        > "${log}" 2>&1
  then
    elapsed=$(( $(date +%s) - started ))
    echo "OK scene=${scene} wall_seconds=${elapsed} $(date)"
    completed=$((completed + 1))
  else
    status=$?
    elapsed=$(( $(date +%s) - started ))
    echo "FAIL scene=${scene} status=${status} wall_seconds=${elapsed} log=${log}"
    failed=$((failed + 1))
  fi
done < "${MANIFEST}"

echo "FINISHED completed=${completed}/${total} failed=${failed}"
[[ "${failed}" -eq 0 ]]

