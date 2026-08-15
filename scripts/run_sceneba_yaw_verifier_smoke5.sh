#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/Lumenarium"

PY="${SCENEBA_PYTHON:-${HOME}/.venvs/lumenarium-py311/bin/python}"
MANIFEST="${SCENEBA_YAW_MANIFEST:-/tmp/sceneba_repair_smoke5.txt}"
ROOT="a10_reusable_results/paper30"
SOURCE_VERSION="${SCENEBA_YAW_SOURCE_VERSION:-v4_deepsearch}"
REFERENCE_VERSION="${SCENEBA_YAW_REFERENCE_VERSION:-v4}"
TARGET_VERSION="${SCENEBA_YAW_TARGET_VERSION:-v4_yaw_verifier}"
RASTER_SIZE="${SCENEBA_YAW_RASTER_SIZE:-128}"
LOG_ROOT="${SCENEBA_YAW_LOG_ROOT:-logs/sceneba_yaw_verifier_smoke5}"

test -s "${MANIFEST}" || {
  echo "Missing yaw manifest: ${MANIFEST}" >&2
  exit 2
}
mkdir -p "${LOG_ROOT}"

completed=0
failed=0
while IFS= read -r scene || test -n "${scene}"; do
  test -n "${scene}" || continue
  bank="${ROOT}/${scene}_${SOURCE_VERSION}_result/S3_pose_inference/${scene}_sceneba_dino_correspondence_bank.json"
  target="${ROOT}/${scene}_${TARGET_VERSION}_result/S4_layout_refinement/${scene}_${TARGET_VERSION}_placement_info_s4.json"
  scene_log="${LOG_ROOT}/${scene}.log"
  if ! test -s "${bank}"; then
    echo "FAIL scene=${scene} missing_bank=${bank}"
    failed=$((failed + 1))
    continue
  fi
  if test -s "${target}"; then
    echo "CACHED scene=${scene}"
    completed=$((completed + 1))
    continue
  fi
  echo "START scene=${scene} raster=${RASTER_SIZE} $(date)"
  started="$(date +%s)"
  set +e
  env \
    LD_LIBRARY_PATH="${HOME}/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
    PYTHONUNBUFFERED=1 \
    "${PY}" -u sceneba_yaw_verifier.py \
      --saved-results "${ROOT}" \
      --scene "${scene}" \
      --source-version "${SOURCE_VERSION}" \
      --reference-version "${REFERENCE_VERSION}" \
      --target-version "${TARGET_VERSION}" \
      --raster-size "${RASTER_SIZE}" \
      > "${scene_log}" 2>&1
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
