#!/usr/bin/env bash
set -euo pipefail

test "$#" -eq 2 || { echo "usage: $0 JOB_ID fast|medium" >&2; exit 2; }
job_id="$1"; profile="$2"
[[ "$job_id" =~ ^[0-9a-f]{32}$ ]] || { echo "invalid job id" >&2; exit 2; }
case "$profile" in fast|medium) ;; *) echo "profile must be fast or medium" >&2; exit 2;; esac

cd "$HOME/Lumenarium"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
jobs_root="${SCENEPROOF_API_JOB_ROOT:-$HOME/Lumenarium/api_jobs}"
root="$jobs_root/$job_id/results"
baseline="v5_sceneproof_collision_partial_commit_certified_api"
source="v4_deepsearch"
if test "$profile" = "medium"; then
  target="v5_sceneproof_medium_visible_proxy_api"
  repair=(--repair)
else
  target="v5_sceneproof_fast_visible_proxy_api"
  repair=()
fi
input="$root/${job_id}_${baseline}_result/S4_layout_refinement/${job_id}_${baseline}_placement_info_s4.json"
target_s4="$root/${job_id}_${target}_result/S4_layout_refinement"
audit="$root/sceneba_audit/$target"
manifest="$jobs_root/$job_id/visible_proxy_manifest.txt"
log_root="$jobs_root/$job_id/logs/visible_proxy_$profile"
test -s "$input" || { echo "missing Fix61 placement: $input" >&2; exit 3; }
mkdir -p "$target_s4" "$audit" "$log_root"
printf '%s\n' "$job_id" > "$manifest"

"$python" sceneproof_visible_support_proxy.py \
  --input "$input" \
  --output "$target_s4/${job_id}_${target}_placement_info_s4.json" \
  --certificate "$audit/visible_support_certificate.json" \
  "${repair[@]}" | tee "$log_root/certificate.log"

env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_RENDER_SOURCE_VERSION="$source" \
  SCENEPROOF_CERTIFIED_VERSION="$target" \
  SCENEPROOF_RENDER_LOG_ROOT="$log_root/render" \
  SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
  IMAGINARIUM_GPU0_ID="${IMAGINARIUM_GPU0_ID:-0}" \
  IMAGINARIUM_GPU1_ID="${IMAGINARIUM_GPU1_ID:-1}" \
  bash scripts/render_sceneproof_certified_paper30.sh \
  > "$log_root/render.log" 2>&1

echo "FIX135_CERTIFICATE=$audit/visible_support_certificate.json"
echo "FIX135_RENDER=$target_s4/${job_id}_${target}_render_simu.png"
