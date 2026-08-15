#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="a10_reusable_results/paper30"
manifest="${SCENEPROOF_FIX43_MANIFEST:-/tmp/sceneproof_fix43_inloop_smoke1_fix55.txt}"
legacy="${SCENEPROOF_LEGACY_VERSION:-v4_legacy_sa5000_bench}"
control="${SCENEPROOF_FIX43_CONTROL_VERSION:-v5_sceneproof_fix43_smooth_smoke1_fix53}"
candidate="${SCENEPROOF_FIX43_GUARDED_VERSION:-v5_sceneproof_fix43_inloop_guarded_smoke1_fix55}"
target="${SCENEPROOF_FIX43_CERTIFIED_VERSION:-v5_sceneproof_fix43_inloop_certified_smoke1_fix55}"
audit="$root/sceneba_audit/$target"
render_log="logs/${target}_locked_camera_render"
render_dir="$root/sceneproof_fix43_inloop_smoke1_renders_fix55"
render_archive="$HOME/sceneproof_fix43_inloop_smoke1_renders_fix55.tar.gz"

printf '%s\n' bedroom_01 > "$manifest"
mkdir -p "$audit"

test -d "$root/bedroom_01_${control}_result" || {
  echo "Missing reusable Fix53 smooth result: $root/bedroom_01_${control}_result" >&2
  exit 2
}

echo "===== FIX43 TRUE IN-LOOP GUARDED START $(date) ====="
env \
  IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
  IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
  IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
  IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
  IMAGINARIUM_S4_TARGET_VERSION="$candidate" \
  IMAGINARIUM_S4_ENGINE=layoutvlm \
  IMAGINARIUM_LAYOUTVLM_STAGE=full \
  IMAGINARIUM_LAYOUTVLM_SOLVER=v5_scenelm \
  IMAGINARIUM_LAYOUTVLM_ITERATIONS=2 \
  IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER=0 \
  IMAGINARIUM_SCENEPROOF_PROGRAM_IR=1 \
  IMAGINARIUM_SCENEPROOF_REQUIRE_FACTOR_PARITY=1 \
  IMAGINARIUM_SCENEPROOF_REQUIRE_BINDING_AUDIT=1 \
  IMAGINARIUM_SCENEPROOF_SHADOW_JACOBIAN_OWNERSHIP=1 \
  IMAGINARIUM_SCENEPROOF_STABLE_LINEARIZATIONS=2 \
  IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR=1 \
  IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=1 \
  IMAGINARIUM_SCENEPROOF_WARM_START_ANCHORED_PLANE_TRANSLATION=1 \
  IMAGINARIUM_SCENEPROOF_PLANE_ANCHOR_NORMAL_LIMIT_M=0.02 \
  IMAGINARIUM_SCENEPROOF_PLANE_PROXY_ABSTAIN_GAP_M=0.02 \
  IMAGINARIUM_SCENEPROOF_PLANE_ATTACH_REQUIRES_WITNESS=1 \
  IMAGINARIUM_SCENEPROOF_MATERIALIZED_WARM_START=1 \
  IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_TANGENT_PROJECTION=1 \
  IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_MAX_SHIFT_M=0.35 \
  IMAGINARIUM_SCENEPROOF_PLANE_COMPONENT_IMAGE_GAUGE=0 \
  IMAGINARIUM_SCENEPROOF_MESH_VISIBILITY_AUDIT=0 \
  IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB=0 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
  IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/$candidate" \
  bash scripts/run_paper30_v4_s4_only_dual_gpu.sh

echo "===== FIX43 IN-LOOP CERTIFICATE START $(date) ====="
"$PY" sceneproof_postsim_component_certifier.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --geometry-version v4_deepsearch \
  --incumbent-version "$control" \
  --candidate-version "$candidate" \
  --target-version "$target" \
  --margin 0.005 \
  --out "$audit/certificate.json" \
  --runtime-jsonl "$audit/certificate_runtime.jsonl"

for version in "$candidate" "$target"; do
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_CERTIFIED_VERSION="$version" \
  SCENEPROOF_RENDER_LOG_ROOT="logs/${version}_locked_camera_render" \
  SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    bash scripts/render_sceneproof_certified_paper30.sh
done

SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_LEGACY_VERSION="$legacy" \
SCENEPROOF_INCUMBENT_VERSION="$control" \
SCENEPROOF_CANDIDATE_VERSION="$candidate" \
SCENEPROOF_CERTIFIED_VERSION="$target" \
SCENEPROOF_INCLUDE_CERTIFICATE_RUNTIME=1 \
SCENEPROOF_REUSE_CERTIFICATE=1 \
SCENEPROOF_ALLOW_SAFE_ABSTAIN=1 \
SCENEPROOF_RENDER_LOG_ROOT="$render_log" \
  bash scripts/eval_sceneproof_postsim_component_certificate_fix21.sh

"$PY" sceneproof_collect_paper30_renders.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --legacy-version "$legacy" \
  --control-version "$control" \
  --candidate-version "$candidate" \
  --certified-version "$target" \
  --out-dir "$render_dir" \
  --archive "$render_archive"

echo "FIX43_INLOOP_CORRECTION_FINISHED target=$target"
echo "FINAL_GATES=$HOME/Lumenarium/$audit/final_gates.json"
echo "RENDER_ARCHIVE=$render_archive"
