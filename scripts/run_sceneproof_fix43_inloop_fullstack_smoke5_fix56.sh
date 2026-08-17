#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
source_version="${SCENEPROOF_SOURCE_VERSION:-v4_deepsearch}"
manifest="${SCENEPROOF_FIX43_MANIFEST:-/tmp/sceneproof_fix43_inloop_smoke5_fix56.txt}"
legacy="${SCENEPROOF_LEGACY_VERSION:-v4_legacy_sa5000_bench}"
control="${SCENEPROOF_FIX43_CONTROL_VERSION:-v5_sceneproof_fix43_smooth_smoke5_fix56}"
candidate="${SCENEPROOF_FIX43_GUARDED_VERSION:-v5_sceneproof_fix43_inloop_guarded_smoke5_fix56}"
target="${SCENEPROOF_FIX43_CERTIFIED_VERSION:-v5_sceneproof_fix43_inloop_certified_smoke5_fix56}"
audit="$root/sceneba_audit/$target"
render_log="logs/${target}_locked_camera_render"
render_dir="${SCENEPROOF_FIX43_RENDER_DIR:-$root/sceneproof_fix43_inloop_smoke5_renders_fix56}"
render_archive="${SCENEPROOF_FIX43_RENDER_ARCHIVE:-$HOME/sceneproof_fix43_inloop_smoke5_renders_fix56.tar.gz}"
source_manifest="${SCENEPROOF_FIX43_SOURCE_MANIFEST:-}"
expected_scenes="${SCENEPROOF_FIX43_EXPECTED_SCENES:-5}"
minimum_nonzero_scenes="${SCENEPROOF_FIX43_MINIMUM_NONZERO_SCENES:-3}"

if [[ -n "$source_manifest" ]]; then
  test -s "$source_manifest" || {
    echo "Missing source manifest: $source_manifest" >&2
    exit 2
  }
  cp "$source_manifest" "$manifest"
else
  printf '%s\n' \
    bedroom_01 livingroom_10 casino_01 official_01 streelitter_01 \
    > "$manifest"
fi
if [[ "$(wc -l < "$manifest")" -ne "$expected_scenes" ]]; then
  echo "Expected $expected_scenes scenes in $manifest" >&2
  exit 2
fi
mkdir -p "$audit"

run_branch() {
  local version="$1"
  local guarded="$2"
  local mesh_visibility=0
  if test "$guarded" = "1"; then
    mesh_visibility="${SCENEPROOF_FIX43_MESH_VISIBILITY_AUDIT:-0}"
  fi
  env \
    IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
    IMAGINARIUM_PAPER30_RESULTS_ROOT="$root" \
    IMAGINARIUM_S4_SOURCE_VERSION="$source_version" \
    IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
    IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
    IMAGINARIUM_S4_TARGET_VERSION="$version" \
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
    IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR="$guarded" \
    IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR="$guarded" \
    IMAGINARIUM_SCENEPROOF_WARM_START_ANCHORED_PLANE_TRANSLATION=1 \
    IMAGINARIUM_SCENEPROOF_PLANE_ANCHOR_NORMAL_LIMIT_M=0.02 \
    IMAGINARIUM_SCENEPROOF_PLANE_PROXY_ABSTAIN_GAP_M=0.02 \
    IMAGINARIUM_SCENEPROOF_PLANE_ATTACH_REQUIRES_WITNESS=1 \
    IMAGINARIUM_SCENEPROOF_MATERIALIZED_WARM_START=1 \
    IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_TANGENT_PROJECTION=1 \
    IMAGINARIUM_SCENEPROOF_PLANE_SIBLING_MAX_SHIFT_M=0.35 \
    IMAGINARIUM_SCENEPROOF_PLANE_COMPONENT_IMAGE_GAUGE=0 \
    IMAGINARIUM_SCENEPROOF_MESH_VISIBILITY_AUDIT="$mesh_visibility" \
    IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB=0 \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
    IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/$version" \
    bash scripts/run_paper30_v4_s4_only_dual_gpu.sh
}

echo "===== FIX43 SMOKE5 SMOOTH START $(date) ====="
run_branch "$control" 0
echo "===== FIX43 SMOKE5 TRUE IN-LOOP GUARDED START $(date) ====="
run_branch "$candidate" 1

# Historical result roots contain the imported-asset geometry snapshot under
# the source version.  A clean source-S3 run first materializes it under the
# smooth SceneLM branch.  Select that provenance once and use it consistently
# for certification and formal evaluation.
geometry_version="$source_version"
geometry_probe="$root/${manifest##*/}.geometry_probe"
if ! "$PY" - "$root" "$manifest" "$source_version" >/dev/null 2>"$geometry_probe" <<'PY'
import sys
from pathlib import Path
from eval_physical_realizability import find_geometry_snapshot
root, manifest, version = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
for scene in (line.strip() for line in manifest.read_text().splitlines()):
    if scene:
        find_geometry_snapshot(root, scene, version)
PY
then
  geometry_version="$control"
fi
rm -f "$geometry_probe"
echo "SCENEPROOF_GEOMETRY_PROVENANCE requested=$source_version selected=$geometry_version"

echo "===== FIX43 SMOKE5 COMPONENT CERTIFICATE START $(date) ====="
"$PY" sceneproof_postsim_component_certifier.py \
  --saved-results "$root" --scenes "$manifest" \
  --geometry-version "$geometry_version" \
  --incumbent-version "$control" --candidate-version "$candidate" \
  --target-version "$target" --margin 0.005 \
  --out "$audit/certificate.json" \
  --runtime-jsonl "$audit/certificate_runtime.jsonl"

if [[ "${SCENEPROOF_FIX43_SKIP_RENDER:-0}" != "1" ]]; then
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_RENDER_SOURCE_VERSION="$source_version" \
  SCENEPROOF_CERTIFIED_VERSION="$target" \
  SCENEPROOF_RENDER_LOG_ROOT="$render_log" \
  SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
    bash scripts/render_sceneproof_certified_paper30.sh
fi

if [[ "${SCENEPROOF_FIX43_SKIP_FORMAL_EVAL:-0}" != "1" ]]; then
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_GEOMETRY_VERSION="$geometry_version" \
  SCENEPROOF_LEGACY_VERSION="$legacy" \
  SCENEPROOF_INCUMBENT_VERSION="$control" \
  SCENEPROOF_CANDIDATE_VERSION="$candidate" \
  SCENEPROOF_CERTIFIED_VERSION="$target" \
  SCENEPROOF_INCLUDE_CERTIFICATE_RUNTIME=1 \
  SCENEPROOF_REUSE_CERTIFICATE=1 \
  SCENEPROOF_ALLOW_SAFE_ABSTAIN=0 \
  SCENEPROOF_RENDER_LOG_ROOT="$render_log" \
    bash scripts/eval_sceneproof_postsim_component_certificate_fix21.sh

  "$PY" sceneproof_fix43_inloop_smoke5_protocol.py \
    --certificate "$audit/certificate.json" \
    --physical "$audit/physical.json" \
    --final-gates "$audit/final_gates.json" \
    --incumbent-version "$control" --target-version "$target" \
    --expected-scenes "$expected_scenes" \
    --minimum-nonzero-scenes "$minimum_nonzero_scenes" --margin 0.005 \
    --out "$audit/protocol.json"
fi

if [[ "${SCENEPROOF_FIX43_SKIP_RENDER:-0}" != "1" ]]; then
  "$PY" sceneproof_collect_paper30_renders.py \
    --saved-results "$root" --scenes "$manifest" \
    --legacy-version "$legacy" --control-version "$control" \
    --candidate-version "$candidate" --certified-version "$target" \
    --out-dir "$render_dir" --archive "$render_archive"
fi

echo "FIX43_INLOOP_SMOKE5_FINISHED target=$target"
echo "FINAL_GATES=$HOME/Lumenarium/$audit/final_gates.json"
echo "PROTOCOL=$HOME/Lumenarium/$audit/protocol.json"
echo "RENDER_ARCHIVE=$render_archive"
