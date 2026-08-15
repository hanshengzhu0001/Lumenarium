#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="a10_reusable_results/paper30"
manifest="${SCENEPROOF_MANIFEST:-$root/manifest.txt}"
legacy="${SCENEPROOF_LEGACY_VERSION:-v4_legacy_sa5000_bench}"
legacy_log="${SCENEPROOF_LEGACY_LOG_ROOT:-logs/paper30_s4_benchmark/legacy_sa5000}"
control="${SCENEPROOF_INCUMBENT_VERSION:-v5_sceneproof_smooth_control_paper30_fix25}"
candidate="${SCENEPROOF_CANDIDATE_VERSION:-v5_sceneproof_guarded_hybrid_paper30_fix25}"
target="${SCENEPROOF_CERTIFIED_VERSION:-v5_sceneproof_postsim_component_certified_paper30_fix25}"
audit="$root/sceneba_audit/$target"
render_log="${SCENEPROOF_RENDER_LOG_ROOT:-logs/${target}_locked_camera_render}"
render_dir="${SCENEPROOF_RENDER_COLLECTION_DIR:-$root/sceneproof_paper30_comparison_renders_fix25}"
render_archive="${SCENEPROOF_RENDER_ARCHIVE:-$HOME/sceneproof_paper30_comparison_renders_fix25.tar.gz}"

test -s "$manifest" || { echo "Missing Paper30 manifest: $manifest" >&2; exit 2; }
test "$(grep -cve '^[[:space:]]*$' "$manifest")" -eq 30 || {
  echo "Paper30 manifest must contain exactly 30 scenes: $manifest" >&2
  exit 2
}
mkdir -p "$audit"

echo "===== PAPER30 CONTROL START $(date) ====="
env \
  IMAGINARIUM_PAPER30_MANIFEST="$manifest" \
  IMAGINARIUM_S4_SOURCE_VERSION=v4_deepsearch \
  IMAGINARIUM_S4_SOURCE_STAGE=S3_pose_inference \
  IMAGINARIUM_S4_SOURCE_PATTERN='*_placement_info.json' \
  IMAGINARIUM_S4_TARGET_VERSION="$control" \
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
  IMAGINARIUM_SCENEPROOF_FULL_SO3_GUARDED_SCHUR=0 \
  IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=0 \
  IMAGINARIUM_SCENELM_KINEMATIC_BACKSUB=0 \
  IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  IMAGINARIUM_S4_SCENE_TIMEOUT=3600 \
  IMAGINARIUM_S4_WORKER_LOG_ROOT="logs/$control" \
  bash scripts/run_paper30_v4_s4_only_dual_gpu.sh

echo "===== PAPER30 GUARDED CANDIDATE START $(date) ====="
SCENEPROOF_SCHUR_MANIFEST="$manifest" \
SCENEPROOF_SCHUR_VERSION="$candidate" \
IMAGINARIUM_SCENEPROOF_IN_LOOP_GUARDED_SCHUR=1 \
IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  bash scripts/run_sceneproof_full_so3_guarded_schur_smoke5.sh

echo "===== PAPER30 COMPONENT CERTIFICATE START $(date) ====="
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

echo "===== PAPER30 LOCKED-CAMERA FINAL RENDER START $(date) ====="
SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_CERTIFIED_VERSION="$target" \
SCENEPROOF_RENDER_LOG_ROOT="$render_log" \
SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  bash scripts/render_sceneproof_certified_paper30.sh

echo "===== PAPER30 METRICS + END-TO-END TIMING START $(date) ====="
SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_LEGACY_VERSION="$legacy" \
SCENEPROOF_LEGACY_LOG_ROOT="$legacy_log" \
SCENEPROOF_INCUMBENT_VERSION="$control" \
SCENEPROOF_CANDIDATE_VERSION="$candidate" \
SCENEPROOF_CERTIFIED_VERSION="$target" \
SCENEPROOF_INCLUDE_CERTIFICATE_RUNTIME=1 \
SCENEPROOF_REUSE_CERTIFICATE=1 \
SCENEPROOF_RENDER_LOG_ROOT="$render_log" \
  bash scripts/eval_sceneproof_postsim_component_certificate_fix21.sh

"$PY" sceneba_paired_audit.py \
  --gt-metrics "$audit/gt_8000.json" \
  --physical-metrics "$audit/physical.json" \
  --baseline "$control" \
  --candidate "$target" \
  --samples 10000 \
  --rotation-margin -0.01 \
  --translation-margin -0.005 \
  --out "$audit/paired_bootstrap_10000.json"

echo "===== PAPER30 RENDER COLLECTION START $(date) ====="
"$PY" sceneproof_collect_paper30_renders.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --legacy-version "$legacy" \
  --control-version "$control" \
  --candidate-version "$candidate" \
  --certified-version "$target" \
  --out-dir "$render_dir" \
  --archive "$render_archive"

echo "PAPER30_FINISHED target=$target"
echo "FINAL_GATES=$HOME/Lumenarium/$audit/final_gates.json"
echo "PAIRED_BOOTSTRAP=$HOME/Lumenarium/$audit/paired_bootstrap_10000.json"
echo "RENDER_COLLECTION=$HOME/Lumenarium/$render_dir"
echo "RENDER_ARCHIVE=$render_archive"
