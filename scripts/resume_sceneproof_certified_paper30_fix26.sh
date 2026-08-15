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

test -s "$audit/certificate.json" || {
  echo "Missing completed Paper30 certificate: $audit/certificate.json" >&2
  exit 2
}
test -s "$audit/certificate_runtime.jsonl" || {
  echo "Missing Paper30 certificate timing: $audit/certificate_runtime.jsonl" >&2
  exit 2
}

SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_CERTIFIED_VERSION="$target" \
SCENEPROOF_RENDER_LOG_ROOT="$render_log" \
SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  bash scripts/render_sceneproof_certified_paper30.sh

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

"$PY" sceneproof_collect_paper30_renders.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --legacy-version "$legacy" \
  --control-version "$control" \
  --candidate-version "$candidate" \
  --certified-version "$target" \
  --out-dir "$render_dir" \
  --archive "$render_archive"

echo "PAPER30_RESUME_FINISHED target=$target"
echo "FINAL_GATES=$HOME/Lumenarium/$audit/final_gates.json"
echo "RENDER_ARCHIVE=$render_archive"
