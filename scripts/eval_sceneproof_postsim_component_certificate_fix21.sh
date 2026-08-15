#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENEPROOF_RESULTS_ROOT:-a10_reusable_results/paper30}"
manifest="${SCENEPROOF_MANIFEST:-/tmp/sceneproof_guarded_hybrid_paired_smoke5.txt}"
legacy="${SCENEPROOF_LEGACY_VERSION:-v4_legacy_sa5000_bench}"
legacy_log="${SCENEPROOF_LEGACY_LOG_ROOT:-logs/paper30_s4_benchmark/legacy_sa5000}"
incumbent="${SCENEPROOF_INCUMBENT_VERSION:-v5_sceneproof_smooth_control_smoke5_fix15}"
candidate="${SCENEPROOF_CANDIDATE_VERSION:-v5_sceneproof_guarded_hybrid_smoke5_fix15}"
target="${SCENEPROOF_CERTIFIED_VERSION:-v5_sceneproof_postsim_component_certified_fix21}"
audit="$root/sceneba_audit/$target"
certificate_runtime="$audit/certificate_runtime.jsonl"
include_certificate_runtime="${SCENEPROOF_INCLUDE_CERTIFICATE_RUNTIME:-0}"
render_log="${SCENEPROOF_RENDER_LOG_ROOT:-}"
reuse_certificate="${SCENEPROOF_REUSE_CERTIFICATE:-0}"
mkdir -p "$audit"

certificate_args=(
  --saved-results "$root"
  --scenes "$manifest"
  --geometry-version "${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}"
  --incumbent-version "$incumbent"
  --candidate-version "$candidate"
  --target-version "$target"
  --margin 0.005
  --out "$audit/certificate.json"
)
if [[ "$include_certificate_runtime" == "1" ]]; then
  certificate_args+=(--runtime-jsonl "$certificate_runtime")
fi
if [[ "$reuse_certificate" == "1" ]]; then
  test -s "$audit/certificate.json" || {
    echo "Missing reusable certificate: $audit/certificate.json" >&2
    exit 2
  }
  if [[ "$include_certificate_runtime" == "1" ]]; then
    test -s "$certificate_runtime" || {
      echo "Missing reusable certificate timing: $certificate_runtime" >&2
      exit 2
    }
  fi
else
  "$PY" sceneproof_postsim_component_certifier.py "${certificate_args[@]}"
fi

runtime_args=(
  --runtime-log "$legacy=$legacy_log"
  --runtime-log "$incumbent=logs/$incumbent"
  --runtime-log "$candidate=logs/$candidate"
)
runtime_components="$incumbent+$candidate"
if [[ "$include_certificate_runtime" == "1" ]]; then
  certificate_component="${target}_certificate"
  runtime_args+=(--runtime-log "$certificate_component=$certificate_runtime")
  runtime_components+="+$certificate_component"
fi
if [[ -n "$render_log" ]]; then
  render_component="${target}_render"
  runtime_args+=(--runtime-log "$render_component=$render_log")
  runtime_components+="+$render_component"
fi

"$PY" eval_physical_realizability.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$legacy,$incumbent,$candidate,$target" \
  --geometry-version "${SCENEPROOF_GEOMETRY_VERSION:-v4_deepsearch}" \
  --baseline-version "$incumbent" \
  "${runtime_args[@]}" \
  --runtime-composite "$target=$runtime_components" \
  --metrics-out "$audit/physical.json" \
  --scene-csv "$audit/physical_scenes.csv" \
  --object-csv "$audit/physical_objects.csv" \
  --report-out "$audit/physical.txt"

"$PY" eval_gt_metrics.py \
  --saved-results "$root" \
  --scenes "$manifest" \
  --versions "$incumbent,$candidate,$target" \
  --min-visible-mask-area 8000 \
  --min-visible-bbox-size 0 \
  --batch-logs logs \
  --metrics-out "$audit/gt_8000.json" \
  --manifest-out "$audit/gt_manifest_8000.json"

gate_args=()
if [[ "${SCENEPROOF_ALLOW_SAFE_ABSTAIN:-0}" == "1" ]]; then
  gate_args+=(--allow-safe-abstain)
fi

"$PY" sceneproof_postsim_component_gate.py \
  --certificate "$audit/certificate.json" \
  --physical "$audit/physical.json" \
  --gt "$audit/gt_8000.json" \
  --legacy-version "$legacy" \
  --incumbent-version "$incumbent" \
  --target-version "$target" \
  --minimum-speedup 1.5 \
  "${gate_args[@]}" \
  --out "$audit/final_gates.json"

cat "$audit/physical.txt"
echo "CERTIFICATE=$audit/certificate.json"
echo "PHYSICAL=$audit/physical.json"
echo "GT=$audit/gt_8000.json"
echo "FINAL_GATES=$audit/final_gates.json"
