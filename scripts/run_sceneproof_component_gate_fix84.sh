#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
scene="${SCENEPROOF_SCENE:-bedroom_01}"
incumbent="${SCENEPROOF_INCUMBENT_VERSION:-v5_sceneproof_pose_serialization_smoke1_fix76}"
candidate="${SCENEPROOF_CANDIDATE_VERSION:-v5_sceneproof_local_settle_candidate_smoke1_fix82}"
audit="$root/sceneba_audit/$candidate"
python="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
object_id="${SCENEPROOF_OBJECT_ID:-single_sofa_chair_1}"
probe="$audit/probes/${object_id}.json"
gates_out="$audit/component_gates_fix84.json"

test -s "$probe" || { echo "Missing probe: $probe" >&2; exit 2; }
test -s "$audit/physical.json" || { echo "Missing physical.json" >&2; exit 2; }
test -s "$audit/physical_objects.csv" || { echo "Missing physical_objects.csv" >&2; exit 2; }
test -s "$audit/gt_8000.json" || { echo "Missing gt_8000.json" >&2; exit 2; }

echo "=== Unit tests for the witnessed exemption ==="
"$python" -m unittest tests.test_sceneproof_support_proxy_exemption -v 2>&1
echo ""

echo "=== Fix84 gates: exemption DISABLED (must reproduce Fix82 failure) ==="
"$python" sceneproof_local_settle_component_gate_fix84.py \
  --probe "$probe" \
  --physical "$audit/physical.json" \
  --physical-objects "$audit/physical_objects.csv" \
  --gt "$audit/gt_8000.json" \
  --incumbent-version "$incumbent" \
  --candidate-version "$candidate" \
  --out "$audit/component_gates_fix84_strict.json"
echo ""

echo "=== Fix84 gates: exemption ENABLED (witnessed proxy disagreement) ==="
"$python" sceneproof_local_settle_component_gate_fix84.py \
  --probe "$probe" \
  --physical "$audit/physical.json" \
  --physical-objects "$audit/physical_objects.csv" \
  --gt "$audit/gt_8000.json" \
  --incumbent-version "$incumbent" \
  --candidate-version "$candidate" \
  --allow-support-proxy-exemption \
  --out "$gates_out"

echo ""
echo "FIX84_STRICT=$(readlink -f "$audit/component_gates_fix84_strict.json")"
echo "FIX84_GATES=$(readlink -f "$gates_out")"
