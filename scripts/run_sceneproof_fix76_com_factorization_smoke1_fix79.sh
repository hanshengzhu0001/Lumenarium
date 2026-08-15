#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
scene="bedroom_01"
baseline="v5_sceneproof_visual_rollback_smoke1_fix43"
candidate="v5_sceneproof_pose_serialization_smoke1_fix76"
manifest="/tmp/sceneproof_fix76_com_factorization_smoke1_fix79.txt"
audit_root="$root/sceneba_audit/$candidate/true_mesh_com_smoke1_fix79"
log_root="logs/sceneproof_fix76_com_factorization_smoke1_fix79"
physical="$root/sceneba_audit/v5_sceneproof_collision_partial_commit_certified_paper30_fix61/physical_objects.csv"
printf '%s\n' "$scene" > "$manifest"

env \
  SCENEPROOF_COM_MANIFEST="$manifest" \
  SCENEPROOF_COM_SMOOTH_VERSION="$baseline" \
  SCENEPROOF_COM_FINAL_VERSION="$candidate" \
  SCENEPROOF_COM_AUDIT_ROOT="$audit_root" \
  SCENEPROOF_COM_LOG_ROOT="$log_root" \
  SCENEPROOF_COM_PHYSICAL_OBJECTS="$physical" \
  bash scripts/run_sceneproof_true_mesh_com_paper30_fix64.sh

baseline_audit="$audit_root/${scene}__${baseline}.json"
candidate_audit="$audit_root/${scene}__${candidate}.json"
action_audit="$audit_root/com_factorization_audit.json"
"$HOME/.venvs/lumenarium-py311/bin/python" \
  sceneproof_com_action_audit_fix78.py \
  --baseline "$baseline_audit" \
  --candidate "$candidate_audit" \
  --out "$action_audit"

echo "FIX79_RESPONSIBILITY=$(readlink -f "$audit_root/responsibility.json")"
echo "FIX79_FACTORIZATION=$(readlink -f "$action_audit")"
echo "FIX79_REPORT=$(readlink -f "$audit_root/responsibility.txt")"
