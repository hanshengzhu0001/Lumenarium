#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
root="a10_reusable_results/paper30"
candidate="v5_sceneproof_pose_serialization_smoke1_fix76"
factor_root="$root/sceneba_audit/$candidate/true_mesh_com_smoke1_fix79"
action_audit="$factor_root/com_factorization_audit.json"
probe_root="$factor_root/local_settle_oracle_fix80"
audit="$probe_root/oracle_fix81.json"
report="$probe_root/oracle_fix81.txt"

"$HOME/.venvs/lumenarium-py311/bin/python" \
  sceneproof_local_settle_oracle_fix80.py \
  --action-audit "$action_audit" \
  --probe-root "$probe_root" \
  --out "$audit" \
  --report "$report"

echo "FIX81_LOCAL_SETTLE_AUDIT=$(readlink -f "$audit")"
echo "FIX81_LOCAL_SETTLE_REPORT=$(readlink -f "$report")"
