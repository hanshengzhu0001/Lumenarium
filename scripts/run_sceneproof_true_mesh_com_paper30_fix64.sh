#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

final="${SCENEPROOF_COM_FINAL_VERSION:-v5_sceneproof_collision_partial_commit_certified_paper30_fix61}"
export SCENEPROOF_COM_AUDIT_ROOT="${SCENEPROOF_COM_AUDIT_ROOT:-a10_reusable_results/paper30/sceneba_audit/${final}/true_mesh_com_fix64}"
export SCENEPROOF_COM_LOG_ROOT="${SCENEPROOF_COM_LOG_ROOT:-logs/sceneproof_true_mesh_com_paper30_fix64}"

exec bash scripts/run_sceneproof_true_mesh_com_paper30_fix62.sh
