#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
export SCENEPROOF_OVERHANG_AUDIT_VERSION="v5_sceneproof_vertical_first_contact_smoke1_fix112"
exec bash scripts/run_sceneproof_support_overhang_projection_audit_smoke1_fix111.sh
