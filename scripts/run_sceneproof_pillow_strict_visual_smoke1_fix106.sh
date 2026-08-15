#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
targets="/tmp/sceneproof_pillow_strict_visual_fix106.tsv"
cat > "$targets" <<'EOF'
bedroom_01	pillow_0
bedroom_01	pillow_1
bedroom_01	pillow_2
bedroom_01	pillow_3
bedroom_01	pillow_6
EOF

export SCENEPROOF_RIGID_TARGETS_FILE="$targets"
export SCENEPROOF_RIGID_TARGET_VERSION="v5_sceneproof_pillow_strict_visual_smoke1_fix106"
# Acceptance remains fail-closed, but permits the witnessed support-proxy
# exemption when true-mesh contact is certified and COM margin is positive.
export SCENEPROOF_ACCEPT_POLICY=relaxed
export SCENEPROOF_FORCE_MEASURED_CANDIDATES=1
export SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}"

bash scripts/run_sceneproof_rigid_only_adaptive_eval_fix84e.sh
