#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

control="v5_sceneproof_fix43_smooth_paper30_fix61"
candidate="v5_sceneproof_collision_partial_commit_paper30_fix61"
target="v5_sceneproof_collision_partial_commit_certified_paper30_fix61"
audit="a10_reusable_results/paper30/sceneba_audit/$target"

env \
  SCENEPROOF_FIX43_SOURCE_MANIFEST=a10_reusable_results/paper30/manifest.txt \
  SCENEPROOF_FIX43_MANIFEST=/tmp/sceneproof_collision_partial_commit_paper30_fix61.txt \
  SCENEPROOF_FIX43_EXPECTED_SCENES=30 \
  SCENEPROOF_FIX43_MINIMUM_NONZERO_SCENES=18 \
  SCENEPROOF_FIX43_CONTROL_VERSION="$control" \
  SCENEPROOF_FIX43_GUARDED_VERSION="$candidate" \
  SCENEPROOF_FIX43_CERTIFIED_VERSION="$target" \
  SCENEPROOF_FIX43_RENDER_DIR=a10_reusable_results/paper30/sceneproof_collision_partial_commit_paper30_renders_fix61 \
  SCENEPROOF_FIX43_RENDER_ARCHIVE="$HOME/sceneproof_collision_partial_commit_paper30_renders_fix61.tar.gz" \
  bash scripts/run_sceneproof_fix43_inloop_fullstack_smoke5_fix56.sh

PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
"$PY" sceneba_paired_audit.py \
  --gt-metrics "$audit/gt_8000.json" \
  --physical-metrics "$audit/physical.json" \
  --baseline "$control" --candidate "$target" \
  --samples 10000 --rotation-margin -0.01 --translation-margin -0.005 \
  --out "$audit/paired_bootstrap_10000.json"

"$PY" - "$audit/protocol.json" <<'PY'
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not protocol.get("passed", False):
    raise SystemExit("Paper30 protocol failed; do not freeze Fix61")
print("PAPER30_PROTOCOL_CONFIRMED=PASS")
PY

echo "PAPER30_FIX61_FINISHED target=$target"
echo "FINAL_GATES=$HOME/Lumenarium/$audit/final_gates.json"
echo "PROTOCOL=$HOME/Lumenarium/$audit/protocol.json"
echo "PAIRED_BOOTSTRAP=$HOME/Lumenarium/$audit/paired_bootstrap_10000.json"
echo "RENDER_ARCHIVE=$HOME/sceneproof_collision_partial_commit_paper30_renders_fix61.tar.gz"
