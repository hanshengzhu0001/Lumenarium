#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
PY="${SCENEPROOF_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="a10_reusable_results/paper30"
target="v5_sceneproof_com_scoped_rollback_paper30_fix68"
protocol="$root/sceneba_audit/v5_sceneproof_collision_partial_commit_certified_paper30_fix61/true_mesh_com_paper30_fix66/protocol.json"
manifest="/tmp/sceneproof_com_scoped_rollback_render_fix69.txt"
render_log="logs/sceneproof_com_scoped_rollback_render_fix69"
collection="$root/sceneproof_com_scoped_rollback_comparison_fix69"
archive="$HOME/sceneproof_com_scoped_rollback_comparison_fix69.tar.gz"

"$PY" - "$protocol" "$manifest" <<'PY'
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
scenes = [
    scene for scene, row in protocol["scenes"].items()
    if row.get("implementation_authorized")
]
if len(scenes) != 6:
    raise SystemExit(f"expected 6 authorized scenes, found {len(scenes)}")
Path(sys.argv[2]).write_text("\n".join(scenes) + "\n", encoding="utf-8")
print("RENDER_SCENES=" + ",".join(scenes))
PY

SCENEPROOF_MANIFEST="$manifest" \
SCENEPROOF_CERTIFIED_VERSION="$target" \
SCENEPROOF_RENDER_LOG_ROOT="$render_log" \
SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-16000}" \
  bash scripts/render_sceneproof_certified_paper30.sh

"$PY" sceneproof_collect_paper30_renders.py \
  --saved-results "$root" --scenes "$manifest" \
  --legacy-version v4_legacy_sa5000_bench \
  --control-version v5_sceneproof_fix43_smooth_paper30_fix61 \
  --candidate-version v5_sceneproof_collision_partial_commit_certified_paper30_fix61 \
  --certified-version "$target" \
  --out-dir "$collection" --archive "$archive"

echo "FIX69_RENDER_MANIFEST=$(readlink -f "$manifest")"
echo "FIX69_RENDER_COLLECTION=$(readlink -f "$collection")"
echo "FIX69_RENDER_ARCHIVE=$(readlink -f "$archive")"
