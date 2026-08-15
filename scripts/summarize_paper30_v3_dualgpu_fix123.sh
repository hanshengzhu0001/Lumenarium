#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
python="$HOME/.venvs/lumenarium-py311/bin/python"
root="$HOME/Lumenarium/a10_reusable_results/paper30"
log_root="$HOME/Lumenarium/logs/paper30_v3_cold_fix123"
out="$root/sceneba_audit/v5_sceneproof_fast_visual_paper30_fix121/v3_cold_fix123_runtime_reconstructed.json"
"$python" - "$root/manifest.txt" "$log_root/runtime_gpu0.jsonl" "$log_root/runtime_gpu1.jsonl" "$out" <<'PY'
import json, statistics, sys
from pathlib import Path
manifest, *runtime_paths, out = map(Path, sys.argv[1:])
scenes = [x.strip() for x in manifest.read_text().splitlines() if x.strip()]
by_scene = {}
for path in runtime_paths:
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        if row.get("status") == "ok": by_scene[row["scene"]] = row
missing = sorted(set(scenes) - set(by_scene))
if missing: raise SystemExit(f"incomplete successful runtime rows: {missing}")
lanes = {0: 0.0, 1: 0.0}
values = []
for row in by_scene.values():
    value = float(row["elapsed_seconds"]); values.append(value); lanes[int(row["gpu"])] += value
ordered = sorted(values)
record = {
  "schema_version": "lumenarium_reconstructed_dual_gpu_runtime_v1",
  "version": "v3_cold_paper30_fix123", "scenes": 30, "a10_devices": 2,
  "timing_status": "reconstructed_from_original_static_odd_even_lanes_after_gpu1_pause",
  "lane_seconds": {str(k): v for k,v in lanes.items()},
  "reconstructed_wall_clock_seconds": max(lanes.values()),
  "gpu_total_seconds": sum(values), "mean_scene_seconds": statistics.mean(values),
  "median_scene_seconds": statistics.median(values),
  "p90_scene_seconds": ordered[min(29, int(.9*30))],
}
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record, indent=2))
print(f"V3_RECONSTRUCTED_RUNTIME={out}")
PY
