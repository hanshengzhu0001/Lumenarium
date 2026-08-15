#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENEPROOF_RESULTS_ROOT:-$HOME/Lumenarium/a10_reusable_results/fix124_v5_fast_cold_paper30}"
manifest="${SCENEPROOF_MANIFEST:-$root/manifest.txt}"
run_id="fix124_v5_fast_cold_paper30"
source="v4_deepsearch"
baseline="v5_sceneproof_collision_partial_commit_certified_${run_id}"
target="v5_sceneproof_vertical_support_visual_${run_id}"
main_log="$HOME/Lumenarium/logs/${run_id}.log"
worker_log_root="$HOME/Lumenarium/logs/$target"
audit="$root/sceneba_audit/$target"
timing_root="$audit/retry_timing_fix125"
snapshot="$timing_root/initial_failed_attempt"
mkdir -p "$snapshot" "$timing_root"

test -x "$python" || { echo "Missing Python: $python" >&2; exit 2; }
test -s "$manifest" || { echo "Missing manifest: $manifest" >&2; exit 2; }

# Preserve the failed-attempt evidence before the Fix114 runner rewrites its
# worker summaries and per-scene logs.  Existing result placements are never
# removed; the runner will cache all 26 completed scenes.
for path in \
  "$worker_log_root/gpu0.log" \
  "$worker_log_root/gpu1.log" \
  "$root/sceneba_audit/$target/summary.json" \
  "$main_log"; do
  test -f "$path" && cp -p "$path" "$snapshot/$(basename "$path")"
done
for scene in computer_room_01 official_01 bathroom_06 workshop_03; do
  for path in "$worker_log_root/${scene}"_gpu*.log; do
    test -f "$path" && cp -p "$path" "$snapshot/$(basename "$path")"
  done
done

now_ns() { date +%s%N; }
seconds_between() { "$python" -c "print(($2-$1)/1e9)"; }

retry_start="$(now_ns)"
set +e
env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_FIX114_SOURCE_VERSION="$source" \
  SCENEPROOF_FIX114_BASELINE_VERSION="$baseline" \
  SCENEPROOF_FIX114_TARGET_VERSION="$target" \
  SCENEPROOF_FIX114_SKIP_RENDER=1 \
  SCENEPROOF_FIX114_SKIP_COMPARISON_ARCHIVE=1 \
  IMAGINARIUM_SCENEPROOF_AUDIT_POSE_RESTORE_TOLERANCE=1e-4 \
  bash scripts/run_sceneproof_vertical_support_final_paper30_fix114.sh \
  > "$timing_root/fix114_retry_and_eval.log" 2>&1
retry_rc=$?
set -e
retry_end="$(now_ns)"
retry_seconds="$(seconds_between "$retry_start" "$retry_end")"

if test "$retry_rc" -ne 0; then
  echo "FIX125_STOP stage=fix114_retry rc=$retry_rc elapsed=$retry_seconds"
  exit "$retry_rc"
fi

render_start="$(now_ns)"
set +e
env \
  SCENEPROOF_RESULTS_ROOT="$root" \
  SCENEPROOF_MANIFEST="$manifest" \
  SCENEPROOF_RENDER_SOURCE_VERSION="$source" \
  SCENEPROOF_CERTIFIED_VERSION="$target" \
  SCENEPROOF_RENDER_LOG_ROOT="$timing_root/final_render" \
  SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
  bash scripts/render_sceneproof_certified_paper30.sh \
  > "$timing_root/final_render.log" 2>&1
render_rc=$?
set -e
render_end="$(now_ns)"
render_seconds="$(seconds_between "$render_start" "$render_end")"
test "$render_rc" -eq 0 || {
  echo "FIX125_STOP stage=final_render rc=$render_rc elapsed=$render_seconds"
  exit "$render_rc"
}

"$python" - \
  "$manifest" "$root" "$target" "$snapshot" "$worker_log_root" \
  "$retry_seconds" "$render_seconds" "$timing_root/timing.json" <<'PY'
import json
import re
import sys
from datetime import datetime
from pathlib import Path

manifest, root, target, snapshot, current_logs = map(Path, sys.argv[1:6])
retry_wall = float(sys.argv[6])
render_wall = float(sys.argv[7])
out = Path(sys.argv[8])
scenes = [x.strip() for x in manifest.read_text().splitlines() if x.strip()]
failed_scenes = ["computer_room_01", "official_01", "bathroom_06", "workshop_03"]

stamp = re.compile(
    r"(?:START|DONE) scene=(\S+) gpu=(\d+)(?: rc=(\d+))? "
    r"(\w{3} \w{3}\s+\d+ \d\d:\d\d:\d\d \w+ \d{4})"
)

def parse_worker(path):
    events = []
    if not path.is_file():
        return events
    for line in path.read_text(errors="replace").splitlines():
        match = stamp.search(line)
        if not match:
            continue
        event = "DONE" if line.startswith("DONE") else "START"
        when = datetime.strptime(match.group(4), "%a %b %d %H:%M:%S %Z %Y")
        events.append({
            "event": event,
            "scene": match.group(1),
            "gpu": int(match.group(2)),
            "return_code": None if match.group(3) is None else int(match.group(3)),
            "timestamp": when.timestamp(),
        })
    return events

def durations(directory):
    events = parse_worker(directory / "gpu0.log") + parse_worker(directory / "gpu1.log")
    starts = {(x["scene"], x["gpu"]): x for x in events if x["event"] == "START"}
    rows = []
    for done in (x for x in events if x["event"] == "DONE"):
        start = starts.get((done["scene"], done["gpu"]))
        if start is None:
            continue
        rows.append({
            "scene": done["scene"],
            "gpu": done["gpu"],
            "elapsed_seconds": done["timestamp"] - start["timestamp"],
            "return_code": done["return_code"],
        })
    return rows

initial_rows = durations(snapshot)
retry_rows = [row for row in durations(current_logs) if row["scene"] in failed_scenes]
completed = []
for scene in scenes:
    placement = (
        root / f"{scene}_{target.name}_result" / "S4_layout_refinement" /
        f"{scene}_{target.name}_placement_info_s4.json"
    )
    if placement.is_file() and placement.stat().st_size:
        completed.append(scene)

record = {
    "schema_version": "sceneproof_fix124_retry_timing_v1",
    "policy": "preserve_26_completed_retry_4_float_restore_parity_then_eval_and_render",
    "pose_restore_tolerance": 1e-4,
    "scenes": len(scenes),
    "completed_final_s4": len(completed),
    "initial_failed_scenes": failed_scenes,
    "initial_fix114_scene_rows": initial_rows,
    "initial_failed_attempt_gpu_seconds": sum(
        row["elapsed_seconds"] for row in initial_rows
        if row["scene"] in failed_scenes
    ),
    "retry_scene_rows": retry_rows,
    "retry_useful_gpu_seconds": sum(row["elapsed_seconds"] for row in retry_rows),
    "retry_fix114_and_eval_wall_seconds": retry_wall,
    "final_render_wall_seconds": render_wall,
    "supplement_wall_seconds": retry_wall + render_wall,
    "accounting_note": (
        "Retry and render wall times are exact for Fix125. Initial Fix114 per-scene "
        "times are reconstructed to one-second resolution from preserved worker logs."
    ),
}
out.write_text(json.dumps(record, indent=2) + "\n")
print(f"FIX125_FINAL_S4={len(completed)}/{len(scenes)}")
print(f"FIX125_RETRY_GPU_SECONDS={record['retry_useful_gpu_seconds']:.3f}")
print(f"FIX125_RETRY_WALL_SECONDS={retry_wall:.3f}")
print(f"FIX125_RENDER_WALL_SECONDS={render_wall:.3f}")
print(f"FIX125_TIMING={out.resolve()}")
PY

echo "FIX125_FINISHED status=0"
echo "FIX125_TIMING=$(readlink -f "$timing_root/timing.json")"
echo "FIX125_FINAL_EVAL=$(readlink -f "$audit/final_eval.json")"
