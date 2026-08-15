#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"

PY="${SCENEBA_WITNESS_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
root="${SCENEBA_WITNESS_RESULTS_ROOT:-a10_reusable_results/paper30}"
audit="${SCENEBA_WITNESS_AUDIT_ROOT:-$root/sceneba_audit/moge_noc_witness_v1}"
blind="${SCENEBA_WITNESS_BLIND_MANIFEST:-$root/sceneba_audit/asset_vlm_recoverable_v1/blind_manifest.json}"
gpu="${SCENEBA_WITNESS_GPU_ID:-1}"
gpu_floor="${SCENEBA_WITNESS_GPU_FREE_FLOOR_MB:-16000}"
source_version="${SCENEBA_WITNESS_SOURCE_VERSION:-v4_deepsearch}"
log_root="${SCENEBA_WITNESS_LOG_ROOT:-logs/sceneba_moge_noc_witness_v1}"
scene_objects="/tmp/sceneba_witness_scene_objects.$$.tsv"
mkdir -p "$audit/cache" "$log_root"
trap 'rm -f -- "$scene_objects"' EXIT

test -s "$blind" || { echo "Missing blind manifest: $blind" >&2; exit 2; }

"$PY" - "$blind" "$scene_objects" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

blind, output = map(Path, sys.argv[1:])
groups = defaultdict(list)
for sample in json.loads(blind.read_text())["samples"]:
    groups[sample["scene"]].append(sample["object_id"])
output.write_text(
    "".join(
        f"{scene}\t{','.join(sorted(set(objects)))}\n"
        for scene, objects in sorted(groups.items())
    )
)
PY

while IFS=$'\t' read -r scene object_ids; do
    test -n "$scene" || continue
    output="$root/${scene}_${source_version}_result/S3_pose_inference/${scene}_sceneba_witness_pose_bank.json"
    if test -s "$output"; then
        echo "CACHED_POSE_BANK scene=$scene"
        continue
    fi
    while true; do
        free="$(
            nvidia-smi -i "$gpu" --query-gpu=memory.free \
                --format=csv,noheader,nounits 2>/dev/null |
                head -1 | tr -d ' '
        )"
        if test -n "$free" && test "$free" -ge "$gpu_floor"; then
            break
        fi
        echo "WAIT_GPU scene=$scene gpu=$gpu free=${free:-unknown}MiB $(date)"
        sleep 60
    done
    echo "START_POSE_BANK scene=$scene objects=$object_ids gpu=$gpu $(date)"
    env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH=1 \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$PY" -u sceneba_build_pose_bank.py \
        --saved-results "$root" \
        --scene "$scene" \
        --source-version "$source_version" \
        --top-k-assets 3 \
        --top-k-views 3 \
        --yaw-offsets 0,90,180,270 \
        --object-ids "$object_ids" \
        --capture-dense-correspondences \
        --out "$output" \
        > "$log_root/${scene}_pose_bank.log" 2>&1
    echo "OK_POSE_BANK scene=$scene $(date)"
done < "$scene_objects"

echo "START_BLIND_WITNESS $(date)"
env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
  "$PY" -u sceneba_moge_noc_witness.py \
    --saved-results "$root" \
    --blind-manifest "$blind" \
    --source-version "$source_version" \
    --device cuda:0 \
    --cache-dir "$audit/cache" \
    --out "$audit/blind_witness_table.json" \
    2>&1 | tee "$log_root/blind_witness_table.log"

echo "WITNESS_BLIND_STAGE_FINISHED table=$audit/blind_witness_table.json"
echo "The answer key has not been read. Run scripts/eval_sceneba_moge_noc_witness8.sh next."
