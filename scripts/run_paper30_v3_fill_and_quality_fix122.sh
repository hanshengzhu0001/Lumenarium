#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
python="$HOME/.venvs/lumenarium-py311/bin/python"
root="$HOME/Lumenarium/a10_reusable_results/paper30"
manifest="$root/manifest.txt"
target=v5_sceneproof_fast_visual_paper30_fix121
audit="$root/sceneba_audit/$target"
log_root="$HOME/Lumenarium/logs/paper30_v3_fill_fix122"
runtime_config="$audit/config_paper30_v3_fix122.yaml"
missing="$audit/v3_missing_fix122.txt"
mkdir -p "$audit/physical_native" "$log_root"

"$python" - "$manifest" "$root" "$missing" <<'PY'
import sys
from pathlib import Path
manifest, root, out = map(Path, sys.argv[1:])
scenes = [x.strip() for x in manifest.read_text().splitlines() if x.strip()]
missing = []
for scene in scenes:
    folder = root / f"{scene}_v3_result" / "S4_layout_refinement"
    if len(list(folder.glob("*_placement_info_s4.json"))) != 1:
        missing.append(scene)
out.write_text("\n".join(missing) + ("\n" if missing else ""))
print(f"V3_CACHE ready={len(scenes)-len(missing)}/{len(scenes)} missing={len(missing)}")
print("V3_MISSING=" + ",".join(missing))
PY

"$python" - "config/config_paper30_v3_final.yaml" "$runtime_config" "$root" <<'PY'
import re, sys
from pathlib import Path
source, target, root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
text = source.read_text(encoding="utf-8")
text, n = re.subn(
    r'(?m)^(\s*save_parent_folder\s*:\s*).+$',
    lambda m: f'{m.group(1)}"{root.as_posix()}"', text, count=1)
if n != 1: raise SystemExit("cannot rewrite v3 save_parent_folder")
text = text.replace(
    "/mnt/kevinzyz/artifacts/Imaginarium-repo/weights_cache/", "weights/")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(text, encoding="utf-8")
PY

list0="$(mktemp /tmp/sceneproof_v3_fix122_gpu0_XXXXXX)"
list1="$(mktemp /tmp/sceneproof_v3_fix122_gpu1_XXXXXX)"
trap 'rm -f -- "$list0" "$list1"' EXIT
awk 'NR % 2 == 1' "$missing" > "$list0"
awk 'NR % 2 == 0' "$missing" > "$list1"

run_worker() {
  local gpu="$1" list="$2" scene image result backup v1_result stage started ended elapsed rc status
  local runtime="$log_root/runtime_gpu${gpu}.jsonl"
  touch "$runtime"
  while IFS= read -r scene || test -n "$scene"; do
    scene="${scene%$'\r'}"; test -n "$scene" || continue
    image="demo/${scene}_v3.png"
    if ! test -s "$image"; then
      test -s "demo/${scene}_v1.png" || { echo "FAIL scene=$scene missing_image"; return 1; }
      cp "demo/${scene}_v1.png" "$image"
    fi
    result="$root/${scene}_v3_result"
    if test -d "$result"; then
      backup="${result}.pre_fix122_$(date +%s)"
      mv "$result" "$backup"
      echo "PRESERVED_PARTIAL scene=$scene path=$backup"
    fi
    mkdir -p "$result"
    v1_result="$root/${scene}_v1_result"
    for stage in S0_geometry_pred_results S1_scene_parsing_results; do
      test -d "$v1_result/$stage" || {
        echo "FAIL scene=$scene reason=missing_v1_cache stage=$stage"
        return 1
      }
      cp -a "$v1_result/$stage" "$result/$stage"
    done
    echo "REUSED_V1_S01 scene=$scene source=$v1_result"
    started="$(date +%s%N)"
    echo "START_V3 scene=$scene gpu=$gpu $(date)"
    set +e
    timeout "${IMAGINARIUM_SCENE_TIMEOUT:-14400}" env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      IMAGINARIUM_FLOOR_VERIFY_V2=1 \
      IMAGINARIUM_S3_STACK_AWARE=1 \
      IMAGINARIUM_S4_STACK_AWARE=1 \
      IMAGINARIUM_S1_LOWCAT_PASS=1 \
      IMAGINARIUM_USE_SAM3_DETECTION=1 \
      IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
      IMAGINARIUM_GPT_LOCK_FILE=/tmp/lumenarium_v3_fix122_gpt.lock \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONUNBUFFERED=1 \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$python" -u run_imaginarium_I2Layout_v3.py "$image" \
      --config "$runtime_config" \
      > "$log_root/${scene}_gpu${gpu}.log" 2>&1
    rc=$?
    set -e
    ended="$(date +%s%N)"
    elapsed="$($python -c "print(($ended-$started)/1e9)")"
    status=fail
    compgen -G "$result/S4_layout_refinement/*_placement_info_s4.json" >/dev/null \
      && test "$rc" -eq 0 && status=ok
    printf '{"scene":"%s","gpu":%s,"elapsed_seconds":%.6f,"status":"%s","return_code":%s}\n' \
      "$scene" "$gpu" "$elapsed" "$status" "$rc" >> "$runtime"
    echo "DONE_V3 scene=$scene gpu=$gpu elapsed=$elapsed status=$status"
    test "$status" = ok || return 1
  done < "$list"
}

run_worker 0 "$list0" > "$log_root/gpu0.log" 2>&1 & p0=$!
run_worker 1 "$list1" > "$log_root/gpu1.log" 2>&1 & p1=$!
echo "V3_GPU0_PID=$p0"
echo "V3_GPU1_PID=$p1"
status=0; wait "$p0" || status=1; wait "$p1" || status=1
test "$status" -eq 0 || { echo "V3_FILL_STOP reason=worker_failure"; exit 1; }

versions="v1,v3,v4_deepsearch,$target"
labels="v1,v3,v4-deepsearch,v5-fast"
"$python" eval_gt_metrics.py --saved-results "$root" --scenes "$manifest" \
  --versions "$versions" --min-visible-mask-area 8000 --min-visible-bbox-size 0 \
  --batch-logs logs --metrics-out "$audit/gt_8000_with_v3_fix122.json" \
  --manifest-out "$audit/gt_manifest_8000_with_v3_fix122.json"

"$python" eval_physical_realizability.py --saved-results "$root" \
  --scenes "$manifest" --versions v3 --geometry-version v3 \
  --baseline-version v3 --collision-policy legacy \
  --metrics-out "$audit/physical_native/v3.json" \
  --scene-csv "$audit/physical_native/v3.scenes.csv" \
  --object-csv "$audit/physical_native/v3.objects.csv" \
  --report-out "$audit/physical_native/v3.txt"

"$python" sceneproof_cross_version_quality_dashboard.py \
  --gt "$audit/gt_8000_with_v3_fix122.json" \
  --physical-dir "$audit/physical_native" \
  --versions "$versions" --labels "$labels" \
  --out-json "$audit/cross_version_quality_with_v3_fix122.json" \
  --out-csv "$audit/cross_version_quality_with_v3_fix122.csv" \
  --out-txt "$audit/cross_version_quality_with_v3_fix122.txt"

echo "V3_FIX122_FINISHED status=0"
echo "V3_FIX122_QUALITY=$audit/cross_version_quality_with_v3_fix122.json"
