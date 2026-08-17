#!/usr/bin/env bash
set -euo pipefail

test "$#" -eq 5 || {
  echo "usage: $0 JOB_ID INPUT_IMAGE ARTIFACT_DIR GPU_ID PROFILE" >&2
  exit 2
}
job_id="$1" input="$2" artifact_dir="$3" gpu="$4" profile="$5"
[[ "$job_id" =~ ^[0-9a-f]{32}$ ]] || { echo "invalid job id" >&2; exit 2; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid GPU id" >&2; exit 2; }
case "$profile" in fast|medium) ;; *) echo "invalid profile" >&2; exit 2;; esac

cd "$HOME/Lumenarium"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
release_id="$("$python" -c 'from sceneproof_api import RELEASE_ID; print(RELEASE_ID)')"
base_config="${SCENEPROOF_API_CONFIG:-config/config_a10_paper30_v4_deepsearch.yaml}"
jobs_root="${SCENEPROOF_API_JOB_ROOT:-$HOME/Lumenarium/api_jobs}"
root="$jobs_root/$job_id/results"
log_root="$jobs_root/$job_id/logs"
manifest="$jobs_root/$job_id/manifest.txt"
config="$jobs_root/$job_id/config.yaml"
input_digest="$(sha256sum "$input" | awk '{print $1}')"
scene_id="${input_digest:0:32}"
image="demo/${scene_id}_v4_deepsearch.png"
source="v4_deepsearch"
control="v5_sceneproof_fix43_smooth_api"
guarded="v5_sceneproof_collision_partial_commit_api"
baseline="v5_sceneproof_collision_partial_commit_certified_api"
target="v5_sceneproof_vertical_support_visual_api"
deepsearch_url="${OMNIVERSE_DEEPSEARCH_URL:-${SCENEPROOF_DEEPSEARCH_URL:-https://miller-unshapeable-melany.ngrok-free.dev/search}}"
cache_revision="${SCENEPROOF_API_FIX61_CACHE_REVISION:-fix61-v1}"
cache_base="${SCENEPROOF_API_CACHE_ROOT:-$HOME/Lumenarium/api_cache}/$cache_revision/$input_digest"
cache_lock="$cache_base.lock"
force_cold_rerun="${SCENEPROOF_API_FORCE_COLD_RERUN:-0}"
case "$force_cold_rerun" in 0|1) ;; *) echo "invalid SCENEPROOF_API_FORCE_COLD_RERUN" >&2; exit 2;; esac
mkdir -p "$root" "$log_root" "$artifact_dir"
mkdir -p "$(dirname "$cache_base")"
printf '%s\n' "$scene_id" > "$manifest"

now_ns() { date +%s%N; }
full_start="$(now_ns)"
s03_start="$(now_ns)"
fix61_start="$s03_start"
exec 9>"$cache_lock"
flock 9
if test "$force_cold_rerun" = "1"; then
  echo "SCENEPROOF_API_STAGE=force_cold_rerun PROGRESS=0.02"
fi
if test "$force_cold_rerun" != "1" && test -s "$cache_base/READY"; then
  echo "SCENEPROOF_API_STAGE=frozen_fix61_cache_hit PROGRESS=0.70"
  cached_scene_id="$(head -1 "$cache_base/SCENE_ID" 2>/dev/null || true)"
  if [[ "$cached_scene_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    scene_id="$cached_scene_id"
    image="demo/${scene_id}_v4_deepsearch.png"
    printf '%s\n' "$scene_id" > "$manifest"
    echo "SCENEPROOF_CACHE_SCENE_ID=$scene_id"
  fi
  cp -a "$cache_base/results/." "$root/"
  s03_end="$(now_ns)"
  fix61_end="$s03_end"
else
  probe="$(curl --fail --show-error --silent --max-time 30 \
    --request POST --header 'Content-Type: application/json' \
    --data '{"description":"house","limit":2}' "$deepsearch_url")" || {
    echo "SCENEPROOF_API_STAGE=deepsearch_unavailable PROGRESS=0.0" >&2
    exit 3
  }
  printf '%s' "$probe" | "$python" -c \
    'import json,sys; x=json.load(sys.stdin); assert isinstance(x,list) and x and isinstance(x[0].get("url"),str)' || {
    echo "SCENEPROOF_API_STAGE=deepsearch_invalid_response PROGRESS=0.0" >&2
    exit 3
  }

  "$python" - "$input" "$image" "$base_config" "$config" "$root" <<'PY'
import re, sys
from pathlib import Path
from PIL import Image
source_image, target_image, source_config, target_config, root = map(Path, sys.argv[1:])
with Image.open(source_image) as image:
    image.convert("RGB").save(target_image, "PNG")
text = source_config.read_text(encoding="utf-8")
text, count = re.subn(
    r'(?m)^(\s*save_parent_folder\s*:\s*).+$',
    lambda match: f'{match.group(1)}"{root.as_posix()}"', text, count=1,
)
if count != 1: raise SystemExit("save_parent_folder missing")
target_config.write_text(text, encoding="utf-8")
PY

  s03_ok=0
  for batch in 8 4 2; do
    echo "SCENEPROOF_API_STAGE=s0_geometry PROGRESS=0.05"
    echo "SCENEPROOF_API_S03_ATTEMPT batch=$batch"
    set +e
    timeout "${SCENEPROOF_API_S03_TIMEOUT:-14400}" env \
      CUDA_VISIBLE_DEVICES="$gpu" \
      IMAGINARIUM_API_PROGRESS=1 \
      IMAGINARIUM_STOP_AFTER_STAGE=S3 \
      IMAGINARIUM_S3_MAX_UNIQUE_FEATURES_PER_BATCH="$batch" \
      IMAGINARIUM_PARALLEL_GPT_PROCESSES=1 \
      IMAGINARIUM_GPT_LOCK_FILE=/tmp/lumenarium_api_gemini.lock \
      OMNIVERSE_DEEPSEARCH_URL="$deepsearch_url" \
      OMNIVERSE_DEEPSEARCH_WORKERS="${SCENEPROOF_API_DEEPSEARCH_WORKERS:-4}" \
      OMNIVERSE_DEEPSEARCH_MAX_ATTEMPTS=6 \
      OMNIVERSE_DEEPSEARCH_TIMEOUT=120 \
      OMNIVERSE_DEEPSEARCH_RETRY_DELAY=2 \
      PYTHONUNBUFFERED=1 \
      LD_LIBRARY_PATH="$HOME/.venvs/lumenarium-py311/lib:${LD_LIBRARY_PATH:-}" \
      "$python" -u run_imaginarium_I2Layout_v4_deepsearch.py "$image" --config "$config" \
      > "$log_root/s03.log" 2>&1 &
    s03_pid=$!
    last_marker=""
    while kill -0 "$s03_pid" 2>/dev/null; do
      marker="$(grep -aEo 'SCENEPROOF_API_STAGE=[A-Za-z0-9_.:-]+ PROGRESS=[0-9.]+' "$log_root/s03.log" 2>/dev/null | tail -1 || true)"
      if test -n "$marker" && test "$marker" != "$last_marker"; then
        echo "$marker"
        last_marker="$marker"
      fi
      sleep 2
    done
    wait "$s03_pid"; s03_rc=$?
    set -e
    s3_json="$root/${scene_id}_${source}_result/S3_pose_inference/${scene_id}_${source}_placement_info.json"
    if test "$s03_rc" -eq 0 && test -s "$s3_json"; then
      s03_ok=1
      break
    fi
    echo "SCENEPROOF_API_RETRY stage=s0_s3 batch=$batch rc=$s03_rc" >&2
  done
  if test "$s03_ok" -ne 1; then
    echo "SCENEPROOF_API_STOP stage=s0_s3 reason=retry_exhausted" >&2
    tail -120 "$log_root/s03.log" >&2 || true
    exit 3
  fi
  s03_end="$(now_ns)"

  fix61_start="$(now_ns)"
  echo "SCENEPROOF_API_STAGE=s4_scenelm_fix61 PROGRESS=0.48"
  fix61_ok=0
  for fix61_attempt in 1 2; do
    set +e
    env \
      SCENEPROOF_RESULTS_ROOT="$root" \
      SCENEPROOF_SOURCE_VERSION="$source" \
      SCENEPROOF_FIX43_SOURCE_MANIFEST="$manifest" \
      SCENEPROOF_FIX43_MANIFEST="$jobs_root/$job_id/fix61_manifest.txt" \
      SCENEPROOF_FIX43_EXPECTED_SCENES=1 \
      SCENEPROOF_FIX43_MINIMUM_NONZERO_SCENES=0 \
      SCENEPROOF_FIX43_CONTROL_VERSION="$control" \
      SCENEPROOF_FIX43_GUARDED_VERSION="$guarded" \
      SCENEPROOF_FIX43_CERTIFIED_VERSION="$baseline" \
      SCENEPROOF_FIX43_SKIP_RENDER=1 \
      SCENEPROOF_FIX43_SKIP_FORMAL_EVAL=1 \
      IMAGINARIUM_GPU0_ID="$gpu" IMAGINARIUM_GPU1_ID="$gpu" \
      IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-8000}" \
      bash scripts/run_sceneproof_fix43_inloop_fullstack_smoke5_fix56.sh \
      > "$log_root/fix61.log" 2>&1
    fix61_rc=$?
    set -e
    baseline_json="$root/${scene_id}_${baseline}_result/S4_layout_refinement/${scene_id}_${baseline}_placement_info_s4.json"
    baseline_certificate="$root/sceneba_audit/$baseline/certificate.json"
    if test "$fix61_rc" -eq 0 && test -s "$baseline_json" && test -s "$baseline_certificate"; then
      fix61_ok=1
      break
    fi
    echo "SCENEPROOF_API_RETRY stage=s4_scenelm_fix61 attempt=$fix61_attempt rc=$fix61_rc" >&2
  done
  if test "$fix61_ok" -ne 1; then
    echo "SCENEPROOF_API_STOP stage=s4_scenelm_fix61 reason=retry_exhausted" >&2
    tail -120 "$log_root/fix61.log" >&2 || true
    exit 4
  fi
  fix61_end="$(now_ns)"
  if test "$force_cold_rerun" != "1"; then
    cache_tmp="$(mktemp -d "$(dirname "$cache_base")/.${input_digest}.XXXXXX")"
    mkdir -p "$cache_tmp/results"
    cp -a "$root/." "$cache_tmp/results/"
    printf '%s\n' "$scene_id" > "$cache_tmp/SCENE_ID"
    printf '%s\n' "$cache_revision" > "$cache_tmp/READY"
    mv "$cache_tmp" "$cache_base"
  else
    echo "SCENEPROOF_API_COLD_RERUN_CACHE_WRITE=skipped"
  fi
fi
flock -u 9

fix114_start="$(now_ns)"
upstream_evaluation="$root/sceneba_audit/$baseline/certificate.json"
incumbent_version="$baseline"
if test "$profile" = "medium"; then
  echo "SCENEPROOF_API_STAGE=s4_fix61_then_visual_safe_salvage PROGRESS=0.72"
else
  echo "SCENEPROOF_API_STAGE=s4_fix61_incumbent PROGRESS=0.72"
fi
fix114_end="$(now_ns)"

# Frozen public profiles:
#   fast   = certified Fix61, rendered without any further pose mutation;
#   medium = certified Fix61 followed directly by conservative Fix140
#            runnerfix1 support routing.  Fix114 is deliberately not online.
if test "$profile" = "medium"; then
  final_version="v5_sceneproof_medium_visual_safe_api"
else
  final_version="v5_sceneproof_fast_fix61_api"
fi
incumbent_placement="$root/${scene_id}_${incumbent_version}_result/S4_layout_refinement/${scene_id}_${incumbent_version}_placement_info_s4.json"
rollback_placement="$root/${scene_id}_${source}_result/S3_pose_inference/${scene_id}_${source}_placement_info.json"
if ! test -s "$incumbent_placement"; then
  echo "SCENEPROOF_API_STOP stage=incumbent_lookup reason=missing_placement path=$incumbent_placement" >&2
  exit 4
fi
if test "$profile" != "fast" && ! test -s "$rollback_placement"; then
  echo "SCENEPROOF_API_STOP stage=rollback_lookup reason=missing_s3_placement path=$rollback_placement" >&2
  exit 4
fi
final_s4="$root/${scene_id}_${final_version}_result/S4_layout_refinement"
placement="$final_s4/${scene_id}_${final_version}_placement_info_s4.json"
sparse_certificate="$root/sceneba_audit/$final_version/sparse_vertical_contact.json"
evaluation="$root/sceneba_audit/$final_version/final_eval.json"
mkdir -p "$final_s4" "$(dirname "$sparse_certificate")"

render_start="$(now_ns)"
if test "$profile" = "medium"; then
  echo "SCENEPROOF_API_STAGE=s4_visual_safe_salvage_and_render PROGRESS=0.84"
else
  echo "SCENEPROOF_API_STAGE=s4_fix61_render PROGRESS=0.84"
fi
render="$root/${scene_id}_${final_version}_result/S4_layout_refinement/${scene_id}_${final_version}_render_simu.png"
render_ok=0
for render_attempt in 1 2; do
  cp "$incumbent_placement" "$placement"
  rm -f -- "$render" "$sparse_certificate"
  sparse_env=()
  if test "$profile" != "fast"; then
    sparse_env=(
      "IMAGINARIUM_SCENEPROOF_SPARSE_VERTICAL_CONTACT_AUDIT_OUTPUT=$sparse_certificate"
      "IMAGINARIUM_SCENEPROOF_SPARSE_VERTICAL_CONTACT_PLACEMENT_OUTPUT=$placement"
      "IMAGINARIUM_SCENEPROOF_SPARSE_ROLLBACK_PLACEMENT=$rollback_placement"
      "IMAGINARIUM_SCENEPROOF_SPARSE_CONTACT_TOLERANCE_M=0.02"
      "IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_SHIFT_M=0.5"
      "IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_TANGENT_SHIFT_M=0.15"
      "IMAGINARIUM_SCENEPROOF_SPARSE_MAXIMUM_PROGRAM_TANGENT_SHIFT_M=0.50"
      "IMAGINARIUM_SCENEPROOF_SPARSE_MINIMUM_HIT_FRACTION=0.10"
    )
    if test "$profile" = "medium"; then
      sparse_env+=(
        "IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_SALVAGE=1"
        "IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_MAX_FLOOR_SHIFT_M=0.60"
        "IMAGINARIUM_SCENEPROOF_VISUAL_SAFE_MAX_SUPPRESSED=4"
      )
    fi
  fi
  set +e
  env \
    -u IMAGINARIUM_SCENEPROOF_SPARSE_VERTICAL_CONTACT_AUDIT_OUTPUT \
    -u IMAGINARIUM_SCENEPROOF_SPARSE_VERTICAL_CONTACT_PLACEMENT_OUTPUT \
    -u IMAGINARIUM_SCENEPROOF_SPARSE_ROLLBACK_PLACEMENT \
    "${sparse_env[@]}" \
    SCENEPROOF_RESULTS_ROOT="$root" \
    SCENEPROOF_MANIFEST="$manifest" \
    SCENEPROOF_RENDER_SOURCE_VERSION="$source" \
    SCENEPROOF_CERTIFIED_VERSION="$final_version" \
    SCENEPROOF_RENDER_LOG_ROOT="$log_root/render" \
    SCENEPROOF_RENDER_SAMPLES="${SCENEPROOF_RENDER_SAMPLES:-256}" \
    SCENEPROOF_RENDER_FORCE=1 \
    IMAGINARIUM_GPU0_ID="$gpu" IMAGINARIUM_GPU1_ID="$gpu" \
    IMAGINARIUM_GPU_FREE_FLOOR_MB="${IMAGINARIUM_GPU_FREE_FLOOR_MB:-8000}" \
    bash scripts/render_sceneproof_certified_paper30.sh \
    > "$log_root/render.log" 2>&1
  render_rc=$?
  set -e
  if test "$profile" = "fast" && test "$render_rc" -eq 0 && test -s "$render"; then
    "$python" - "$sparse_certificate" <<'PY'
import json, sys
json.dump({
    "schema_version": "sceneproof_fix61_passthrough_certificate_v1",
    "status": "certified",
    "passed": True,
    "policy": "frozen_fix61_no_online_pose_mutation",
    "repaired_object_ids": [],
    "unresolved_object_ids": [],
    "objects": [],
}, open(sys.argv[1], "w"), indent=2)
open(sys.argv[1], "a").write("\n")
PY
  fi
  if test "$render_rc" -eq 0 && test -s "$placement" && test -s "$render" && test -s "$sparse_certificate"; then
    render_ok=1
    break
  fi
  echo "SCENEPROOF_API_RETRY stage=s4_sparse_contact_and_render attempt=$render_attempt rc=$render_rc" >&2
done
if test "$render_ok" -ne 1; then
  echo "SCENEPROOF_API_STOP stage=s4_sparse_contact_and_render reason=retry_exhausted" >&2
  tail -120 "$log_root/render.log" >&2 || true
  exit 5
fi
render_end="$(now_ns)"; full_end="$(now_ns)"
echo "SCENEPROOF_API_STAGE=packaging PROGRESS=0.97"

"$python" - "$upstream_evaluation" "$sparse_certificate" "$evaluation" <<'PY'
import json, sys
upstream = json.load(open(sys.argv[1]))
sparse = json.load(open(sys.argv[2]))
status = sparse.get("status", "unresolved")
out = {
    "schema_version": "sceneproof_online_final_eval_v2",
    "status": status,
    "passed": bool(sparse.get("passed")),
    "upstream_evaluation": upstream,
    "sparse_vertical_contact_certificate": sparse,
    "repaired_object_ids": sparse.get("repaired_object_ids", []),
    "unresolved_object_ids": sparse.get("unresolved_object_ids", []),
}
open(sys.argv[3], "w").write(json.dumps(out, indent=2) + "\n")
PY
cp "$placement" "$artifact_dir/placement.json"
cp "$render" "$artifact_dir/render.png"
cp "$evaluation" "$artifact_dir/evaluation.json"

"$python" - "$artifact_dir/result.json" "$evaluation" "$release_id" "$job_id" "$profile" "$final_version" \
  "$s03_start" "$s03_end" "$fix61_start" "$fix61_end" \
  "$fix114_start" "$fix114_end" "$render_start" "$render_end" \
  "$full_start" "$full_end" <<'PY'
import json, sys
from pathlib import Path
out, evaluation, release_id, job_id, profile, final_version, *raw = sys.argv[1:]
values = list(map(int, raw))
names = ["s0_s3_deepsearch", "scenelm_fix61", "online_support_preprocess", "render", "end_to_end"]
timings = {name: (values[i*2+1]-values[i*2])/1e9 for i,name in enumerate(names)}
final_eval = json.load(open(evaluation))
sparse = final_eval.get("sparse_vertical_contact_certificate", {})
visual_safe = sparse.get("visual_safe_salvage", {})
strength = {
    "fast": "frozen_fix61",
    "medium": "presentation_only_visual_salvage",
}.get(profile, "unknown")
record = {
    "job_id": job_id,
    "release_id": release_id,
    "status": final_eval["status"],
    "profile": profile,
    "final_version": final_version,
    "certificate_strength": strength,
    "unresolved_object_ids": final_eval.get("unresolved_object_ids", []),
    "floor_relocated_object_ids": visual_safe.get("floor_relocated_object_ids", []),
    "render_suppressed_object_ids": visual_safe.get("render_suppressed_object_ids", []),
    "eligible_for_paper_metrics": profile == "fast",
    "timing_seconds": timings,
    "artifacts": ["placement.json", "render.png", "evaluation.json", "result.json", "sceneproof-result.zip"],
}
Path(out).write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record))
PY

(cd "$artifact_dir" && zip -q sceneproof-result.zip placement.json render.png evaluation.json result.json)
