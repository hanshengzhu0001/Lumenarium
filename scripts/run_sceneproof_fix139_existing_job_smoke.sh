#!/usr/bin/env bash
set -euo pipefail

test "$#" -ge 1 && test "$#" -le 2 || {
  echo "usage: $0 EXISTING_JOB_ID [fast|medium]" >&2
  exit 2
}
old_job="$1"
profile="${2:-medium}"
case "$profile" in fast|medium) ;; *) echo "invalid profile: $profile" >&2; exit 2;; esac
cd "$HOME/Lumenarium"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
input="$($python - "$old_job" <<'PY'
import sqlite3, sys
con = sqlite3.connect("api_state/jobs.sqlite3")
row = con.execute("SELECT input_path FROM jobs WHERE job_id=?", (sys.argv[1],)).fetchone()
if not row:
    raise SystemExit("existing job not found")
print(row[0])
PY
)"
test -s "$input" || { echo "missing input: $input" >&2; exit 2; }
job_id="$($python -c 'import uuid; print(uuid.uuid4().hex)')"
artifact_dir="$HOME/Lumenarium/api_smoke_fix139/$job_id/$profile"
mkdir -p "$artifact_dir"
bash scripts/run_sceneproof_frozen_single_job_fix115.sh \
  "$job_id" "$input" "$artifact_dir" 0 "$profile"
echo "FIX139_SMOKE_JOB_ID=$job_id"
echo "FIX139_RESULT=$artifact_dir/result.json"
echo "FIX139_EVALUATION=$artifact_dir/evaluation.json"
echo "FIX139_RENDER=$artifact_dir/render.png"
