#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
test -n "${SCENEPROOF_WORKER_TOKEN:-}" || {
  echo "SCENEPROOF_WORKER_TOKEN is required" >&2; exit 2;
}
test -n "${SCENEPROOF_API_URL:-}" || {
  echo "SCENEPROOF_API_URL is required" >&2; exit 2;
}
gpu="${SCENEPROOF_WORKER_GPU:-0}"
args=(--api-url "$SCENEPROOF_API_URL" --gpu "$gpu")
if test -n "${SCENEPROOF_WORKER_ID:-}"; then
  args+=(--worker-id "$SCENEPROOF_WORKER_ID")
fi
exec "$python" -m sceneproof_api.worker "${args[@]}"
