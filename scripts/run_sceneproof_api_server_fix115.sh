#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
python="${IMAGINARIUM_PYTHON:-$HOME/.venvs/lumenarium-py311/bin/python}"
test -n "${SCENEPROOF_WORKER_TOKEN:-}" || {
  echo "SCENEPROOF_WORKER_TOKEN is required" >&2; exit 2;
}
export SCENEPROOF_API_STATE_ROOT="${SCENEPROOF_API_STATE_ROOT:-$HOME/Lumenarium/api_state}"
exec "$python" -m uvicorn sceneproof_api.app:app \
  --host "${SCENEPROOF_API_HOST:-0.0.0.0}" \
  --port "${SCENEPROOF_API_PORT:-8080}" \
  --workers 1
