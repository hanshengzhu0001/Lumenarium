#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Lumenarium"
if test -z "${SCENEPROOF_WORKER_TOKEN:-}" && test -s .env.lumenarium; then
  set -a
  # shellcheck disable=SC1091
  source .env.lumenarium
  set +a
fi
: "${SCENEPROOF_WORKER_TOKEN:?export the same ASCII SCENEPROOF_WORKER_TOKEN used by the API and workers}"
mkdir -p logs
pkill -f 'python.*-m sceneproof_api.worker' 2>/dev/null || true
pkill -f 'uvicorn.*sceneproof_api.app:app' 2>/dev/null || true
sleep 2
nohup env SCENEPROOF_WORKER_TOKEN="$SCENEPROOF_WORKER_TOKEN" "$HOME/.venvs/lumenarium-py311/bin/python" \
  -m uvicorn sceneproof_api.app:app --host 0.0.0.0 --port 8080 \
  > "$HOME/Lumenarium/logs/sceneproof_api_server.log" 2>&1 < /dev/null &
server_pid=$!
for _ in $(seq 1 30); do curl --fail --silent http://127.0.0.1:8080/healthz >/dev/null && break; sleep 1; done
# The server is started under nohup, so an import-time failure exists only in
# its log.  Without surfacing it here the sole symptom is a silent curl failure,
# which points the reader at the network instead of at the real cause.
if ! curl --fail --silent http://127.0.0.1:8080/healthz; then
  echo "SCENEPROOF_API_START_FAILED=1 log=$HOME/Lumenarium/logs/sceneproof_api_server.log" >&2
  tail -n 40 "$HOME/Lumenarium/logs/sceneproof_api_server.log" >&2 || true
  exit 1
fi
echo
IFS=',' read -r -a gpu_ids <<< "${SCENEPROOF_API_GPU_IDS:-0,1}"
for gpu in "${gpu_ids[@]}"; do
  test -n "$gpu" || continue
  nohup env SCENEPROOF_WORKER_TOKEN="$SCENEPROOF_WORKER_TOKEN" "$HOME/.venvs/lumenarium-py311/bin/python" \
    -m sceneproof_api.worker --api-url http://127.0.0.1:8080 --gpu "$gpu" --worker-id "$(hostname):gpu${gpu}" \
    > "$HOME/Lumenarium/logs/sceneproof_api_gpu${gpu}.log" 2>&1 < /dev/null &
  printf 'SCENEPROOF_API_GPU%s_PID=%s\n' "$gpu" "$!"
done
printf 'SCENEPROOF_API_SERVER_PID=%s\n' "$server_pid"
echo 'SCENEPROOF_API_URL=https://embedding.lightart.qq.com/'
