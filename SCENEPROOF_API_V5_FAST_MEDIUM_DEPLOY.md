# SceneProof V5 API deployment

The coordinator exposes a browser UI and REST API. Two GPU workers, one per
A10, claim jobs through SQLite-backed leases. The profiles are:

- `fast`: DeepSearch S0--S3 + SceneLM/Fix61.
- `medium`: V5-fast + SceneProof/Fix114.

Each successful job provides `placement.json`, `render.png`, `evaluation.json`,
`result.json`, and `sceneproof-result.zip`. `placement.json` is the authoritative
scene representation for Blender/UE import; it is not a standalone interactive
viewer. The web UI previews `render.png` and downloads the ZIP.

## Start coordinator

```bash
cd "$HOME/Lumenarium"
PY="$HOME/.venvs/lumenarium-py311/bin/python"
"$PY" -m pip install -r requirements-api.txt
export SCENEPROOF_WORKER_TOKEN='replace-with-a-long-random-token'
nohup bash scripts/run_sceneproof_api_server_fix115.sh \
  > "$HOME/Lumenarium/logs/sceneproof_api_server.log" 2>&1 < /dev/null &
```

## Start one worker per A10

Run both commands with the same token and coordinator URL:

```bash
export SCENEPROOF_WORKER_TOKEN='replace-with-a-long-random-token'
export SCENEPROOF_API_URL='http://127.0.0.1:8080'
nohup bash scripts/run_sceneproof_api_worker_fix115.sh 0 \
  > "$HOME/Lumenarium/logs/sceneproof_api_gpu0.log" 2>&1 < /dev/null &
nohup bash scripts/run_sceneproof_api_worker_fix115.sh 1 \
  > "$HOME/Lumenarium/logs/sceneproof_api_gpu1.log" 2>&1 < /dev/null &
```

Open `http://HOST:8080/` in a browser. Before exposing it outside a trusted
network, put TLS and user authentication in front of the coordinator. Worker
authentication does not authenticate public users.
