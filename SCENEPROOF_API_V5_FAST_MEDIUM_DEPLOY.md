# Lumenarium API deployment

This is the operator quick reference. The canonical zero-to-running guide is
[`README.md`](README.md#start-from-a-clean-machine); do not maintain a second,
divergent install procedure here.

## Profiles

| Profile | Production behavior | Paper metrics |
|---|---|---|
| `fast` | frozen SceneLM/Fix61 | eligible |
| `medium` | Fix61 plus bounded visual-safe cleanup | presentation only |
| `best` | one V5-fast run plus exhaustive true-surface support audit and transactional first-contact repair | validation required before paper reporting |

## Start or restart

The private `.env.lumenarium` must already contain the Gemini-compatible visual
API configuration, DeepSearch URL and a self-generated ASCII worker token.

```bash
cd "$HOME/Lumenarium"
bash scripts/bootstrap_lumenarium.sh verify
bash scripts/bootstrap_lumenarium.sh start
curl -s http://127.0.0.1:8080/healthz
```

Monitor the coordinator and one worker per selected GPU:

```bash
tail -F \
  "$HOME/Lumenarium/logs/sceneproof_api_server.log" \
  "$HOME/Lumenarium/logs/sceneproof_api_gpu0.log" \
  "$HOME/Lumenarium/logs/sceneproof_api_gpu1.log"
```

The public UI is deployed at
[https://embedding.lightart.qq.com/](https://embedding.lightart.qq.com/).
TLS and public-user access control are provided by the upstream gateway; the
worker token authenticates workers only and must never be exposed to browsers.

## Outputs

Every completed job packages `placement.json`, `geometry.json`, `render.png`,
`evaluation.json`, `result.json` and `sceneproof-result.zip`. New inputs execute
S0--S4. Byte-identical inputs may reuse the frozen S0--S3/Fix61 cache; the UI's
cold-rerun control explicitly bypasses it. `best` is a single run: it does not
spawn three child cold starts or perform GT-free trial selection.

## Client usage

The public endpoint is `https://embedding.lightart.qq.com/`; locally the same
API serves on `http://127.0.0.1:8080`. Three calls cover a complete job.

**1. Submit** (`POST /v1/jobs`, multipart form; PNG/JPEG up to 25 MiB):

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/jobs \
  -F "image=@/path/to/input.png" \
  -F "profile=fast" \
  -F "force_cold_rerun=false" \
  -H "Idempotency-Key: my-client-key-1"
```

`profile` is one of `fast | medium | best | demo`. Byte-identical inputs with
the same key hit the frozen S0--S3/Fix61 cache; set `force_cold_rerun=true` to
bypass it. The response carries `job.job_id`.

**2. Poll** (`GET /v1/jobs/{job_id}`):

```bash
curl -sS http://127.0.0.1:8080/v1/jobs/<job_id>
```

Poll until `state` is `succeeded` (or `failed` / `cancelled`).

**3. Download artifacts** (`GET /v1/jobs/{job_id}/artifacts/{name}`):

```bash
curl -sS -o render.png \
  http://127.0.0.1:8080/v1/jobs/<job_id>/artifacts/render.png
curl -sS -o result.zip \
  http://127.0.0.1:8080/v1/jobs/<job_id>/artifacts/sceneproof-result.zip
```

Every completed job packages `placement.json`, `geometry.json`, `render.png`,
`evaluation.json`, `result.json`, and `sceneproof-result.zip`. A minimal
polling client is embedded in the browser UI (`sceneproof_api/app.py`), and a
cancel endpoint is `POST /v1/jobs/{job_id}/cancel`. Public deployment must
never expose the worker token (`X-Worker-Id` headers on `/internal/*`); the
upstream gateway handles TLS and public access control.
