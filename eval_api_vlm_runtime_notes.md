# Eval API/VLM Runtime Notes

Last updated: 2026-07-01

## Current problem

The current eval bottleneck is not Blender, GPU, or the `/mnt` symlink move. The slow path is GPT/VLM latency during S1/S2 calls. In recent probes, a single S1 request can take several minutes, and during busy hours some requests stall until timeout.

Observed API behavior:

- Existing stable wrapper uses `/api/v1/gpt/call` async tasks plus polling.
- `/api/v1/gpt/stream` was tested with `gpt-5.4` and `gpt-5.2`.
- During daytime / busy queue, stream tests returned 429 / 500 / 502, including image requests.
- Therefore the daytime-safe path is still `/call` with timeout and fallback.

The phrase decoding fix is already validated. The remaining runtime issue is mainly API queue stability and whether to spend VLM calls on non-core S2 refinements during probes.

## Current daytime policy

Use async `/call`, protect every VLM request with timeout/fallback, and skip non-critical S2 VLM refinements while probing S1/S2/S4 retention.

Recommended environment:

```bash
IMAGINARIUM_GPT_MAX_WAIT=600
IMAGINARIUM_PARALLEL_GPT_TIMEOUT=600
IMAGINARIUM_FLOOR_VERIFY_TIMEOUT=600
IMAGINARIUM_GROUP_ANALYSIS_TIMEOUT=600
IMAGINARIUM_SKIP_S2_VLM=1
```

This keeps S1 from hanging indefinitely and avoids spending 600s on S2 dimension/texture VLM when the immediate question is anonymous-label rate and S2/S4 retention.

## Beijing night policy

Between 20:00 and 10:00 Beijing time, we can try the synchronous streaming endpoint with `gpt-5.4` and keep VLM enabled:

```bash
IMAGINARIUM_SKIP_S2_VLM=0
```

Use this only after a small text+image smoke test confirms `/api/v1/gpt/stream` is healthy in that window. If stream returns 429 / 500 / 502, fall back to `/call` with the 600s timeout policy above.

The intended rule is:

- 20:00-10:00 Beijing time: try `gpt-5.4` stream, do not skip VLM.
- Other times: use `/call` async and skip S2 VLM for targeted probes.

## Current evidence

Clean S1/S2 after phrase decoding is much better than the old frozen full run:

- Old completed4: S1 anonymous 92.2%, S2/S1 7.8%, S4/S1 5.2%, matched 1/114.
- Clean fixdecode completed4: S1 anonymous 12.2%, S2/S1 87.8%, S4/S1 87.8%, matched 56/114.

The current targeted probe is checking whether this holds for low-recovery categories:

```text
laundromat_01
livingroom_02
supermarket_01
streelitter_01
```

Current run:

```text
run_name: targeted_timeout600_v3_resume_skip_s2vlm
output_root: saved_results_targeted_timeout600_v3
log: batch_logs/batch_eval_targeted_timeout600_v3_resume_skip_s2vlm_20260701_091635.log
```

As of the note time, `laundromat_01` has reached S4. `livingroom_02` is in S1 VLM calls. `supermarket_01` and `streelitter_01` have not started yet.

## Next eval steps

1. Let the targeted probe finish or timeout cleanly.
2. Compute anonymous rate and S2/S4 retention for completed scenes.
3. Run calibrated visible-GT metrics only after S4 exists for the completed subset.
4. If low categories still look healthy, move from API/runtime repair back to algorithm diagnostics:
   - v3 parent / stack gating errors.
   - remaining category retrieval gaps.
   - matcher calibration as secondary analysis, not as a substitute for clean S1/S2.

## 2026-07-01 targeted4 calibrated metrics

`targeted_timeout600_v3_resume_skip_s2vlm` finished all 4 v3 scenes:

```text
laundromat_01
livingroom_02
streelitter_01
supermarket_01
```

With visible-GT filtering (`min_visible_mask_area=1024`, `min_visible_bbox_size=8`) and synonym-aware matching:

```text
v3 matched 41/148
object recovery 27.7%
prediction match rate 80.4%
parent accuracy 70.7%
rotation AUC@60 50.8%
translation AUC@0.5m 19.4%
```

The visible-GT denominator was reduced from 194 to 148, so filtering helps the eval denominator, but the main remaining gap is not matcher strictness. For streetlitter and supermarket, every generated object matched GT (`prediction_match_rate=100%`), but S1/S2 produced too few objects relative to visible GT.

S2 attrition is still entirely anonymous-label driven in this subset:

```text
S1 68
S2 51
S4 51
anonymous S1 12
scene graph missing from retrieval 12
anonymous share of missing 100%
```

## 2026-07-01 S2 VLM probe

A single-scene probe reran `supermarket_01` from cached S0/S1 with `IMAGINARIUM_SKIP_S2_VLM=0`.

Outputs:

```text
saved_results_targeted_timeout600_v3_s2vlmprobe
eval_gt_metrics_s2vlmprobe_supermarket_calibrated.json
eval_matching_diagnostics_s2vlmprobe_supermarket_calibrated.json
```

The VLM calls were healthy in this window:

```text
S2 dimension refinement GPT time: 27.6s
S2 texture selection GPT time: 12.3s
```

But S2/S4 recovery did not improve:

```text
baseline supermarket: matched 9/57, pred 9, recovery 15.8%, parent 77.8%
S2-VLM supermarket:  matched 9/57, pred 9, recovery 15.8%, parent 44.4%
```

Reason: S2 VLM refines dimensions/textures; it does not resolve anonymous labels. The same four objects were still skipped by retrieval:

```text
object_1
object_2
object_3
object_4
```

So the next useful fix is not "turn S2 VLM back on for coverage"; it is S1 category/coverage repair and/or an explicit anonymous-object relabeling step before S2 retrieval.

## 2026-07-02 stream endpoint smoke test

Current status:

- `/api/v1/gpt/stream` is reachable, but not usable as the primary eval VLM path yet.
- Minimal stream request with `reasoning` returns HTTP 500:
  - endpoint: `POST /api/v1/gpt/stream`
  - model: `gpt-5.2`
  - body shape: OpenAI Responses-style `input`, `reasoning`, `max_output_tokens`
  - response: `text/plain; charset=utf-8`, `Internal Server Error`
- Minimal stream request without `reasoning` returns HTTP 429:
  - response: `{"detail":"并发已满，请稍后重试"}`
- The same token and host work through `/api/v1/gpt/call`:
  - `/call` returns HTTP 202 with a `task_id`
  - task polling stays `pending` when the queue is saturated.

Interpretation:

- Auth and base routing are OK.
- Queue saturation is real right now.
- The HTTP 500 appears to be a server-side stream handler bug or unsupported
  Responses field path, because a minimal valid request should return either
  SSE events or a structured 4xx/5xx error event, not raw `Internal Server Error`.

What to report to API maintainers:

- `POST /api/v1/gpt/stream` with a minimal Responses API body containing
  `reasoning` returns raw HTTP 500.
- The same endpoint without `reasoning` returns a structured 429, so the stream
  route exists but the Responses-field handling is inconsistent.
- `/api/v1/gpt/call` accepts the same credential and enqueues tasks, so this is
  not a token or DNS issue.
- Please either support the documented `reasoning` field on stream, or reject it
  with a structured SSE/JSON error instead of raw 500.

Local policy until fixed:

- Keep eval on `/call` with timeout/fallback.
- Do not switch production probes to stream until a text+image smoke test returns
  valid SSE `response.output_text.delta` / `response.completed` events.

## 2026-07-02 local slow-tail fallback fix

Implemented local mitigation for slow VLM requests:

- `parallel_processing_requests` now uses a batch-level deadline and collects
  any request that has already completed, regardless of original request order.
- Before this, a slow earlier request could block later completed results and
  cause unnecessary fallback loss.
- Unfinished requests at the deadline are still replaced with conservative
  fallbacks and the worker pool is terminated.

Implemented S1 scene-graph fallback:

- If a region-level scene-graph VLM request times out, objects from that region
  are no longer silently dropped from `scene_graph_result`.
- Missing objects are filled with a conservative graph entry:
  - wall-like objects (`curtain`, `picture`, `mirror`, `sign`, etc.) try wall
    support;
  - other objects default to floor unless local geometry finds a strong support;
  - weak/anonymous supports are rejected.

This should improve dense scenes such as `livingroom_01`, where 4/6 region
requests returned but 2 timed out at 600s.

## 2026-07-02 Gemini `/call` integration

Added a Gemini non-streaming task path for eval VLM calls.

Implementation:

- `utils/llm_api.py` now auto-detects `/api/v1/gemini/call`.
- GPT endpoints keep the OpenAI Responses payload.
- Gemini endpoints use `generateContent` style payloads:
  - `contents[].parts[].text`
  - `contents[].parts[].inlineData` for JPEG image inputs
  - `systemInstruction` for JSON/no-markdown behavior
  - `generationConfig.maxOutputTokens`
- The same polling helper reads `/api/v1/tasks/{task_id}`.
- Result extraction now supports both:
  - GPT `data.output[].content[].text`
  - Gemini `data.candidates[].content.parts[].text`

Config:

```text
config/config_gemini_sync.yaml
endpoint: https://lightaiapi.lightspeed.qq.com/api/v1/gemini/call
model: gemini-3.1-pro-preview
key: same existing LightAI key
```

Smoke tests:

```text
text-only: OK, 6.2s
image+text: demo/livingroom_01_v1.png -> "living room", 12.9s
```

Recommended targeted probe command:

```bash
IMAGINARIUM_WEIGHT_CACHE_DIR=/mnt/kevinzyz/artifacts/Imaginarium-repo/weights_cache \
IMAGINARIUM_GPU_IDS=0 \
IMAGINARIUM_S4_USE_TORCH_VOXELS=1 \
IMAGINARIUM_S4_SKIP_RENDER=1 \
IMAGINARIUM_S1_LOWCAT_PASS=1 \
IMAGINARIUM_S1_RELABEL_ANONYMOUS=1 \
IMAGINARIUM_S3_STACK_AWARE=1 \
IMAGINARIUM_S4_STACK_AWARE=1 \
PYTHONUNBUFFERED=1 \
/ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
  --run-name gemini_sync_parentstack_v3 \
  --output-root saved_results_gemini_sync_parentstack_v3 \
  --config-template config/config_gemini_sync.yaml \
  --scenes eval_sample_parentstack_probe2.txt \
  --v3-only \
  --gpu-count 1 \
  --no-clean \
  --timeout 7200 \
  --gpt-max-wait 600 \
  --gpt-max-retries 1
```

Policy:

- Use Gemini `/call` first for VLM-heavy S1/S2 probes.
- Keep SSE/stream disabled until LightAI stream returns stable image+text events.
- If `/call` queues during peak hours, keep the existing timeout/fallback behavior.
- For Gemini, cap batched VLM requests with:

```bash
IMAGINARIUM_PARALLEL_GPT_PROCESSES=2
```

Reason:

- Text/image smoke tests and the first S1 requests returned quickly.
- A 5-region scene-graph batch returned 1/5 quickly, then the remaining tasks
  waited in the LightAI task queue.
- The current key appears to have low Gemini concurrency, so batched S1/S2 VLM
  should run at 2-way concurrency until the service quota is confirmed higher.

First completed cap-2 probe:

```text
run: gemini_sync_parentstack_cap2_v3
scene: livingroom_01 v3
S4 output: saved_results_gemini_sync_parentstack_cap2_v3/livingroom_01_v3_result/S4_layout_refinement/livingroom_01_v3_placement_info_s4.json
batch log: batch_logs/batch_eval_gemini_sync_parentstack_cap2_20260702_021717.log
elapsed: 897s
```

Runtime behavior:

- Initial S1 image parsing: 14.1s, 17.4s, 37.5s.
- S1 semantic dedup: 24.8s.
- S1 region scene graph, 5 regions at cap-2: 18.8s, 32.5s, 31.2s, 110.1s, 101.8s.
- Floor verification: 14.7s, 24.2s.
- Group/facing analysis: 6.1s, 12.1s.
- Anonymous relabel: 8.4s.
- S2 dimension refinement: 21.4s, 58.7s.
- S2 texture selection: 22.7s.
- No 429 or timeout observed after capping batched VLM to 2 processes.

Calibrated visible-GT diagnostics for this scene:

```text
v3 S1=25, S2=24 (96.0%), S4=23 (92.0%), anon_s1=0.0%
v3 matched=15/27 (55.6%), parent_acc=0.733, rot_auc60=0.354, trans_auc05=0.057
```

Important failure signal:

- Gemini improved S1 coverage and eliminated anonymous labels on this scene.
- Parent/gating still needs a wall-hanging guard:
  - `curtain_0` was reparented to `small_storage_box_0` and then moved 1.4m in S4.
  - `wall_mounted_picture_frame_0` was deleted by floor verification even though it is a valid wall object.
- S2 asset retrieval still has category/visual mistakes:
  - one `pillow` selected an armchair-like asset.

Next fixes after Gemini integration:

- Protect wall-hanging classes (`curtain`, `picture/frame`, `mirror`, `poster`, `sign`) from floor-verification deletion or non-wall reparenting.
- Add/strengthen S2 category guardrails so small soft objects cannot retrieve chair/sofa-scale assets.

## 2026-07-02 Gemini cap-2 checkpoint and next plan

Current recommended eval VLM flow:

```bash
IMAGINARIUM_WEIGHT_CACHE_DIR=/mnt/kevinzyz/artifacts/Imaginarium-repo/weights_cache \
IMAGINARIUM_GPU_IDS=1 \
IMAGINARIUM_PARALLEL_GPT_PROCESSES=2 \
IMAGINARIUM_S4_USE_TORCH_VOXELS=1 \
IMAGINARIUM_S4_SKIP_RENDER=1 \
IMAGINARIUM_S1_LOWCAT_PASS=1 \
IMAGINARIUM_S1_RELABEL_ANONYMOUS=1 \
IMAGINARIUM_S3_STACK_AWARE=1 \
IMAGINARIUM_S4_STACK_AWARE=1 \
PYTHONUNBUFFERED=1 \
/ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
  --run-name gemini_sync_parentstack_cap2_v3 \
  --output-root saved_results_gemini_sync_parentstack_cap2_v3 \
  --config-template config/config_gemini_sync.yaml \
  --scenes eval_sample_parentstack_probe2.txt \
  --v3-only \
  --gpu-count 1 \
  --timeout 7200 \
  --gpt-max-wait 600 \
  --gpt-max-retries 1
```

Observed stage time for `livingroom_01_v3`:

```text
S0 geometry: about 78s
S1 parsing: about 373s
S2 retrieval: about 144s
S3 pose: about 16s
S4 Blender/layout: about 277s
Total batch elapsed: 897s
```

VLM/API timing inside the same run:

```text
S1 initial image parsing: 14.1s, 17.4s, 37.5s
S1 semantic dedup: 24.8s
S1 region scene graph, 5 regions, cap-2: 18.8s, 32.5s, 31.2s, 110.1s, 101.8s
S1 floor verification: 14.7s, 24.2s
S1 group/facing: 6.1s, 12.1s
S1 anonymous relabel: 8.4s
S2 dimension refinement: 21.4s, 58.7s
S2 texture selection: 22.7s
```

Decision:

- Gemini `/call` is usable for full VLM flow.
- Keep batched VLM concurrency at 2 unless LightAI confirms a higher Gemini concurrency quota.
- Do not use stream/SSE as the eval path yet.

Next fix/test order:

1. Parent/wall-hanging guard:
   - Keep wall-hanging objects attached to wall-like parents.
   - Do not delete `wall_mounted_picture_frame`, `curtain`, `mirror`, `poster`, `sign` just because they are not floor-supported.
   - Do not reparent them to floor objects or small support objects such as storage boxes.
2. S2 category guardrails:
   - Prevent small soft objects (`pillow`, `cushion`) from retrieving chair/sofa-scale assets.
   - Use object class, asset retrieval class, and size ratio as hard filters before visual similarity rerank.
3. Rerun targeted Gemini cap-2 probes:
   - `livingroom_01` first, because it exposed both wall-hanging and pillow retrieval issues.
   - Then `workshop_01`, because it tests low-recall industrial/tool categories.
4. Run calibrated visible-GT metrics and matching diagnostics on those probes.
5. Only if both are stable, expand to low-recall targeted set:
   - `supermarket`, `streelitter`, `workshop`, `laundromat`, `livingroom`.

## 2026-07-02 Implemented fixes: S1 wall-hanging guard + S2 soft asset guard

Two code-level fixes implemented and smoke-tested. Both are now in a running probe
(`wallsoft_probe_v3`, livingroom_01 v3 only).

### S1 wall-hanging guard (`modules/_s1_legacy_functions.py`)

Problem root cause:
- `verify_floor_parent_with_vlm_v2` checked `isOnFloor` before `isHangingOnWall`.
- When VLM returned both flags, wall objects (curtain, picture_frame, mirror, poster, sign)
  were sent to floor verification and either deleted or reparented to floor objects
  (e.g. small_storage_box) instead of keeping their wall parent.

Implementation:

- `_S1_WALL_SUPPORTED_HINTS` = {curtain, window, picture, frame, photo, photograph, map,
  sign, billboard, poster, mirror, clock}
- `_is_wall_supported_object(name)`: token-based detection (also handles wall_mounted_ prefix)
- `_mark_wall_supported_object(obj_name, props, parent_path)`:
  sets supported=wall_*, isOnFloor=False, isHangingOnWall=True, isAgainstWall=True
- `_resolve_wall_parent_for_object()`: finds a structural wall parent using
  supported/most_like_wall/againstWall properties

Integration points (3 locations):
1. Scene graph fallback (missing region fill): wall-suggestive objects default to wall parent
2. Floor verification v2 flow: wall-supported objects are protected and re-anchored to wall
   before floor-verification deletion logic runs
3. `verify_floor_parent_with_vlm_v2` entry: wall check happens BEFORE isOnFloor check,
   so dual-flag VLM outputs no longer force wall objects into the floor verification path

Smoke test result:
- curtain_0, wall_mounted_picture_frame_0, mirror_0, poster_0, sign_0: all guarded (forced to wall_0)
- pillow_0, floor_lamp_0: correctly not affected (not in wall hints)

### S2 soft asset guard (`modules/_s2_legacy_functions.py`)

Problem root cause:
- CSV asset annotation was dirty: `a_SM_Armchair.001` had `class_en=pillow`
- Pure class/retrieval_class matching picked armchair-scale assets for pillow/cushion queries
- Visible in livingroom_01 probe: one pillow retrieved an armchair-like asset

Implementation:

- `_S2_SOFT_ITEM_HINTS` = {pillow, cushion}
- `_S2_SOFT_ASSET_NEGATIVE_HINTS` = {armchair, chair, sofa, couch, ottoman, stool, bench, seat}
- `_S2_SOFT_ASSET_POSITIVE_HINTS` = {pillow, cushion}
- `_soft_asset_guard_allows(item_class_name, asset_name, asset_class_name, fbx_size)`:
  - Step 1: skip if item is not soft (pillow/cushion)
  - Step 2: reject if asset name contains furniture tokens (e.g. armchair)
  - Step 3: reject if neither asset name nor asset class contains pillow/cushion token
  - Step 4: reject if max dim > 1.15m or second dim > 0.85m (size guard)
  - env var `IMAGINARIUM_S2_SOFT_ASSET_GUARD=0` to disable for A/B testing
- Called from `_process_and_sort_candidates` during S2 final retrieval

Smoke test result:
- a_SM_Armchair.001 (class_en=pillow, name=armchair): REJECTED (soft_item_reject_furniture_asset_name)
- 0_SM_Bed_Modular_Pillow_3: ALLOWED
- a_SM_pillow_004: ALLOWED
- 0_SM_Sofa_1_ottoman: REJECTED (soft_item_reject_furniture_asset_name)
- A regular chair item: ALLOWED (not_soft_item, guard doesn't apply)

### Current probe: wallsoft_probe_v3

```text
run_name: wallsoft_probe_v3
output_root: saved_results_wallsoft_probe_v3
config: config/config_wallsoft_probe_v3.yaml (template from config_gemini_sync.yaml)
scenes: eval_sample_lowrecall_livingroom_sentinel.txt (livingroom_01 only)
v3-only, GPU 0, Gemini cap=2, timeout=7200, gpt_max_wait=420, gpt_max_retries=2
```

Status: running. The probe will validate whether:
- curtain_0 and wall_mounted_picture_frame_0 are correctly kept on wall_0 (not deleted or reparented)
- pillow items only retrieve pillow-like assets (not armchair)

## 2026-07-02 Next plan (post wallsoft_probe_v3)

### Phase A: Validate wallsoft fixes on livingroom_01

1. Wait for wallsoft_probe_v3 to complete.
2. Compare S1 scene_graph_result.json before/after: check curtain and picture_frame parent fields.
3. Compare S2 retrieval_results_final.json before/after: check pillow asset names and classes.
4. Run calibrated visible-GT metrics:

```bash
python eval_gt_metrics.py \
  --output-root saved_results_wallsoft_probe_v3 \
  --scene livingroom_01 \
  --variant v3 \
  --output eval_gt_metrics_wallsoft_probe_livingroom_calibrated.json
```

5. Compare against baseline (gemini_sync_parentstack_cap2_v3):
   - Parent accuracy (target: improve from 0.733; wall items should no longer be mis-parented)
   - S2/S4 retention (S1→S2→S4 retention should be ≥ baseline)
   - Matched count (if curtain/picture_frame are kept, matched should increase)
   - Object recovery rate

### Phase B: Extend to workshop_01 probe

6. If wallsoft fixes are validated on livingroom_01, run workshop_01:

```bash
IMAGINARIUM_WEIGHT_CACHE_DIR=/mnt/kevinzyz/artifacts/Imaginarium-repo/weights_cache \
IMAGINARIUM_PARALLEL_GPT_PROCESSES=2 \
PYTHONUNBUFFERED=1 \
/ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
  --run-name wallsoft_probe_v3_workshop \
  --config-template config/config_gemini_sync.yaml \
  --scenes eval_sample_workshop_sentinel.txt \
  --output-root saved_results_wallsoft_probe_v3 \
  --v3-only --gpu-count 1 --timeout 7200 \
  --gpt-max-wait 420 --gpt-max-retries 2 --no-clean
```

7. Run calibrated metrics on workshop_01, compare parent acc and matched vs baseline.

### Phase C: Calibrated visible-GT metrics for both probes

8. Compute across both scenes:
   - `parent_accuracy` (primary target for wall guard fix)
   - `S2/S1 retention`, `S4/S1 retention`
   - `matched` count, `object_recovery`
   - `rotation_auc60`, `translation_auc05`

9. Compare all metrics against `gemini_sync_parentstack_cap2_v3` baseline.

### Phase D: Low-recall targeted set expansion

10. If both livingroom_01 and workshop_01 show improvements (or no regression) on parent acc
    and matched, expand to targeted low-recall set:

```text
supermarket_01
streelitter_01
workshop_01  (if not already run)
laundromat_01
livingroom_01 (if needing clean rerun)
```

11. Run full batch on this set:

```bash
IMAGINARIUM_WEIGHT_CACHE_DIR=/mnt/kevinzyz/artifacts/Imaginarium-repo/weights_cache \
IMAGINARIUM_PARALLEL_GPT_PROCESSES=2 \
PYTHONUNBUFFERED=1 \
/ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
  --run-name wallsoft_v3_targeted \
  --config-template config/config_gemini_sync.yaml \
  --scenes eval_sample_lowrecall_targeted_set.txt \
  --output-root saved_results_wallsoft_v3_targeted \
  --v3-only --gpu-count 1 --timeout 7200 \
  --gpt-max-wait 420 --gpt-max-retries 2
```

12. Run calibrated metrics across the full targeted set, compare against old baseline.

### Phase E: v1 vs v3 comparison on parent/stack metrics

13. After v3 targeted set is stable, run v1 on the same targeted set:

```bash
IMAGINARIUM_WEIGHT_CACHE_DIR=/mnt/kevinzyz/artifacts/Imaginarium-repo/weights_cache \
IMAGINARIUM_PARALLEL_GPT_PROCESSES=2 \
PYTHONUNBUFFERED=1 \
/ssd/kevinzyz/miniconda3/envs/imaginarium/bin/python batch_eval.py \
  --run-name wallsoft_v1_targeted \
  --config-template config/config_gemini_sync.yaml \
  --scenes eval_sample_lowrecall_targeted_set.txt \
  --output-root saved_results_wallsoft_v1_targeted \
  --v1-only --gpu-count 1 --timeout 7200 \
  --gpt-max-wait 420 --gpt-max-retries 2
```

14. Compare v1 vs v3 on:
    - Parent accuracy (expect v3 stack-aware to be ≥ v1)
    - S2/S4 retention
    - Object recovery
    - Rotation/translation AUC
    - Stacking fix count and fix coverage (v3-only metrics)

### Decision gate after Phase B

Before expanding to full targeted set or v1, confirm:
- wall-hanging guard works: curtain/picture_frame/mirror/poster/sign are consistently kept on wall_0
- soft asset guard works: pillow/cushion retrieve only pillow-sized assets
- no regression: S1 coverage, S2/S4 retention, parent acc not worse than baseline
- Gemini API stays stable at cap=2 throughout the probe period
