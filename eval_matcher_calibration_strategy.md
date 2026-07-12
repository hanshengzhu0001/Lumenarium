# Matcher Calibration Strategy

Date: 2026-07-01

## Current Position

We are not claiming that the matcher can now discover every synonym automatically.
The current calibrated matcher is an evaluation calibration layer, built from
observed high-frequency false negatives in GT diagnostics.

The goal is to avoid under-counting object recovery when S1/S2 produces a
semantically correct but differently named category, for example:

- `small_speaker` / `speaker`
- `desk_bookshelf` / `storage_rack` / `storage_shelf`
- `lcd_tv` / `vintage_television_set` / `computer_monitor`
- `large_wooden_crate` / `cardboard_box` / `military_box`
- `large_potted_plant` / `potted_plant` / `flower_pot`

This is not a generation change. It only affects GT evaluation matching.

This means calibrated matching should be used to measure the current frozen
outputs and diagnose failure modes. It is not the mechanism we should rely on
when GT is unavailable.

## No-GT Path After Calibration

When GT is not available, synonym discovery must move upstream into detection,
category policy, and retrieval. The intended path is:

1. Build a canonical asset taxonomy from `class_en`, `retrieval_class_en`,
   asset names, and optional asset captions.
2. Normalize S1 labels into this taxonomy before S2. This handles easy aliases
   such as plural forms, suffixes, and known naming variants.
3. For uncertain labels, use the detected object crop/mask as evidence:
   - crop the object region from the input image,
   - compare it against pre-rendered multi-view asset embeddings,
   - use DINOv2/CLIP local and global similarity,
   - include size/aspect priors,
   - keep top-k candidates rather than forcing a weak top-1.
4. If text label and visual retrieval disagree, mark the object as uncertain
   and keep it visible in diagnostics instead of silently canonicalizing it.
5. Use GT calibration results only to tune thresholds and audit common aliases,
   not as a dependency for future no-GT runs.

In short: GT matcher calibration tells us what the right taxonomy/retrieval
behavior should look like. Future no-GT runs should find near-synonyms through
S1 detection plus crop-to-asset retrieval, not through GT labels.

## How We Found The Synonyms

The loop is diagnostic, not manual guessing:

1. Run visible-GT calibrated metrics.
2. Run `eval_matching_diagnostics.py`.
3. Compare:
   - unmatched visible GT categories
   - unmatched S4 predicted categories
   - matched GT categories
   - S1/S2/S4 retention
4. Add a canonical group only when a repeated false-negative pattern is visible.
5. Re-run metrics on the same saved results to separate matcher effects from generation effects.

Example from `supermarket_01`:

- Before calibrated synonym expansion: `matched=9/57`.
- After canonical groups such as speaker/shelf/tv/box/tank/radio: `matched=17/57`.
- After S1 relabel + lowcat policy probe: `matched=19/57`, with `S2/S1=95.5%`, `S4/S1=90.9%`, anonymous rate `0`.

This told us two things:

- The matcher had been under-counting many correct predictions because GT and S1/S2 used different category names.
- After calibration, the remaining loss is mostly true S1 coverage misses, not S2/S4 attrition.

## What The Matcher Should Not Do

The matcher should not become an unlimited fuzzy matcher.

Risky examples:

- Matching all furniture to all furniture.
- Matching every container to every box when the scene has multiple distinct containers.
- Matching by weak token overlap only.
- Letting generic groups hide real S1 category mistakes.

The current guardrail is:

- Exact asset match has highest priority.
- Specific canonical groups score higher than generic groups.
- Generic groups are lower-confidence and should be monitored.
- Diagnostics must still report unmatched categories so calibration does not hide coverage gaps.

## How To Scale Beyond Small Manual Groups

For a larger asset library, the strategy should become semi-automatic:

1. Build canonical class ids from the asset CSV / taxonomy.
2. Seed aliases from:
   - `class_en`
   - `retrieval_class_en`
   - asset names
   - GT `class_en` / `caption_en`
3. Propose synonym candidates using text embeddings or an LLM, but do not accept them blindly.
4. Validate proposed aliases by running diagnostics on held-out scenes.
5. Keep an audit file of accepted aliases with examples.

For asset identity, category matching alone is not enough. The stronger method is:

- crop the target object from the generated/input image,
- compare it with pre-rendered multi-view asset embeddings,
- add size/aspect priors,
- optionally rerank top-k by local DINOv2/CLIP patch similarity.

That is asset retrieval/evaluation calibration, not just category synonym matching.

## Do We Need Larger-Sample Testing?

Yes. One supermarket scene is enough to prove the failure mode and validate the
fix direction, but not enough to freeze the policy.

Recommended next test:

1. Run a small targeted probe over low-recall categories:
   - `supermarket`
   - `streelitter`
   - `workshop`
   - `laundromat`
   - `livingroom`
2. Use 1-2 scenes per category first.
3. Track:
   - S1 object count
   - anonymous rate
   - S2/S1 retention
   - S4/S1 retention
   - calibrated matched/visible-GT
   - top unmatched GT categories
   - top unmatched S4 prediction categories
4. Only if trends are stable, expand to stratified20/stratified30.

Current remaining S1 coverage bottlenecks from `supermarket_01`:

- `billboard`
- `street_railing`
- `discarded_industrial_component`
- `cardboard_box`
- `fuel_tank`

The lowcat pass has now been split into narrower prompt groups so that
`street_railing` does not compete with `storage_rack/storage_shelf`, and
`sign/billboard` does not compete with vending/beverage machines.

## Decision Rule

Proceed in this order:

1. Confirm the split lowcat prompt improves `street_railing` / `billboard` recall on small probes.
2. Keep calibrated matcher enabled for reporting, but continue reporting raw unmatched categories.
3. Do not run full 302 again until a targeted low-category sample shows stable S1 coverage gains.
4. After lowcat stabilizes, run stratified20 or stratified30 v3-only, then add v1 comparison if needed.

## 2026-07-01 Targeted Probe Notes

Streaming endpoint status:

- `/api/v1/gpt/stream` was smoke-tested with `gpt-5.2` and returned HTTP 500.
- The probe therefore used the existing `/call` async endpoint with 600s wait/retry limits.
- This is acceptable for probing, but slow VLM tail latency remains a runtime risk.

Completed probe results:

- `laundromat_01` from `saved_results_lowrecall_splitprobe5_v3`
  - S1=32, S2=32, S4=32, anonymous=0.
  - calibrated visible-GT matched 23/43 = 53.5%.
  - parent acc 0.478, rot AUC@60 0.373, trans AUC@0.5 0.022.
- `streelitter_01` + `supermarket_01` from `saved_results_lowrecall_splitprobe_core3_v3`
  - S1=73, S2=70, S4=70, anonymous=3.
  - S2/S1 = 95.9%, S4/S1 = 95.9%.
  - calibrated visible-GT matched 22/80 = 27.5%.
  - parent acc 0.636, rot AUC@60 0.098, trans AUC@0.5 0.098.

Lowcat evidence:

- `streelitter_01` lowcat added `fuel_tank`, `portable_generator`, and `discarded_industrial_component`.
- `supermarket_01` split `street_railing` prompt accepted `street_railing_0`, and S2 retrieved a railing asset.
- The split prompt improves low-recall category coverage, but can over-detect small `storage_rack` / structure-like fragments.

Failures / fixes discovered:

- `livingroom_02` can hang after initial S1 detection, likely around long-prompt DINO/detection handling. It should be isolated and retried after adding detection timeouts or prompt-size safeguards.
- `workshop_01` failed because semantic dedup GPT timed out, produced an empty DINO prompt, and `visualize()` crashed with `mask_image` unassigned. Fix by falling back to the pre-dedup label set when dedup times out or returns empty.
- Initial S1 worker all-empty handling was patched so all-None GPT detection no longer silently continues into bad state.

Current interpretation:

- The main phrase-decoding/S1-to-S2 retention issue is fixed for these probes.
- The next high-value fix is not another full rerun; it is parent/stack gating plus robust fallback for semantic dedup.
- Parent/gating evidence:
  - `laundromat_01` reparented several floor-supported washers/storage items to carpet/table/file cabinet.
  - `streelitter_01` reparented cardboard boxes to `portable_generator_0`.
  - These errors directly explain weak parent and translation metrics even when object recovery is acceptable.

Additional completed probe:

- `workshop_01` from `saved_results_lowrecall_workshop_retry_v3`
  - The semantic-dedup fallback patch fixed the prior empty-prompt failure.
  - S1=28, scene-graph=24, S2=24, S4=24, anonymous=0.
  - S2/S1 = 85.7%, S4/S1 = 85.7%.
  - calibrated visible-GT matched 17/43 = 39.5%.
  - parent acc 0.765, rot AUC@60 0.396, trans AUC@0.5 0.069.
  - Remaining unmatched GT is mostly fine-grained workshop clutter:
    `map`, `industrial_oil_can`, `toolbox`, `folding_ladder`,
    `hand_truck`, `welding_gas_cylinder`, `hardware_tool`, and
    `industrial_bench_drill`.
- `livingroom_01` from `saved_results_lowrecall_livingroom_sentinel_v3`
  - Completed S4 successfully, but took 1581s because VLM slow-tail dominated.
  - Initial S1 VLM and semantic dedup succeeded; GroundingDINO was not the
    blocker.
  - Region scene-graph generation had 6 requests; 4 returned in 27-89s and
    2 timed out at 600s, producing partial scene graph coverage.
  - S1=23, scene-graph=16, S2=16, S4=16, anonymous=0.
  - S2/S1 = 69.6%, S4/S1 = 69.6%.
  - calibrated visible-GT matched 11/27 = 40.7%.
  - parent acc 0.727, rot AUC@60 0.173, trans AUC@0.5 0.045.
  - Floor verification reparented `storage_rack_0` to `object_0_0`; S4 later
    treated `storage_rack_0` as floating and moved it 1.6m. This is more
    parent/stack gating evidence.

## Calibration vs Future No-GT Retrieval

The current synonym/category matcher is only an evaluation calibration layer.
It answers: "Given GT labels, are we unfairly counting semantically equivalent
predictions as wrong?" It should not be treated as the production mechanism for
finding objects when GT is absent.

For future larger-scale runs without GT, the intended path is:

1. Detect or segment candidate objects in the input/rendered image.
2. Normalize each detection into a canonical category using:
   - detector label,
   - VLM phrase,
   - local context,
   - the asset taxonomy.
3. Retrieve assets within that canonical category or a small compatible group.
4. Rank candidates by visual fit:
   - crop-to-multi-view DINOv2/CLIP similarity,
   - bbox aspect and physical size prior,
   - view consistency across rendered asset views,
   - optional category-confidence prior.
5. Keep top-k candidates when ambiguous, and let S3/S4 or a reranker choose.

So "finding every synonym" is not the end goal. The synonym map should remain
small, audited, and diagnostic-focused. The scalable mechanism is canonical
taxonomy plus visual retrieval/reranking.

Low-recall targeted probe status:

- Completed full S4 probes: `laundromat_01`, `streelitter_01`,
  `supermarket_01`, `workshop_01`, `livingroom_01`.
- `livingroom_02` remains a known pathological scene, but `livingroom_01`
  proves the pipeline can complete with the current fallback settings.
- Before expanding to stratified20/30, the two highest-value fixes are:
  - parent/stack gating: avoid reparenting floor-supported storage/boxes/racks
    to weak or anonymous supports and avoid large S4 movement caused by those
    edges;
  - VLM slow-tail fallback: dense scenes should not lose entire scene-graph
    regions when 1-2 region requests time out.

## 2026-07-02 Parent / Stack Gating Fix

Implemented conservative gating for S1 floor verification and S3 stack-aware
placement:

- S1 floor verification no longer reparents floor-supported objects to weak or
  anonymous supports such as `object_0_0`, pillows, books, cables, loose
  components, or other ambiguous small objects.
- Floor-anchor categories are protected from VLM overcorrection:
  `rack`, `shelf`, `cabinet`, `table`, `desk`, `bench`, `chair`, `sofa`,
  `machine`, `plant`, `lamp`, `tank`, `generator`, `bicycle`, `tire`,
  `carpet/rug`, etc.
- Geometric reparent now requires stronger 2D evidence:
  - higher horizontal overlap threshold;
  - candidate parent must be below the object;
  - candidate parent must be a strong support-like category;
  - vertical image gap must be close enough.
- S3 stack-aware gate now treats `object_0_0`-style names as anonymous and
  rejects stack pairs involving them.
- S3 stack-aware gate also rejects oversized upper objects by max dimension and
  volume, so large anchors such as storage racks are not treated as movable
  cargo.

Expected effect:

- Prevent cases like `storage_rack_0 -> object_0_0` in `livingroom_01`.
- Prevent floor-supported boxes/racks/storage items from being moved far by S4
  due to a weak parent edge.
- Keep legitimate cargo-like stacking (`box`/`crate`/`toolbox` on shelf/pallet)
  available, but only when category and geometry both agree.

Recommended validation:

1. Rerun a tiny `livingroom_01 + workshop_01` v3-only probe with cached/no-clean
   where possible.
2. Compare:
   - scene-graph missing-from-retrieval count;
   - S2/S4 retention;
   - parent accuracy;
   - S4 movement distances for racks/storage/boxes.
3. Only after this improves or stays neutral, run stratified20/30.
