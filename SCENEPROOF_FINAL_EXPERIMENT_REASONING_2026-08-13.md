# Lumenarium / SceneProof final experiment record

Date: 2026-08-13 (Asia/Shanghai)

This note is the authoritative record for the final experience article and
paper. It separates measured results, code-defined lineage, supported causal
attribution, open confounders, and runtime-accounting limitations. Do not infer
results from fix numbers or historical PNGs.

## 1. Final system lineage

| Name | Perception / retrieval / pose | S4 optimizer and repair | Intended role |
|---|---|---|---|
| v1 | original pipeline | legacy SA | original baseline |
| v3 cold Fix123 | v1 plus floor verification and stack-aware S3 | stack-aware legacy SA5000 | strongest recovered-object baseline |
| v4-deepsearch | v3 pipeline, except S2 uses Omniverse DeepSearch | stack-aware legacy SA | retrieval upgrade baseline |
| V5-fast Fix121 | frozen v4-deepsearch S0-S3 | SceneLM/Fix61 plus conservative SceneProof visual support repair | final fast system |

Code-defined lineage:

- `run_imaginarium_I2Layout_v3.py` enables floor verification and stack-aware
  S3/S4.
- `run_imaginarium_I2Layout_v4_deepsearch.py` keeps those switches and adds
  `IMAGINARIUM_USE_DEEPSEARCH=1`; the defining v3-to-v4 change is S2 retrieval.
- `run_imaginarium_I2Layout_v5_scenelm.py` keeps v4-deepsearch S0-S3 and
  replaces the S4 backend with relation-conditioned SceneLM.
- Fix61 is the frozen quantitative SceneProof aggregate baseline. Fix114/117
  are conservative post-Fix61 support/visibility repair layers; they do not
  repair arbitrary upstream retrieval or pose failures.

## 2. Paper30 protocol

- 30 scenes.
- GT pose headline uses objects whose visible S1 mask area is at least 8000 px.
- The paper headline must use **Primary** metrics, not the overall line printed
  by the dashboard.
- Rotation: aligned Rotation AUC at 60 degrees.
- Translation: aligned Translation AUC at 0.5 m.
- Physical scores use each version's native frozen geometry and the common
  legacy collision policy unless explicitly labelled relation-conditioned.
- GT is used only for evaluation, never for selection or optimization.

## 3. Measured Primary pose quality

| Version | Primary recovery | Primary parent | Rotation AUC@60 | Translation AUC@0.5 m |
|---|---:|---:|---:|---:|
| v1 | 89.49% | **89.32%** | **48.13%** | **23.73%** |
| v3 cold Fix123 | **91.40%** | 87.80% | 48.11% | 20.36% |
| v4-deepsearch | 88.22% | 80.14% | 31.34% | 12.19% |
| V5-fast Fix121 | 88.22% | 80.14% | 31.38% | 12.14% |

Interpretation:

- v3 has the highest Primary recovery in this run (+1.91 pp over v1).
- v1 remains best on Primary parent and narrowly best on rotation and
  translation.
- V5-fast is pose-noninferior to its actual upstream source, v4-deepsearch:
  rotation differs by +0.035 pp and translation by -0.055 pp. These are tiny
  compared with the v3-to-v4 changes (-16.77 pp rotation, -8.17 pp
  translation).

### Final headline table (quality and approximate speed)

| Version | Primary recovery | Primary parent | Rot. AUC@60 | Trans. AUC@0.5 m | Physical macro | Runtime status |
|---|---:|---:|---:|---:|---:|---|
| v1 | 89.49% | **89.32%** | **48.13%** | **23.73%** | 52.98% | historical cached benchmark |
| v3 cold Fix123 | **91.40%** | 87.80% | 48.11% | 20.36% | unresolved: 41.20% fresh evaluator vs 52.14% stale/dashboard value | ~23.8 min/scene; 11.9 useful GPU-hours; ~5.9 h ideal two-A10 makespan |
| v4-deepsearch | 88.22% | 80.14% | 31.34% | 12.19% | 54.58% | frozen DeepSearch benchmark |
| V5-fast Fix121 | 88.22% | 80.14% | 31.38% | 12.14% | **62.10%** | final fast path; use its clean benchmark runtime |

Final naming: V5-fast is DeepSearch S0--S3 plus SceneLM/Fix61; V5-medium adds
the conservative Fix114 true-mesh repair layer. Cold S0--S3 is 636.949
s/scene, 5.308 GPU-hours, and 2.680 measured two-A10 wall hours. V5-fast is
829.879 s/scene and 6.916 GPU-hours (3.458 ideal balanced two-A10 hours).
V5-medium is 996.212 s/scene and 8.302 GPU-hours (4.151 ideal balanced two-A10
hours). Fix61 S4 is 192.930 s/scene and 3.513x faster than legacy; Fix114 adds
166.333 s/scene, yielding V5-medium S4 at 359.263 s/scene and 1.887x. The
415.570 s two-A10 final-render wall time is reported separately.

The recovered cold-stage accounting is exact: S0 9.687 s/scene (1.52%), S1
443.036 (69.56%), S2 137.451 (21.58%), S3 44.790 (7.03%), and orchestration
overhead 1.986 (0.31%). Their sum is 636.949 s/scene with effectively zero
closure error. Therefore the dominant end-to-end optimization target is S1;
neither SceneLM/Fix61 nor the Fix114 layer is the main full-chain bottleneck.

The v3 runtime is an estimate rather than an uninterrupted-run measurement.
It uses the observed mean of the 27 successful runtime rows to impute the three
missing successful rows:

- measured successful mean: 38,498.558 / 27 = 1,425.873 s per scene
  (23.765 min);
- estimated useful Paper30 compute: 42,776.176 s = 11.882 GPU-hours;
- ideal balanced two-A10 makespan: 21,388.088 s = 5.941 h;
- recorded failed/retry overhead: 7,820.308 s = 2.172 GPU-hours;
- estimated campaign compute including recorded retries: 14.054 GPU-hours.

For prose, round this to **about 24 min/scene, 11.9 useful GPU-hours for
Paper30, or 5.9 h on two balanced A10s**. State separately that the interrupted
engineering campaign consumed at least another 2.17 GPU-hours. Actual elapsed
wall time was longer because of pauses, duplicate work, network failures, and
load imbalance.

## 4. Why v4/V5 pose AUC is lower than v3

### What the evidence establishes

1. The loss is already present in v4-deepsearch, before SceneLM/SceneProof.
2. V5-fast and v4-deepsearch use the same frozen upstream S0-S3 and have almost
   identical recovery, parent, rotation, and translation metrics.
3. Therefore SceneLM, Fix61, and Fix114/117 are not responsible for the large
   v3-to-v4 pose-AUC drop.
4. The defining code change from v3 to v4-deepsearch is S2 asset retrieval.
   Retrieved assets can change canonical axes, origin, scale, aspect ratio,
   mesh footprint, and the mapping from image evidence to a Blender pose. GT
   pose AUC is sensitive to all of these even when semantic recovery remains
   good.
5. The simultaneous drop in Primary parent accuracy (87.80% to 80.14%) shows
   that the upstream difference is not rotation alone: retrieval-conditioned
   geometry and support/parent inference also changed.

### What is a supported inference, not yet a strict causal proof

The most likely cause is **retrieval-domain / asset-frame mismatch introduced
by DeepSearch**, amplified by support-parent changes. However, the current v3
cold run and historical v4-deepsearch cache were not produced as a single
paired run with identical frozen S1 observations, identical random seeds, and
only S2 toggled. S1/VLM stochasticity and cold-start differences remain
possible secondary confounders.

In shorter article language: **the dominant observed break occurs when the S2
retriever changes from the legacy DINOv2-based retrieval path to DeepSearch**.
It is reasonable to attribute the bulk of the rotation/translation decline to
that change. It is not reasonable to claim that DINOv2 feature quality itself
was directly ablated, or that DeepSearch is the sole cause, because asset
library coverage, canonical frames, scale conventions, stochastic S1 output,
and parent inference are coupled in the cached comparison.

The paper-safe wording is:

> The pose-AUC reduction appears upstream of SceneLM and coincides with the
> switch to DeepSearch retrieval. The near-identical v4-deepsearch and V5-fast
> pose metrics isolate the SceneLM/SceneProof backend from this loss. A strict
> attribution to retrieval alone would require a paired, seed-locked S2-only
> ablation.

### Minimal ablation needed for definitive attribution

Freeze the same S1 scene graph, masks, camera, and random seed; run traditional
S2 versus DeepSearch S2; feed both through the same S3 and the same S4 backend;
then measure per-object changes in asset ID, canonical-frame rotation, scale,
parent, rotation error, and translation error. This is diagnostic work, not a
requirement for shipping V5-fast.

## 5. Physical quality and a current audit warning

Previously frozen native physical headline values were:

| Version | Physical macro |
|---|---:|
| v1 | 52.98% |
| v4-deepsearch | 54.58% |
| V5-fast Fix121 | **62.10%** |

The new v3 evaluator printed `macro=0.411952`, while the generated dashboard
printed `physical_macro=0.521355` for v3. This is an unresolved provenance
conflict (likely stale alias or differing aggregate field selection). Until the
exact JSON field and file provenance are reconciled, **do not publish a v3
physical macro**. The pose metrics above are unaffected.

The robust conclusion remains: V5-fast materially improves physical
realizability over its v4-deepsearch source while retaining the same pose
quality. Its physical improvement is attributed to SceneLM/SceneProof, not to
an upstream recovery improvement.

## 6. v3 runtime and compute accounting

### Frozen S4-only speed result

The latest same-input Paper30 S4 benchmark is **677.77 s/scene for legacy
SA5000 versus approximately 192.93 s/scene for Fix61**, or **3.513x**.  This is
the authoritative S4 headline and supersedes the earlier intermediate weekly
result of 218.80 s/scene (3.10x).  The separate 384.39 to 186.42 second result
was a one-scene certified-composite smoke and is not the Paper30 headline.
Do not compare the S4-only 192.93 seconds directly with the reconstructed
1,425.9 seconds/scene V3 full-chain number.

The final conservative Fix114 layer has now been measured on all 30 scenes:
**166.333 s/scene** (4,990 useful GPU-seconds).  Consequently the deployed
Fix61+Fix114 S4 path is **359.263 s/scene**, or **1.887x faster than legacy
SA5000**.  Fix61's 3.513x remains the optimizer/backend ablation; 1.887x is the
honest final visual-support system result.  The separate 30-scene 256-sample
render took **415.570 s wall time on two A10s** and is excluded from S4-only.
The retry parser's zero failed-overhead field is invalid because four first
attempts are known to have failed; report that overhead as incompletely
recovered rather than zero.

Observed accounting at completion:

- successful runtime rows: 27 scenes;
- successful recorded GPU time: 38,498.558 s = 10.694 GPU-hours;
- recorded failed/retry time: 7,820.308 s = 2.172 GPU-hours;
- missing successful runtime rows: `bedroom_09`, `diningroom_09`, and
  `supermarket_02`;
- `supermarket_02` accumulated 6,777.062 s of recorded failed attempts, then
  succeeded by reusing S1-S3 and running the corrected S4 separately.

Consequences:

- The campaign proves 30/30 output coverage and measures quality.
- It does **not** provide an exact uninterrupted two-A10 wall-clock benchmark.
- Do not fill the three missing successful times with file ctime/mtime or use a
  failed attempt as a successful end-to-end time.
- Report (a) measured successful GPU-hours for the 27 fully timed scenes,
  (b) failed/retry campaign overhead, and (c) v3 runtime as incomplete or a
  bounded estimate until stage logs reconstruct the remaining useful work.
- A publication-grade v3 speed number requires either exact stage-log recovery
  or one clean timed rerun of the three missing scenes.

### Reportable approximate v3 speed

When an approximate comparison is acceptable, use the transparent mean
imputation above. It is adequate for the order-of-magnitude conclusion that
legacy SA5000 is much slower than the fast SceneLM backend, but not for claims
about small percentage differences. Recommended caption:

> v3 runtime is reconstructed from 27/30 successful per-scene runtime records;
> the remaining three are mean-imputed. The estimate is 11.9 useful GPU-hours
> for Paper30 (5.9 h ideal two-A10 makespan), excluding 2.17 recorded GPU-hours
> of failed retries.

## 7. Robustness fixes made during Fix123

These do not define a new algorithmic v3 variant. They are fail-closed
execution fixes:

- missing structural parent during OBB refinement preserves the original fitted
  OBB;
- missing wall/floor/ceiling in S4 alignment, rotation, or translation preserves
  the incumbent pose;
- transient HTTP stream failures receive bounded retries;
- per-scene atomic claims prevent duplicate dual-GPU work and stale claims are
  reclaimed after owner death;
- a failed scene no longer terminates the entire Paper30 queue.

These changes should be shared by v3 baselines and V5-fast/V5-high tooling, but
must be described as infrastructure robustness rather than quality gains.

## 8. Recommended final positioning

- **V5-fast** is the recommended deployed and main paper system: best measured
  physical macro among the frozen comparison, v4-equivalent pose quality, and
  substantially lower S4 cost than SA5000.
- **v3** is an important accuracy baseline: highest Primary recovery and much
  stronger pose AUC than v4/V5, but slower SA5000 and worse physical collision
  behavior in the newly evaluated run.
- **V5-high** should be a multi-cold-start selector over the same reliable
  pipeline, with selection based only on observable certificates—not GT—and
  should report the additional compute explicitly.
- Fix114/117 remains a qualitative conservative support repair layer. Fix61 is
  the quantitative aggregate anchor unless a later variant passes all local and
  Paper30 aggregate gates.

## 9. Rules for the final article and paper

1. Never mix overall pose metrics with Primary 8000px+ headlines.
2. Never claim that SceneLM caused the v3-to-v4 pose loss.
3. Never claim retrieval-only causality without the paired S2 ablation.
4. Never report the unresolved v3 physical macro conflict as settled.
5. Separate algorithmic runtime from failed retries, duplicated work, network
   interruptions, and engineering-debug overhead.
6. Treat historical beauty renders as qualitative evidence unless their saved
   placement roundtrip is certified.
7. Record Fix61 and the final visual variant together: quantitative baseline
   versus conservative qualitative repair.

## 10. Next execution objective

Run a clean V5-fast Paper30 cold benchmark (Fix124) on two A10s. It must not
reuse result artifacts and must report both end-to-end time with the final
256-sample render and API latency without that render. The runner uses eight
DeepSearch requests in parallel (four per GPU), bounded HTTP retries, adaptive
S3 batch fallback (16 to 8 to 4 to 2 on OOM), per-attempt logs, artifact validation,
and failure isolation before entering SceneLM/Fix61 and the conservative visual
support layer. Quality is evaluated after timing using the 8000px+ Primary
protocol.

Flux fine-tuning is explicitly deferred until the V5-fast/V5-high API path is
packaged and the Fix124 benchmark is frozen. If time remains, it should be a
separate upstream ablation rather than silently changing the final benchmark.
