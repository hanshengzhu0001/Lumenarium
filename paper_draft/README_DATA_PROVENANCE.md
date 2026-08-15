# SceneProof paper draft: data and update notes

This file records what is measured, what is inferred, and what remains a
placeholder in `sceneproof_paper_draft.tex`.

## Frozen measured values

Primary protocol: objects with at least 8,000 visible S1-mask pixels.

| Version | Recovery | Parent | Rotation AUC@60 | Translation AUC@0.5 m | Physical macro |
|---|---:|---:|---:|---:|---:|
| V1 | 89.49% | 89.32% | 48.13% | 23.73% | 52.98% |
| V3 support-aware | 91.40% | 87.80% | 48.11% | 20.36% | withheld |
| V4-deepsearch | 88.22% | 80.14% | 31.34% | 12.19% | 54.58% |
| V5-fast | 88.22% | 80.14% | 31.38% | 12.14% | 62.10% |

V3 physical is withheld because the fresh evaluator printed 41.20% while an
older dashboard reported 52.14%. Do not publish either before provenance is
reconciled.

## V1, V3, and V4 runtime disclosure

- V1 and V3 use the same legacy-SA runtime profile.
- V3 has 27 measured successful rows: 38,498.558 seconds total.
- V1/V3 measured mean: approximately 1,425.873 seconds/scene.
- V1/V3 mean-imputed useful Paper30 compute: 11.882 GPU-hours.
- V1/V3 ideal balanced two-A10 makespan: 5.941 hours.
- Recorded failed/retry overhead: 2.172 GPU-hours, reported separately.
- DeepSearch saves approximately 80 seconds/scene in S2 relative to V1/V3.
- V4-deepsearch is therefore reported approximately as 1,345.9 seconds/scene,
  11.216 useful GPU-hours, or 5.608 ideal balanced two-A10 hours.

The V1/V3 equality and V4 saving are approximate system-level reporting
assumptions. They are suitable for an order-of-magnitude comparison, not a
fine-grained percentage-speed claim.

## V5-fast and V5-medium final report

The final headline quality row is Primary recovery 88.22%, Primary parent
80.14%, rotation AUC@60 31.38%, translation AUC@0.5 m 12.14%, and physical
macro 62.10%. V5-fast is Fix61: 192.930 seconds/scene (1.608 GPU-hours), or
3.513x versus legacy S4. V5-medium adds Fix114 at 166.333 seconds/scene, giving
359.263 seconds/scene (2.994 GPU-hours) and 1.887x versus legacy S4.

Fix124 cold S0--S3 is recovered from all 30 successful worker timing rows:
636.949 seconds/scene, 5.308 GPU-hours, and 2.680 measured two-A10 wall hours.
The exact stage decomposition is S0 9.687, S1 443.036, S2 137.451, S3 44.790,
and overhead 1.986 seconds/scene. The accounting closure error is effectively
zero. S1 represents 69.56% of S0--S3 and is the principal bottleneck.
V5-fast S0--S4 is 829.879 seconds/scene and 6.916 GPU-hours; V5-medium is
996.212 seconds/scene and 8.302 GPU-hours. Their 3.458 and 4.151 ideal
two-A10 hours are compute normalizations, not observed full-chain wall times.

Report both algorithmic useful compute and failed/retry overhead. The clean
benchmark should use adaptive S3 batch fallback 16 -> 8 -> 4 -> 2, eight
DeepSearch requests in parallel, atomic per-scene claims, bounded transient
HTTP retries, structural-parent fail-closed handling, and final artifact
validation.

## Frozen S4-only benchmark

- Legacy SA5000 S4: 677.77 seconds/scene.
- Latest Fix61 S4: approximately 192.93 seconds/scene.
- Speedup: approximately 3.513x.
- This supersedes the older intermediate weekly-report value of 218.80
  seconds/scene (3.10x).
- The 384.39 -> 186.42 second, 2.06x figure was a single-scene certified
  composite smoke and must not be used as the Paper30 headline.
- Fix114 support/visibility repair adds 166.333 seconds/scene (4,990 useful
  GPU-seconds over 30 scenes).
- Final Fix61 + Fix114 S4 is 359.263 seconds/scene, or 1.887x versus legacy.
- The final 30-scene 256-sample render took 415.570 seconds wall time on two
  A10s and is reported outside S4-only time.
- The generated `FAILED_RETRY_OVERHEAD_SECONDS=0` is an accounting miss, not a
  measured zero: four first attempts failed before the tolerance correction,
  and their complete durations were not recovered by the parser. Exclude this
  field or label the retry overhead incomplete.

Do not mix this S4-only result with the approximately 1,425.9 seconds/scene V3
full-chain reconstruction. Fix124 will supply the clean V5-fast full-chain
runtime.

## Claims that must remain qualified

- The large pose-AUC reduction is already present in V4-deepsearch and is not
  caused by SceneLM/SceneProof.
- It coincides with the S2 retriever change and is most plausibly explained by
  asset-frame/retrieval-domain mismatch plus parent changes.
- This is attribution evidence, not strict causality, until a seed-locked,
  frozen-S1 S2-only ablation is run.
- V5-high is planned: three cold starts, selected only by observable
  certificates, with no GT access and explicit extra compute.
- Flux fine-tuning is future upstream work and is not part of V5-fast.

## Attribution

The paper builds on Imaginarium (arXiv:2510.15564). DeepSearch is attributed
to Calvin Gu and the Tencent team. Final author order and individual
contributions remain to be supplied by the project owner.
