# SceneProof frozen full-chain and API objective

The frozen production candidate is DeepSearch S0-S3 followed by SceneLM/Fix61
and the transactional true-mesh SceneProof/Fix114 repair. Fix61 remains the
quantitative anchor; Fix114 is the complete method and final visual output.

## Immediate benchmark

The Paper30 benchmark must run in an isolated results root and report:

- S0-S3 including DeepSearch wall time;
- SceneLM/Fix61 wall time;
- SceneProof/Fix114 wall time;
- final locked-camera render wall time;
- full two-A10 makespan and per-scene JSONL timings;
- final GT and physical metrics, not only stage success.

S2 uses four DeepSearch workers per A10 process, for eight concurrent requests
in total. The benchmark never overwrites the accepted Paper30 cache.

## API phase after benchmark

Package the frozen pipeline behind one job API with immutable release/config
IDs, stage-level progress, idempotent cache keys, per-host leases, resumable
artifacts, and the same certificate/fail-closed semantics as the benchmark.
The two A10 execution slots will consume independent scene jobs; orchestration
must not change algorithm thresholds or camera policy.
