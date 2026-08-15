# V5-fast and V5-medium quality and speed report

## Headline result

Evaluation uses Paper30 Primary objects with at least 8,000 visible pixels.

| Metric | V5-fast |
|---|---:|
| Primary recovery | 88.22% |
| Primary parent | 80.14% |
| Rotation AUC@60 | 31.38% |
| Translation AUC@0.5 m | 12.14% |
| Physical macro | 62.10% |

Relative to V4-deepsearch, V5-fast preserves recovery and parent accuracy,
changes rotation by approximately +0.04 percentage points and translation by
-0.05 percentage points, and improves physical macro from 54.58% to 62.10%
(+7.52 percentage points).

## Final S4 runtime

| Scope | Mean/scene | Paper30 useful GPU time | Speedup vs legacy S4 |
|---|---:|---:|---:|
| Legacy SA5000 S4 | 677.770 s | 5.648 h | 1.000x |
| SceneLM/Fix61 core | 192.930 s | 1.608 h | 3.513x |
| SceneProof/Fix114 add-on | 166.333 s | 1.386 h | additive |
| V5-fast S4 (Fix61) | 192.930 s | 1.608 h | 3.513x |
| V5-medium S4 (Fix61 + Fix114) | 359.263 s | 2.994 h | 1.887x |

Cold S0--S3 is 636.949 s/scene and 5.308 useful GPU-hours, with measured
two-A10 wall time 9,649.662 s (2.680 h). V5-fast S0--S4 is therefore 829.879
s/scene, 6.916 useful GPU-hours, or 3.458 ideal balanced two-A10 hours.
V5-medium is 996.212 s/scene, 8.302 useful GPU-hours, or 4.151 ideal balanced
two-A10 hours. Ideal-balanced totals are compute normalizations.

| Cold stage | Mean seconds/scene | GPU-hours | Share of S0--S3 |
|---|---:|---:|---:|
| S0 | 9.687 | 0.081 | 1.52% |
| S1 | 443.036 | 3.692 | 69.56% |
| S2 | 137.451 | 1.145 | 21.58% |
| S3 | 44.790 | 0.373 | 7.03% |
| Orchestration overhead | 1.986 | -- | 0.31% |
| **S0--S3 total** | **636.949** | **5.308** | **100.00%** |

The accounting closure error is effectively zero. S1 is the dominant
full-chain bottleneck; reducing SceneLM or Fix114 time alone cannot deliver the
largest end-to-end speedup.

The final 256-sample render is a separate output stage: 415.570 seconds wall
time on two A10s. It is not included in the S4 mean or speedup.

## Full-chain timing status

Fix124 completed cold S0--S3 for 30/30 scenes. Its coordinator later exited
during the Fix114/evaluation phase and did not write the planned combined
benchmark JSON. Therefore no cached supplement time or S4 mean is presented
as a measured cold S0--S4 wall time. The exact S0--S3 total can be recovered
from the 30 successful `runtime_gpu*.jsonl` records without rerunning scenes.

## Reporting recommendation

Use 62.10% and 3.513x for V5-fast/Fix61. Report V5-medium as the additional
Fix114 true-mesh repair operating point with 1.887x final-S4 speedup. Keep its
quality row separate until a corresponding aggregate score is selected; do
not silently reuse the Fix61 quality score.
