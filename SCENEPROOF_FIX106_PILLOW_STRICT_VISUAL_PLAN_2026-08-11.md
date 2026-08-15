# Fix106 pillow strict-visual compromise

## Goal

Improve the obvious pillow-settling appearance in `bedroom_01` without running full-scene or Paper30 bulk simulation and without weakening the formal Fix61 result.

## Frozen baseline

`v5_sceneproof_collision_partial_commit_certified_paper30_fix61`

## Scope

Only the five existing pillow support witnesses in `bedroom_01` are probed:

- `pillow_0`
- `pillow_1`
- `pillow_2`
- `pillow_3`
- `pillow_6`

Each object is simulated in an independent Blender process with full SO(3). The primary profile remains 1.0 second, 10/10 substeps and solver iterations, damping 0.8/0.8, friction 100. Adaptive retries reuse the already audited Fix84e policy.

## Important policy change

An unresolved but safely restored measured probe may now be materialized as a temporary candidate so that it reaches the complete evaluator. This is not an acceptance exemption.

The final candidate uses strict acceptance plus one general witnessed exception: an OBB support-score regression may be ignored only when the exact-mesh certificate is stable or marginal, its COM margin is strictly positive, declared-parent contact exists, the exact gap is within tolerance, and attribution proves only the mutated object changed. Collision, plane, semantic, rotation, translation, boundary, restoration, or true-mesh support failure still causes an object-level rollback to Fix61.

## Rendering protocol

The formal comparison renders only Fix61 and the strict-final candidate with the source-S3 locked camera. If no pillow passes, the final image must equal Fix61 and the experiment is reported as unresolved rather than improved.

## Interpretation

This is a bounded visual-repair experiment, not a new aggregate baseline. It must not replace Fix61 unless the strict final report is non-inferior and at least one pillow pose is retained.
