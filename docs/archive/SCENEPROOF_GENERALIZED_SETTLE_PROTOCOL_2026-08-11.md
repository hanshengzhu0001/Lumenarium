# SceneProof generalized fail-closed settle protocol

Status: implementation authorized on 2026-08-11. Fix61 remains frozen until this protocol produces a measured non-inferior subset.

## Baseline and objective

- Frozen baseline: `v5_sceneproof_collision_partial_commit_certified_paper30_fix61`.
- Objective: repair visually obvious unsupported or COM-unstable objects without full-scene bulk simulation.
- The rule is relation- and geometry-driven. Object class names such as pillow, chair, or lamp are never eligibility conditions.

## Candidate discovery

Read the cached Fix61 true-mesh COM/support audit and select an object only when all conditions hold:

1. Its exact-mesh COM margin is negative, or its declared parent has no valid exact contact witness.
2. A declared support parent exists and is present in the placement.
3. The object is non-structural and is not held by `inside`, wall-attach, ceiling-attach, or another kinematic relation incompatible with gravity.
4. The support relation is not part of an unresolved cycle.
5. The object is not simultaneously used as a dynamic support parent.
6. An isolated Blender rigid-body probe can be constructed.

The discovery report must list every eligible and excluded object with a machine-readable reason. No hand-written scene/object allowlist may decide eligibility.

## Candidate generation

- Simulate one object per process; all other objects are passive colliders.
- Preserve independent full SO(3).
- Horizontal sliding is forbidden. XY motion is accepted only when bounded by the real-mesh rotation chord, `||delta_xy|| <= 2 r_xy sin(delta_theta/2) + 5 mm`; pure drops therefore permit only numerical XY drift, while tipping may use the minimum motion explainable by rotation about contact.
- Rotation audit always uses the shortest SO(3) geodesic in `[0, 180 deg]`; candidates above 90 degrees are rejected as an uncontrolled flip. Vertical motion must remain in `[-0.5 m, +0.005 m]`, so upward launches and excessive falls fail closed.
- Primary profile: 1.0 s, active `CONVEX_HULL`, passive `MESH`, world 10/10, damping 0.8/0.8, friction 100.
- Adaptive retries are diagnostic and bounded: damping 0.5/0.5, then friction 0.5 only if the COM witness remains unstable without falling.
- Restoration failure, new exact collision, timeout, or missing pose is an immediate object-level rejection.

## Uniform exact-mesh gate

The same gate applies to every object class:

1. COM margin improves and the final exact-mesh COM margin is strictly positive.
2. Declared-parent contact exists.
3. Exact-mesh contact gap is within tolerance.
4. No new exact-mesh collision.
5. Exact evaluated-mesh BVH overlap triangle-pair counts do not increase for any object. OBB prism penetration depth and volume remain diagnostic only because they fill real cavities and can falsely report worsening.
6. True-mesh floating/contact gap does not increase.
7. Boundary does not regress.
8. Plane and semantic families do not regress.
9. Rotation and translation recovery do not regress beyond the established tolerance.
10. Incumbent restoration is certified.

An OBB support-score regression may be treated as a proxy disagreement only when term counts are unchanged, attribution proves only the mutated object changed, the reported family delta is exactly reconstructed, and the exact-mesh COM/contact conditions above pass. This witness rule is identical for pillows, chairs, lamps, bowls, and every other eligible object.

## Global subset gate

Passing local gates creates proposals, not commits. Combine proposals using dependency components and re-evaluate the measured Paper30 result.

- Check collision, support, plane, semantic, rotation, and translation independently.
- If any official family regresses, roll back the responsible component and re-evaluate.
- Continue until a measured non-inferior subset is found or every proposal is rejected.
- Macro improvement cannot override a family regression.
- Zero retained changes is a valid fail-closed outcome and must be reported as no improvement.

## Rendering

- Render only scenes containing finally retained objects.
- Use the source-S3 locked camera.
- Produce baseline and final images with identical settings.
- Diagnostic close-ups and rejected candidates must be labelled diagnostic and excluded from formal comparison.

## Current evidence entering this protocol

- Fix84e: three rigid objects achieved positive COM margins, but all passed only relaxed gates and the combined result regressed physical macro by `-0.000085259` and support by `-0.000208791`; therefore Fix84e is not a new baseline.
- Fix106: all five pillow proposals failed strict gates. `pillow_1` is the only plausible witnessed candidate: COM margin `+0.0025345 m`, exact gap `0.0002182 m`, no new collision, but its OBB support term regressed. It remains a proposal, not an accepted result.
- Fix61 remains the official aggregate baseline.

## Implementation order

1. Generate a cached, automatic candidate manifest from the Fix61 true-mesh audit.
2. Reuse existing valid probes; simulate only missing eligible objects.
3. Apply the uniform local gate and write per-object rejection reasons.
4. Build dependency components and search a fail-closed non-inferior subset.
5. Re-evaluate Paper30 and render retained scenes only.

No further category-specific fixes are authorized inside this line.
