# SceneProof current status — 2026-08-06

This is the authoritative execution note.  Do not infer the current stage
from the largest fix number alone.

## Main objective

The active research line remains executable `SUPPORT` programs with true-mesh
centre-of-mass (COM) stability:

1. measure the true-mesh/filled-voxel COM;
2. construct a witnessed support region from real mesh contact;
3. classify stable, marginal, unstable, or ABSTAIN;
4. apply a scoped COM correction or local gravity settle only when witnessed;
5. reject/rollback on any collision, support, plane, boundary, rotation, or
   translation regression.

We have completed responsibility measurement and scoped rollback.  A promoted
COM projection/settle operator has **not** yet been run on Paper30.

## Milestones and their roles

| Node | Role | Status |
|---|---|---|
| Fix27 | Locked source-S3 camera and float32-safe render comparison | infrastructure baseline; not an optimizer result |
| Fix43 | Visual rollback Smoke1; its in-process render exposed settled pillows | useful visual witness, but the settled pose was not fully serialized |
| Fix61 | Paper30 collision partial-commit aggregate baseline | frozen aggregate baseline: 30/30, about 3.513x SA5000, physical macro +0.004064; support still -0.001977 vs Smooth |
| Fix62–67 | Cached true-mesh COM responsibility, voxel COM/contact, counterfactual oracle | measurement and localization complete |
| Fix68 | Paper30 scoped COM rollback | 6/30 scenes rolled back, support +0.000496 and macro +0.000144 vs Fix61; no direct COM projection yet |
| Fix70–72 | Support identity and rendered pose-ownership audits | saved JSON ownership checked; did not see unsaved post-simulation transforms |
| Fix73–75 | Forced render and pillow pose lineage | proved old beauty PNGs and adjacent saved poses diverged; first persisted guarded pillow change occurs at Fix55 |
| Fix76 | Serialize every placement-owned Blender root after physics | active repair branch; captures settled `pillow_1` and `pillow_2` which previously lost rigid-body ownership before serialization |
| Fix77 | Re-evaluate Fix76 with separate pose and stochastic-render gates | current validation step; no Blender rerun required |
| Fix78 | Cached Fix76 true-mesh COM action routing | next audit-only Smoke1; separates witnessed COM projection from missing-contact gravity-settle probes |
| Fix79 | Factor intrinsic child tipping versus finite-parent overhang | required correction before mutation; translation cannot repair intrinsic COM/contact instability |
| Fix80 | Process-isolated local full-SO(3) gravity-settle oracle | current next run; four Fix79 witnesses, audit-only, no pose commit |
| Fix81 | Dtype-aware Fix80 restoration re-evaluation | cached re-evaluation only; `2.3841858e-7` is two float32 ULP, not state leakage; @1confirmed 2026-08-06**: `single_sofa_chair_1` sole candidate, all 4 incumbents restored, no new collisions |
| Fix82 | Single-object full component gate for the Fix81 winner | **completed 2026-08-07: FAILED**; 12/13 gates pass, `support_noninferior` fails (-0.008). Decision: rollback. |
| Fix83 | Per-object support regression audit | **completed 2026-08-07**: traced the -0.008 to an OBB proxy vs. true-mesh measurement disagreement. Chair rotation caused OBB z_min (proxy) to report 0.084m gap while true-mesh contact remained at 0.003m. |
| Fix84 | Generalised true-mesh witness exemption in component gates | **completed 2026-08-07: PASSED**; `single_sofa_chair_1` passes all gates under the witnessed exemption. 10 conditions satisfied (E0 denominator stability, E1 attribution, E2 amplitude, E3-E6 true-mesh evidence, E7 hard gates). Decision: `render_candidate_before_scoped_commit_with_witnessed_exemption`. First gravity-settle candidate through complete gates. Not yet promoted to pose commit. |

## Fix81 confirmed (2026-08-06)

Fix81 re-evaluated Fix80 probes with dtype-aware float32 restoration tolerance.
All four probes passed incumbent restoration with no new collisions:

- `single_sofa_chair_1`: `locally_promising_requires_full_component_gates` — sole candidate entering Fix82
- `floor_lamp_0`: `visibility_or_support_unresolved` — rejected (settled pose still unstable)
- `pillow_0`: `visibility_or_support_unresolved` — rejected (settled pose still unstable)
- `pillow_3`: `visibility_or_support_unresolved` — ABSTAIN (insufficient contact evidence)
- `ALL_INCUMBENTS_RESTORED=True`

## Fix76 interpretation

The legacy serializer updated only `MESH` objects that still had a rigid body.
Drop simulation bakes ACTIVE transforms and removes those rigid bodies.  The
in-process render therefore showed settled objects, while JSON retained their
pre-simulation poses.  Fix76 changes ownership to
`all_placement_owned_blender_roots` and rejects render-time pose drift.

Fix76 changed three saved pillow matrices relative to the original Fix43 JSON:

- `pillow_1`: translation 0.3621554 m, substantial rotation;
- `pillow_2`: translation 0.471884211 m, substantial rotation;
- `pillow_6`: rotation-only float-level change (~1.46e-7 Frobenius).

The first Fix76 audit reported PSNR 60.03 dB and 0.125% pixels differing by
more than two levels.  That is a near-identical independent Cycles render, but
the original 0.1% raster threshold was too strict.  Fix77 keeps the stronger
object-pose drift guard and treats stochastic beauty-render parity separately.

## Next required sequence

1. ✅ Fix77 passed all serialization, pose-drift, and beauty-roundtrip gates.
2. ✅ Visual inspection of Fix76 in-process and round-trip images confirmed.
3. ✅ Fix78 cached true-mesh COM responsibility on Fix76 serialized state — completed.
4. ✅ Fix79 factor intrinsic tipping vs. parent-surface overhang — completed; no translation-only COM projection candidate found.
5. ✅ Fix80 process-isolated local full-SO(3) gravity-settle oracle — completed.
6. ✅ Fix81 dtype-aware restoration re-evaluation — completed; `single_sofa_chair_1` sole candidate.
7. ✅ Fix82 full component gates for `single_sofa_chair_1` — completed but FAILED.
8. ✅ Fix83 per-object support regression audit — traced to OBB proxy vs. true-mesh disagreement.
9. ✅ **Fix84 generalised true-mesh witness exemption — `single_sofa_chair_1` PASSES all gates.** COM margin -0.061→+0.244 (stable), true-mesh contact 0.0043→0.0029 (improved), 24/25 objects unchanged. Decision: `render_candidate_before_scoped_commit_with_witnessed_exemption`.
10. 🔲 **Decision point**: `single_sofa_chair_1` has a gate-passing gravity-settle candidate on Smoke1. Options: (a) render the candidate for visual verification, (b) promote to a scoped pose commit on `bedroom_01`, (c) expand to Smoke5 with the exemption enabled, (d) promote only after the exemption is formally justified in the paper/technical report.

Fix61 remains the aggregate reporting baseline.

## Fix82 detailed result (2026-08-07)

```
OBJECT=single_sofa_chair_1
PASSED=False
ROT_DELTA=+0.000000000  TRANS_DELTA=+0.000000000
SUPPORT_DELTA=-0.008000010284022752
COLLISION_DELTA=0.0  PLANE_DELTA=0.0

Gates (12/13 pass, 1 fail):
  ✅ collision_noninferior
  ✅ plane_noninferior
  ✅ rotation_noninferior
  ✅ translation_noninferior
  ✅ no_new_exact_mesh_collision
  ✅ true_mesh_boundary_evaluable
  ✅ true_mesh_boundary_noninferior
  ✅ true_mesh_support_stable          ← chair is locally stable!
  ✅ incumbent_restoration_certified
  ✅ no_evaluator_failures
  ✅ semantic_unchanged_or_nonevaluable
  ❌ support_noninferior               ← scene-level support: 0.5759→0.5679

Physical scores:
  fix76  support=0.5759  macro=0.4552
  fix82  support=0.5679  macro=0.4532  delta: macro=-0.0027

DECISION=rollback_object_to_fix76
```
