#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Lumenarium"
version="${SCENEPROOF_PLANE_CHART_VERSION:-v5_sceneproof_assignment_compensated_joint_tangent_smoke1_fix52}"
placement_dir="a10_reusable_results/paper30/bedroom_01_${version}_result/S4_layout_refinement"
placement="$(find "$placement_dir" -maxdepth 1 -name '*_placement_info_s4.json' -print -quit)"
test -n "$placement"

"$HOME/.venvs/lumenarium-py311/bin/python" - "$placement" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
audit = data["sceneproof_mesh_visibility_audit"]
print("FILE =", path)
print(
    "MEASURED/ZERO/ISOLATED =",
    audit["objects_measured"],
    len(audit["zero_visible_object_ids"]),
    audit["isolated_diagnostics"],
)
print(
    "CAMERA_DELTA/POSE_DELTA =",
    audit["camera_max_abs_delta"],
    audit["pose_max_abs_delta"],
)
print(
    "PLACEMENT/RENDER/UNOWNED_MESH_GROUPS =",
    audit["objects_with_meshes"],
    audit.get("render_mesh_groups"),
    audit.get("unowned_mesh_groups"),
)
for object_id in audit["plane_sibling_object_ids"]:
    record = audit["objects"].get(object_id, {})
    attribution = record.get("occlusion_attribution", {})
    print(
        "OBJECT=", object_id,
        "VISIBLE=", record.get("rendered_visible_pixels"),
        "ISOLATED=", record.get("isolated_visible_pixels"),
        "CLASS=", record.get("visibility_class", "scene_visible"),
        "IOU=", record.get("iou"),
        "PRECISION=", record.get("precision"),
        "RECALL=", record.get("recall"),
        "OCCLUDER=", attribution.get("dominant_occluder"),
        "OCCLUDER_FRACTION=", attribution.get(
            "dominant_occluder_fraction"
        ),
        "UNKNOWN_FRACTION=", (
            attribution.get("unknown_or_background_pixels", 0)
            / max(attribution.get("isolated_pixels", 0), 1)
            if attribution else None
        ),
    )
patch = audit.get("finite_plane_patch_audit", {})
print("FINITE_PLANE_PATCH:")
for record in patch.get("objects", []):
    if record.get("child_id") not in audit["plane_sibling_object_ids"]:
        continue
    print(
        "OBJECT=", record.get("child_id"),
        "HOST=", record.get("plane_id"),
        "CONTAINED=", record.get("contained"),
        "OUTSIDE_M=", record.get("maximum_outside_distance_m"),
        "MIN_TRANSLATION_UV_M=", record.get("minimum_translation_uv_m"),
        "MIN_TRANSLATION_WORLD_M=", record.get(
            "minimum_translation_world_m"
        ),
        "OCCLUDER=", record.get("dominant_occluder"),
        "CROSS_PLANE=", record.get("cross_plane_occlusion"),
        "REASON=", record.get("reason"),
    )
tangent = audit.get("tangent_candidate_audit", {})
print(
    "TANGENT_AUDIT=",
    "enabled=", tangent.get("enabled"),
    "passing_candidates=", tangent.get("passing_candidates"),
    "mutates=", tangent.get("mutates_placement"),
)
for record in tangent.get("objects", []):
    if record.get("host_plane") is None:
        continue
    print(
        "TANGENT_OBJECT=", record.get("object_id"),
        "HOST=", record.get("host_plane"),
        "OCCLUDER=", record.get("dominant_occluder"),
        "PREDICTED_M=", record.get("predicted_offset_m"),
        "SELECT=", record.get("would_select_offset_m"),
        "REASON=", record.get("reason"),
    )
    for candidate in record.get("candidates", []):
        print(
            "  offset=", candidate["offset_m"],
            "visible=", candidate["visible_pixels"],
            "recall_gain=", candidate["recall_gain"],
            "iou=", candidate["iou"],
            "patch=", candidate["finite_patch_contained"],
            "visual_fail=", candidate["visibility_failures"],
            "physical_fail=", candidate["physical_failures"],
            "passed=", candidate["passed"],
        )
joint = audit.get("joint_tangent_candidate_audit", {})
print(
    "JOINT_TANGENT_AUDIT=",
    "enabled=", joint.get("enabled"),
    "passing_candidates=", joint.get("passing_candidates"),
    "mutates=", joint.get("mutates_placement"),
)
for component in joint.get("components", []):
    print(
        "JOINT_COMPONENT=", component.get("object_ids"),
        "HOST=", component.get("host_plane"),
        "CROSS_PLANE=", component.get("cross_plane_object_ids"),
        "ASSIGNMENT=", component.get("assignment"),
        "WITNESS_OVERRIDES=", component.get(
            "rendered_witness_overrides"
        ),
        "DESIRED=", component.get("desired_offsets_m"),
        "GRID/FEASIBLE/RENDERED=", (
            component.get("grid_candidates_total"),
            component.get("physics_feasible_grid_candidates_total"),
            component.get("rendered_grid_candidates"),
        ),
        "GRID_SEEDS=", component.get("grid_seed_source_by_object"),
        "SELECT=", component.get("would_select"),
        "REASON=", component.get("reason"),
    )
    for candidate in component.get("candidates", []):
        print(
            "  trial=", candidate.get("trial_index"),
            "offsets=", candidate["offsets_m"],
            "union_recall_gain=", candidate["union_recall_gain"],
            "union_iou=", candidate["union_iou"],
            "recovered=", candidate.get("recovered_object_ids"),
            "unresolved=", candidate.get("unresolved_object_ids"),
            "visible_member_fail=", candidate.get(
                "visible_member_failures"
            ),
            "assigned=", candidate["assigned_metrics"],
            "visual_fail=", candidate["visibility_failures"],
            "physical_fail=", candidate["physical_failures"],
            "patch_fail=", candidate["finite_patch_failures"],
            "passed=", candidate["passed"],
        )
if audit["mutates_placement"]:
    raise SystemExit("visibility audit unexpectedly mutates placement")
if audit["camera_max_abs_delta"] > 1e-6:
    raise SystemExit("visibility audit changed the camera")
if audit["pose_max_abs_delta"] > 1e-6:
    raise SystemExit("visibility audit changed object poses")

visible_side = audit.get("visible_side_candidate_audit", {})
print(
    "VISIBLE_SIDE =",
    "enabled=", visible_side.get("enabled"),
    "passing_candidates=", visible_side.get("passing_candidates"),
    "mutates=", visible_side.get("mutates_placement"),
)
for record in visible_side.get("objects", []):
    print(
        "VISIBLE_SIDE_OBJECT=", record.get("object_id"),
        "PLANE=", record.get("plane_id"),
        "Q01=", record.get("signed_distance_q01_m"),
        "MIN_SHIFT=", record.get("minimum_visible_side_shift_m"),
        "SELECT=", record.get("would_select_shift_m"),
        "REASON=", record.get("reason"),
    )
    for candidate in record.get("candidates", []):
        print(
            "  shift=", candidate["shift_m"],
            "visible=", candidate["visible_pixels"],
            "recall_gain=", candidate["recall_gain"],
            "iou=", candidate["iou"],
            "exact_gap=", candidate["exact_attachment_gap_m"],
            "visual_fail=", candidate["visibility_failures"],
            "physical_fail=", candidate["physical_failures"],
            "passed=", candidate["passed"],
        )
PY
