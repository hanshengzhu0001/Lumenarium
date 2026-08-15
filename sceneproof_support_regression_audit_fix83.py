#!/usr/bin/env python3
"""Audit support regression: Fix76 (incumbent) vs Fix82 (candidate).  v2.

Traces the per-object support delta and the support-parent chain to explain
why a locally stable gravity-settle candidate degrades the scene-level
support score.

v2 fixes: inf containment objects were incorrectly excluded because
``math.isfinite(inf)`` is False but ``linear_score(inf, tol) = 0.0`` is the
correct contribution.  Inside-containment objects are now also traced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def csv_float(raw: str | None) -> float | None:
    """Parse a CSV cell into a float or None."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def linear_score(error: float, tolerance: float) -> float:
    """Replicate eval_physical_realizability.linear_score exactly."""
    if not math.isfinite(error):
        return 0.0
    return max(0.0, 1.0 - error / tolerance)


def support_score(
    gap: float | None,
    containment: float | None,
    overlap: float | None,
    inside_error: float | None,
    *,
    contact_tol: float = 0.05,
    containment_tol: float = 0.05,
    overlap_tol: float = 0.9,
) -> float | None:
    """Reconstruct the per-object support score exactly as in
    ``eval_physical_realizability.evaluate_one_scene``."""
    # Inside-containment path (spatial == "inside")
    if inside_error is not None:
        return linear_score(inside_error, containment_tol)
    # Regular support path (gap / containment / overlap)
    # Note: linear_score handles inf containment → 0.0, which is correct.
    if gap is not None and containment is not None and overlap is not None:
        return (
            linear_score(gap, contact_tol)
            + linear_score(containment, containment_tol)
            + min(1.0, overlap / overlap_tol)
        ) / 3.0
    return None


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit support regression between two versions."
    )
    parser.add_argument("--object-csv", type=Path, required=True,
                        help="physical_objects.csv from the Fix82 audit")
    parser.add_argument("--incumbent-placement", type=Path, required=True,
                        help="Fix76 placement_info_s4.json")
    parser.add_argument("--candidate-placement", type=Path, required=True,
                        help="Fix82 candidate placement_info_s4.json")
    parser.add_argument("--probe", type=Path, required=True,
                        help="Fix82 local-settle probe JSON")
    args = parser.parse_args()

    # --- 1. Load per-object data from CSV ---
    rows = load_csv(args.object_csv)
    incumbent_rows = {
        row["object_id"]: row
        for row in rows
        if row["version"] == "v5_sceneproof_pose_serialization_smoke1_fix76"
    }
    candidate_rows = {
        row["object_id"]: row
        for row in rows
        if row["version"] == "v5_sceneproof_local_settle_candidate_smoke1_fix82"
    }

    # --- 2. Load support chains ---
    incumbent = json.loads(args.incumbent_placement.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_placement.read_text(encoding="utf-8"))

    def support_parent(placement: dict, oid: str) -> str | None:
        supported = placement.get("obj_info", {}).get(oid, {}).get("supported")
        if not supported:
            return None
        if isinstance(supported, list):
            return str(supported[0]) if supported else None
        return str(supported)

    target_id = "single_sofa_chair_1"

    inc_chain = {oid: support_parent(incumbent, oid) for oid in incumbent_rows}
    cand_chain = {oid: support_parent(candidate, oid) for oid in candidate_rows}

    def children_of(parent_id: str, chain: dict) -> list[str]:
        return sorted(k for k, p in chain.items() if p == parent_id)

    inc_children = children_of(target_id, inc_chain)
    cand_children = children_of(target_id, cand_chain)

    # --- 3. Probe details (trimmed for brevity) ---
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    before_support = probe.get("before_support", {}) or {}
    after_support = probe.get("after_support", {}) or {}
    translation = probe.get("translation_delta_m", "N/A")
    rotation = probe.get("rotation_delta_deg", "N/A")

    # --- 4. Header ---
    lines = []
    sep = "-" * 95
    lines.append(sep)
    lines.append("SCENEPROOF SUPPORT REGRESSION AUDIT — Fix83 v2")
    lines.append(sep)
    lines.append("")
    lines.append(f"Target:         {target_id}")
    lines.append(f"Translation:    {translation:.8g} m" if isinstance(translation, (int, float)) else f"Translation:    {translation}")
    lines.append(f"Rotation:       {rotation:.8g} deg" if isinstance(rotation, (int, float)) else f"Rotation:       {rotation}")
    lines.append(f"Before COM margin: {before_support.get('com_signed_margin_m', 'N/A')}")
    lines.append(f"After  COM margin: {after_support.get('com_signed_margin_m', 'N/A')}")
    lines.append(f"Before stability:  {before_support.get('stability_class', 'N/A')}")
    lines.append(f"After  stability:  {after_support.get('stability_class', 'N/A')}")
    lines.append("")

    # --- 5. Per-object delta table ---
    lines.append(sep)
    fmt = "{:32} {:>8} {:>8} {:>8}  {:>9} {:>9} {:>7}  {:>9} {:>20}"
    lines.append(fmt.format("object_id", "inc", "cand", "delta", "gap(m)", "cont(m)", "overlap", "inside(m)", "parent"))
    lines.append(sep)

    total_inc = 0.0
    total_cand = 0.0
    count = 0
    affected: list[tuple[str, float, str]] = []

    for oid in sorted(incumbent_rows.keys()):
        inc_r = incumbent_rows[oid]
        cand_r = candidate_rows.get(oid, {})

        inc_gap      = csv_float(inc_r.get("support_contact_gap_m"))
        inc_cont     = csv_float(inc_r.get("support_containment_error_m"))
        inc_overlap  = csv_float(inc_r.get("support_footprint_overlap_ratio"))
        inc_inside   = csv_float(inc_r.get("inside_containment_error_m"))

        cand_gap     = csv_float(cand_r.get("support_contact_gap_m"))
        cand_cont    = csv_float(cand_r.get("support_containment_error_m"))
        cand_overlap = csv_float(cand_r.get("support_footprint_overlap_ratio"))
        cand_inside  = csv_float(cand_r.get("inside_containment_error_m"))

        inc_s = support_score(inc_gap, inc_cont, inc_overlap, inc_inside)
        cand_s = support_score(cand_gap, cand_cont, cand_overlap, cand_inside)

        if inc_s is not None and cand_s is not None:
            delta = cand_s - inc_s
            total_inc += inc_s
            total_cand += cand_s
            count += 1
        elif inc_s is not None:
            delta = -inc_s
            total_inc += inc_s
            count += 1
        elif cand_s is not None:
            delta = cand_s
            total_cand += cand_s
            count += 1
        else:
            delta = 0.0

        parent = inc_chain.get(oid) or "—"
        marker = ""
        if abs(delta) > 1e-9:
            marker = " *** CHANGED"
            affected.append((oid, delta, parent))
        elif oid == target_id:
            marker = " [TARGET]"

        # Determine display path
        if inc_inside is not None:
            kind = "INSIDE"
        elif inc_gap is not None:
            kind = "REGULAR"
        else:
            kind = "NO_PARENT"

        disp_inc = f"{inc_s:.4f}" if inc_s is not None else "None"
        disp_cand = f"{cand_s:.4f}" if cand_s is not None else "None"
        disp_gap = f"{inc_gap:.6g}" if inc_gap is not None else "—"
        disp_cont = f"{inc_cont:.6g}" if inc_cont is not None else "—"
        disp_overlap = f"{inc_overlap:.4g}" if inc_overlap is not None else "—"
        disp_inside = f"{inc_inside:.6g}" if inc_inside is not None else "—"

        lines.append(
            f"{oid:32} {disp_inc:>8} {disp_cand:>8} {delta:>+8.4f}  "
            f"{disp_gap:>9} {disp_cont:>9} {disp_overlap:>7}  "
            f"{disp_inside:>9} {parent:20}"
            f"{marker}"
        )

    lines.append(sep)
    inc_mean = total_inc / count if count else 0.0
    cand_mean = total_cand / count if count else 0.0
    lines.append(
        f"FAMILY MEAN (reconstructed): fix76={inc_mean:.6f}  "
        f"fix82={cand_mean:.6f}  delta={cand_mean - inc_mean:+.6f}  "
        f"(n={count} objects contribute to support family)"
    )

    # --- 6. Detailed per-object raw metric diff (only for changed objects) ---
    if affected:
        lines.append("")
        lines.append(sep)
        lines.append("RAW METRIC DIFFS (objects with score change):")
        lines.append(sep)
        for oid, delta, parent in affected:
            lines.append("")
            lines.append(f"  {oid}  delta={delta:+.6f}  parent={parent}")
            ir = incumbent_rows[oid]
            cr = candidate_rows.get(oid, {})
            for key in [
                "support_contact_gap_m",
                "support_containment_error_m",
                "support_footprint_overlap_ratio",
                "inside_containment_error_m",
            ]:
                vi = ir.get(key, "")
                vc = cr.get(key, "")
                if vi != vc:
                    lines.append(f"    {key}:  fix76={vi}  →  fix82={vc}")
            # Also show local_realizability change
            li = ir.get("local_realizability", "")
            lc = cr.get("local_realizability", "")
            if li != lc:
                lines.append(f"    local_realizability:  fix76={li}  →  fix82={lc}")

    # --- 7. Support chain analysis ---
    lines.append("")
    lines.append(sep)
    lines.append("SUPPORT CHAIN ANALYSIS")
    lines.append(sep)

    # Show full support chain for the target
    chain = []
    cur = target_id
    while cur:
        chain.append(cur)
        cur = inc_chain.get(cur)
    lines.append(f"Target chain: {' → '.join(chain)}")

    # Show what objects the target supports
    if inc_children:
        lines.append(f"Objects supported BY {target_id}: {inc_children}")
    else:
        lines.append(f"{target_id} supports NO other objects (is a leaf).")

    # Show what other objects have the same parent as the target
    target_parent = inc_chain.get(target_id)
    if target_parent:
        siblings = sorted(
            k for k, p in inc_chain.items()
            if p == target_parent and k != target_id
        )
        lines.append(f"Sibling objects (same parent '{target_parent}'): {siblings}")

    # --- 8. Conclusion ---
    lines.append("")
    lines.append(sep)
    lines.append("CONCLUSION")
    lines.append(sep)
    lines.append("")

    if not affected:
        lines.append("No per-object support score changed between Fix76 and Fix82.")
        lines.append("The -0.008 family-level delta reported by the component gates")
        lines.append("cannot be reproduced from per-object CSV metrics.")
        lines.append("")
        lines.append("Possible explanations:")
        lines.append("  1. The physical_objects.csv and physical.json may have been")
        lines.append("     generated from different eval runs or with different")
        lines.append("     geometry snapshots.")
        lines.append("  2. The support family's N (number of scored objects) may differ")
        lines.append("     between versions — e.g. an object gains/loses a support parent")
        lines.append("     and enters/leaves the denominator.")
        lines.append("  3. A rounding/aggregation artefact in the scene-level report")
        lines.append("     that does not appear at the per-object level.")
    elif len(affected) == 1 and affected[0][0] == target_id:
        lines.append(
            "The support regression is CONFINED to the target object itself.\n"
            "No other object's support score changed.\n"
            "The gravity-settled pose has worse footprint-based support metrics\n"
            "(gap/containment/overlap) relative to its parent, even though the\n"
            "true-mesh COM audit certifies it as locally stable.\n"
            "This is a discrepancy between the PROXY footprint metric and the\n"
            "true-mesh COM support measurement — the former penalizes poses that\n"
            "the latter accepts."
        )
    elif any(oid in inc_children for oid, _, _ in affected):
        lines.append(
            "The support regression PROPAGATES to children of the target.\n"
            "Changing the chair's pose degrades support for objects ON it."
        )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
