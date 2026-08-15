#!/usr/bin/env python3
"""Prepare and select fail-closed subsets of accepted settle candidates."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(args: argparse.Namespace) -> None:
    fields, rows = read_tsv(args.selected)
    accepted = [row for row in rows if row.get("accepted") == "true"]
    if not accepted:
        raise SystemExit("No accepted candidates")
    manifest = []
    for mask in range(1 << len(accepted)):
        chosen = {
            (row["scene"], row["object_id"])
            for bit, row in enumerate(accepted)
            if mask & (1 << bit)
        }
        subset_rows = []
        for row in rows:
            copy = dict(row)
            copy["accepted"] = str((row["scene"], row["object_id"]) in chosen).lower()
            subset_rows.append(copy)
        subset = args.out_dir / f"subset_{mask:02d}.tsv"
        write_tsv(subset, fields, subset_rows)
        manifest.append(
            {
                "mask": mask,
                "count": len(chosen),
                "objects": [f"{scene}/{obj}" for scene, obj in sorted(chosen)],
                "selected": str(subset.resolve()),
                "target_version": f"{args.target_prefix}_subset_{mask:02d}",
            }
        )
    (args.out_dir / "subsets.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SUBSETS={len(manifest)} CANDIDATES={len(accepted)}")


def select(args: argparse.Namespace) -> None:
    subsets = json.loads((args.out_dir / "subsets.json").read_text(encoding="utf-8"))
    audited = []
    for subset in subsets:
        physical_path = args.out_dir / f"physical_{subset['mask']:02d}.json"
        physical = json.loads(physical_path.read_text(encoding="utf-8"))
        versions = physical["versions"]
        base = versions[args.baseline_version]["aggregate"]
        target = versions[subset["target_version"]]["aggregate"]
        deltas = {}
        passed = True
        for family in ("collision", "support", "plane", "semantic"):
            before = base.get("families", {}).get(family, {}).get("score")
            after = target.get("families", {}).get(family, {}).get("score")
            delta = None
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (before, after)):
                delta = float(after) - float(before)
                passed = passed and delta >= -args.tolerance
            deltas[family] = delta
        macro = float(target["headline_macro_realizability"]) - float(
            base["headline_macro_realizability"]
        )
        passed = passed and macro >= -args.tolerance
        audited.append({**subset, "family_deltas": deltas, "macro_delta": macro, "passed": passed})
    passing = [row for row in audited if row["passed"]]
    if not passing:
        raise SystemExit("No fail-closed subset (empty subset should always pass)")
    winner = max(passing, key=lambda row: (row["count"], row["macro_delta"], row["mask"]))
    fields, rows = read_tsv(Path(winner["selected"]))
    write_tsv(args.final_selected, fields, rows)
    audit = {
        "schema_version": "sceneproof_generalized_settle_subset_selection_v1",
        "baseline_version": args.baseline_version,
        "tolerance": args.tolerance,
        "subsets": audited,
        "winner": winner,
        "decision": "promote_noninferior_maximum_cardinality_subset",
    }
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASSING={len(passing)}/{len(audited)} WINNER_MASK={winner['mask']:02d} "
        f"RETAINED={winner['count']} OBJECTS={','.join(winner['objects']) or '-'} "
        f"MACRO_DELTA={winner['macro_delta']:+.9f} FAMILY_DELTAS={json.dumps(winner['family_deltas'], sort_keys=True)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--selected", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--target-prefix", required=True)
    p.set_defaults(func=prepare)
    s = sub.add_parser("select")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--baseline-version", required=True)
    s.add_argument("--final-selected", type=Path, required=True)
    s.add_argument("--audit", type=Path, required=True)
    s.add_argument("--tolerance", type=float, default=1e-9)
    s.set_defaults(func=select)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
