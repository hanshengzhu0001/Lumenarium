#!/usr/bin/env python3
"""Select near-pass representatives without class or object allowlists."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROXY_ONLY_FAILURES = {
    "support_noninferior",
    "no_increased_collision_penetration",
    "no_increased_collision_volume",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    selected = []
    rejected = []
    with args.trials.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = Path(row["gates"])
            if path.suffix != ".json" or not path.is_file():
                rejected.append({**row, "reason": "missing_gate_json"})
                continue
            try:
                gate = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                rejected.append({**row, "reason": "invalid_gate_json"})
                continue
            failures = {
                name
                for name, passed in gate.get("gates", {}).items()
                if passed is False
            }
            if failures and failures <= PROXY_ONLY_FAILURES:
                selected.append(
                    {
                        "scene": row["scene"],
                        "object_id": row["object_id"],
                        "previous_failures": sorted(failures),
                    }
                )
            else:
                rejected.append(
                    {**row, "reason": "not_proxy_only_nearpass", "failures": sorted(failures)}
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(f"{r['scene']}\t{r['object_id']}\n" for r in selected),
        encoding="utf-8",
    )
    args.audit.write_text(
        json.dumps({"selected": selected, "rejected": rejected}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out.resolve()}")
    print(f"SELECTED={len(selected)} REJECTED={len(rejected)}")


if __name__ == "__main__":
    main()
