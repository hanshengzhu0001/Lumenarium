#!/usr/bin/env python3
import argparse
import json
import statistics
from pathlib import Path


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.audit_root)
    groups = {}
    failures = []
    for path in sorted(root.glob("runtime_w*.jsonl")):
        rows = load_jsonl(path)
        if not rows:
            continue
        workers = int(rows[0]["workers"])
        bad = [row for row in rows if row["status"] != "ok"]
        failures.extend(bad)
        values = [float(row["elapsed_seconds"]) for row in rows]
        groups[workers] = {
            "scenes": len(rows),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "rows": rows,
        }

    baseline = groups.get(1, {}).get("mean_seconds")
    for workers, record in groups.items():
        record["speedup_vs_w1"] = (
            baseline / record["mean_seconds"] if baseline else None
        )

    output = {
        "schema_version": "deepsearch_s2_concurrency_audit_v1",
        "scope": "cached_s0_s1_s2_only_skip_s2_vlm",
        "groups": {str(key): value for key, value in sorted(groups.items())},
        "failures": failures,
        "passed": bool(groups.get(1) and groups.get(8) and not failures),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    for workers, record in sorted(groups.items()):
        print(
            f"WORKERS={workers} SCENES={record['scenes']} "
            f"MEAN={record['mean_seconds']:.3f}s "
            f"MEDIAN={record['median_seconds']:.3f}s "
            f"SPEEDUP={record['speedup_vs_w1']:.3f}x"
        )
    print(f"FAILURES={len(failures)} PASSED={output['passed']}")


if __name__ == "__main__":
    main()
