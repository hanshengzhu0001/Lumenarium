#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


TIME_COST = re.compile(r"(?:GEMINI|GPT|OPENAI)[^\n]*Time cost\s+([0-9.]+)s", re.I)
RETRY = re.compile(r"Error when sending request to the server \(Retry (\d+)/(\d+)\)")
POOL_FAIL = re.compile(r"GPT request (\d+)/(\d+) failed")


def phase_rows(rows, start_text, end_text):
    start = next((i for i, row in enumerate(rows) if start_text in row), None)
    if start is None:
        return []
    end = next((i for i in range(start + 1, len(rows)) if end_text in rows[i]), len(rows))
    return rows[start:end + 1]


def summarize(rows):
    latency = [float(m.group(1)) for row in rows if (m := TIME_COST.search(row))]
    retries = [m.groups() for row in rows if (m := RETRY.search(row))]
    pool_failures = [m.groups() for row in rows if (m := POOL_FAIL.search(row))]
    return {
        "completed_api_attempts_with_latency": len(latency),
        "latency_seconds": latency,
        "latency_sum_seconds": sum(latency),
        "latency_mean_seconds": sum(latency) / len(latency) if latency else None,
        "latency_max_seconds": max(latency) if latency else None,
        "transport_retry_events": len(retries),
        "pool_failure_events": len(pool_failures),
        "note": "A successful latency row counts an API attempt, not necessarily a unique logical request.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--scene", default="bedroom_01")
    ap.add_argument("--source-version", default="v4_deepsearch")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    log = args.results_root / f"{args.scene}_{args.source_version}_result" / "stage_logs" / "S1_parsing.log"
    rows = log.read_text(errors="replace").splitlines()
    phases = {
        "initial_scene_graph": phase_rows(
            rows, "Part 3.3", "Part 3.3: Scene graph generation finished"
        ),
        "floor_parent_verification": phase_rows(
            rows, "Part 3.4", "Part 3.4: Floor parent verification finished"
        ),
        "semantic_group_facing": phase_rows(
            rows, "Starting Part 3.5", "Part 3.5: Semantic analysis finished"
        ),
    }
    result = {
        "schema_version": "sceneproof_s1_api_concurrency_audit_v1",
        "scene": args.scene,
        "log": str(log.resolve()),
        "runtime_policy": {
            "requested_workers_per_function": 4,
            "fix124_environment_process_cap": 1,
            "effective_workers_per_call": 1,
            "cross_process_lock": "/tmp/lumenarium_fix124_gemini.lock",
            "effective_cross_scene_api_concurrency": 1,
            "deepsearch_workers_are_not_gemini_workers": True,
        },
        "retry_policy": {
            "response_transport_attempts": 5,
            "json_or_dict_parse_attempts": 3,
            "worst_case_api_attempts_per_logical_request": 15,
        },
        "phases": {name: summarize(value) for name, value in phases.items()},
        "floor_verification_classification": (
            "VLM/API dominated plus small local grid creation and O(N^2) bbox reparenting; "
            "it is not a purely local OBB loop."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("FIX124_GEMINI_PROCESS_CAP=1")
    print("FIX124_CROSS_PROCESS_LOCK=/tmp/lumenarium_fix124_gemini.lock")
    print("EFFECTIVE_GEMINI_CONCURRENCY=1")
    for name, row in result["phases"].items():
        print(
            f"PHASE={name} API_ATTEMPTS={row['completed_api_attempts_with_latency']} "
            f"LATENCY_SUM={row['latency_sum_seconds']:.3f} "
            f"LATENCY_MEAN={row['latency_mean_seconds']} LATENCY_MAX={row['latency_max_seconds']} "
            f"TRANSPORT_RETRIES={row['transport_retry_events']} POOL_FAILURES={row['pool_failure_events']}"
        )
    print(f"FIX132_AUDIT={args.out.resolve()}")


if __name__ == "__main__":
    main()
