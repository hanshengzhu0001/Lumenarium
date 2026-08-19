#!/usr/bin/env python3
"""Record which trial seed the V5-demo profile must reuse.

Reading guide:
  list [SCENE_PREFIX]  every finished run with its seed, newest first
  set  SEED | JOB_ID   write DEMO_PIN.json atomically
  show                 print the pin the service will actually use
  clear                disable the demo profile again

Seeds are read from the artifacts the service already wrote, at
api_state/artifacts/<release>/<digest>/<profile>/<run>/result.json, so a pin can
be set with the server stopped and without trusting shell history.  The write is
atomic because a half-written pin would be indistinguishable from an unset one
while the service is running.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sceneproof_api.demo_pin import (  # noqa: E402  (path setup must precede)
    PIN_PATH,
    SEED_ENVIRONMENT_VARIABLE,
    SEED_MAXIMUM,
    load_demo_pin,
)

JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def _records(state_root: Path) -> list[dict]:
    found = []
    for path in state_root.glob("artifacts/*/*/*/*/result.json"):
        try:
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            modified = path.stat().st_mtime
        except (OSError, ValueError):
            continue
        seed = document.get("trial_seed")
        if not isinstance(seed, int):
            continue
        digest = path.parents[2].name
        found.append({
            "seed": seed,
            "job_id": str(document.get("job_id") or ""),
            "profile": str(document.get("profile") or path.parents[1].name),
            "status": str(document.get("status") or ""),
            "final_version": str(document.get("final_version") or ""),
            "digest": digest,
            "scene_id": digest[:32],
            "modified": modified,
        })
    return sorted(found, key=lambda record: record["modified"], reverse=True)


def _write_pin(record: dict, pin_path: Path) -> dict:
    pin = {
        "seed": record["seed"],
        "scene_id": record.get("scene_id"),
        "job_id": record.get("job_id") or None,
        "profile": "best",
        "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": record.get("note")
        or "V5-demo runs the best pipeline with this seed instead of deriving one.",
    }
    temporary = pin_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    temporary.replace(pin_path)
    return pin


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("list", "set", "show", "clear"))
    parser.add_argument("value", nargs="?", default=None)
    parser.add_argument("--state-root", default=None)
    parser.add_argument("--pin-path", default=None)
    arguments = parser.parse_args()
    state_root = Path(
        arguments.state_root or ROOT / "api_state"
    ).expanduser().resolve()
    pin_path = Path(arguments.pin_path) if arguments.pin_path else PIN_PATH

    if arguments.action == "show":
        pin = load_demo_pin(pin_path)
        print(json.dumps(pin or {"seed": None}, indent=2))
        print(f"file={pin_path}  env_override={SEED_ENVIRONMENT_VARIABLE}")
        return 0

    if arguments.action == "clear":
        _write_pin({"seed": None}, pin_path)
        print(f"cleared {pin_path}; the demo profile now refuses submissions")
        return 0

    records = _records(state_root)
    if arguments.action == "list":
        if not records:
            print(f"no result.json with a trial_seed under {state_root}/artifacts")
            return 1
        prefix = (arguments.value or "").lower()
        print(f"{'modified':16} {'seed':>11} {'profile':8} {'status':26} "
              f"{'scene':18} job")
        for record in records:
            if prefix and not record["digest"].startswith(prefix):
                continue
            print("{:16} {:>11} {:8} {:26} {:18} {}".format(
                time.strftime("%m-%d %H:%M", time.localtime(record["modified"])),
                record["seed"], record["profile"], record["status"][:26],
                record["digest"][:16], record["job_id"][:12]))
        return 0

    if not arguments.value:
        parser.error("set requires a seed or a 32-hex job id")
    chosen: dict | None = None
    if JOB_ID_PATTERN.match(arguments.value):
        chosen = next(
            (r for r in records if r["job_id"] == arguments.value), None
        )
        if chosen is None:
            print(f"no result.json found for job {arguments.value}; "
                  f"run 'list' to see what this host has", file=sys.stderr)
            return 1
    else:
        try:
            seed = int(arguments.value)
        except ValueError:
            print("value must be an integer seed or a 32-hex job id",
                  file=sys.stderr)
            return 2
        if not 0 <= seed <= SEED_MAXIMUM:
            print(f"seed must be within 0..{SEED_MAXIMUM}", file=sys.stderr)
            return 2
        chosen = {"seed": seed, "note": "pinned from an explicit seed value."}
    pin = _write_pin(chosen, pin_path)
    print(json.dumps(pin, indent=2))
    print(f"written to {pin_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
