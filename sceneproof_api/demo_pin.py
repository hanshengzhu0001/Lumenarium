"""Pinned trial seed for the demo profile.

Why this is its own module: the coordinator must refuse a demo job before it is
queued, and the worker must inject the same number when it runs.  A demo whose
two halves disagree about the seed is worse than having no demo profile.

Why a missing pin is an error rather than a fallback: this profile's only promise
is "you get the run that was rehearsed".  Falling back to the job-derived seed
would still render a scene and still report success, so the operator would learn
the promise was void only by noticing the output differs -- on stage.  Rejected
alternative: pinning through the worker's process environment alone.  It cannot
be validated at submission time, so a missing pin would surface ten minutes into
a cold run instead of in the first second.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


PIN_PATH = Path(__file__).with_name("DEMO_PIN.json")
SEED_ENVIRONMENT_VARIABLE = "SCENEPROOF_API_DEMO_SEED"
SEED_MAXIMUM = 0x7FFFFFFF
UNSET_REASON = (
    "the demo profile requires a pinned trial seed, and none is set on this "
    "host; run: python scripts/pin_sceneproof_demo_seed_fix154.py set "
    f"<seed-or-job-id>   (or export {SEED_ENVIRONMENT_VARIABLE}=<seed> before "
    "restarting the service)"
)


def _coerce_seed(value: object) -> int | None:
    """Accept only what the pipeline accepts: a plain integer in seed range."""
    if value is None or isinstance(value, bool):
        return None
    try:
        seed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return seed if 0 <= seed <= SEED_MAXIMUM else None


def load_demo_pin(path: Path | None = None) -> dict:
    """Return the effective pin, or an empty dict when it is unset.

    The environment variable wins over the file so a single run can be pinned
    without editing state that outlives it; the file is the durable form,
    because workers are restarted far more often than they are reconfigured.
    """
    from_environment = _coerce_seed(os.environ.get(SEED_ENVIRONMENT_VARIABLE))
    if from_environment is not None:
        return {"seed": from_environment, "source": SEED_ENVIRONMENT_VARIABLE}
    location = PIN_PATH if path is None else Path(path)
    try:
        # utf-8-sig, not utf-8: an operator editing this file on the box may
        # leave a BOM, and a BOM would otherwise read as "no pin is set" at the
        # worst possible moment.
        document = json.loads(location.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    seed = _coerce_seed(document.get("seed"))
    if seed is None:
        return {}
    pin = {"seed": seed, "source": str(location)}
    for key in ("scene_id", "job_id", "profile", "pinned_at", "note"):
        if document.get(key):
            pin[key] = document[key]
    return pin


def demo_seed(path: Path | None = None) -> int | None:
    return load_demo_pin(path).get("seed")
