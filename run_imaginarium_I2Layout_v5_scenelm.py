#!/usr/bin/env python3
"""Imaginarium v5 SceneLM entry point.

v5 keeps the v4-deepsearch perception/retrieval/pose stack and replaces only
the S4 Adam/SA backend with relation-conditioned matrix-free SceneLM.  Every
setting remains environment-overridable for paired ablations.
"""

import os

os.environ["IMAGINARIUM_FLOOR_VERIFY_V2"] = "1"
os.environ["IMAGINARIUM_S3_STACK_AWARE"] = "1"
os.environ["IMAGINARIUM_S4_STACK_AWARE"] = "1"
os.environ["IMAGINARIUM_USE_DEEPSEARCH"] = "1"
os.environ["IMAGINARIUM_USE_LAYOUTVLM"] = "1"
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_STAGE", "full")
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_SOLVER", "v5_scenelm")
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_ITERATIONS", "30")
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER", "0")

from run_imaginarium_I2Layout import main


if __name__ == "__main__":
    main()
