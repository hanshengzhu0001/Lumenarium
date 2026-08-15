#!/usr/bin/env python3
"""Frozen v4-fast entry point: v4-deepsearch plus Adam-400 S4."""

import os

os.environ["IMAGINARIUM_FLOOR_VERIFY_V2"] = "1"
os.environ["IMAGINARIUM_S3_STACK_AWARE"] = "1"
os.environ["IMAGINARIUM_S4_STACK_AWARE"] = "1"
os.environ["IMAGINARIUM_USE_DEEPSEARCH"] = "1"
os.environ["IMAGINARIUM_USE_LAYOUTVLM"] = "1"
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_STAGE", "full")
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_SOLVER", "adam")
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_ITERATIONS", "400")
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_ACTIVE_SET_ROUTER", "0")

from run_imaginarium_I2Layout import main


if __name__ == "__main__":
    main()
