#!/usr/bin/env python3
"""Imaginarium v4 entry point.

v4 = v4-deepsearch + the full 400-iteration LayoutVLM S4 optimizer.

The legacy SA path remains available through the original launcher. Both the
LayoutVLM stage and iteration count remain environment-overridable for
ablation experiments.
"""

import os

os.environ["IMAGINARIUM_FLOOR_VERIFY_V2"] = "1"
os.environ["IMAGINARIUM_S3_STACK_AWARE"] = "1"
os.environ["IMAGINARIUM_S4_STACK_AWARE"] = "1"
os.environ["IMAGINARIUM_USE_DEEPSEARCH"] = "1"
os.environ["IMAGINARIUM_USE_LAYOUTVLM"] = "1"
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_STAGE", "full")
os.environ.setdefault("IMAGINARIUM_LAYOUTVLM_ITERATIONS", "400")

from run_imaginarium_I2Layout import main


if __name__ == "__main__":
    main()
