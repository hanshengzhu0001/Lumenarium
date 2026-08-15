#!/usr/bin/env python3
"""
Imaginarium CLI Entry Point - v4-deepsearch (v3 + S2 DeepSearch)

v4-deepsearch keeps v3's floor verification and stack-aware S3/S4 behavior,
while replacing only S2 asset retrieval with Omniverse DeepSearch.

Usage:
    python run_imaginarium_I2Layout_v4_deepsearch.py <image_path> [--debug] [--clean]
"""
import os

os.environ['IMAGINARIUM_FLOOR_VERIFY_V2'] = '1'
os.environ['IMAGINARIUM_S3_STACK_AWARE'] = '1'
os.environ['IMAGINARIUM_S4_STACK_AWARE'] = '1'
os.environ['IMAGINARIUM_USE_DEEPSEARCH'] = '1'

from run_imaginarium_I2Layout import main


if __name__ == "__main__":
    main()
