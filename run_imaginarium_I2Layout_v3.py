#!/usr/bin/env python3
"""
Imaginarium CLI Entry Point — V3 (方案A(v2) + 堆叠感知(v3))
Imaginarium 命令行入口 —— V3

v3 在 v2 基础上新增"堆叠感知"：
  - 检测落地物体之间的堆叠关系（如 wooden_pallet 叠在 crate 下方）
  - 在 S3 自动标注 stacking_pairs
  - 在 S4 模拟退火中给堆叠物体的 upper 提供 z 轴向上扰动
  - 加入堆叠分离惩罚，防止 SA 把叠在一起的物体推开

核心改动：
  IMAGINARIUM_FLOOR_VERIFY_V2=1   (v2: 保护支撑物不误删)
  IMAGINARIUM_S3_STACK_AWARE=1    (v3: S3 堆叠检测)
  IMAGINARIUM_S4_STACK_AWARE=1    (v3: S4 堆叠感知 SA)

用法 / Usage:
    python run_imaginarium_I2Layout_v3.py <image_path> [--debug] [--clean]

示例 / Example:
    python run_imaginarium_I2Layout_v3.py demo/custom_scene3.png --clean
"""

import os

# 开启 v2: 方案A 地面验证（保护支撑物不误删）
os.environ['IMAGINARIUM_FLOOR_VERIFY_V2'] = '1'

# 开启 v3: 堆叠感知
os.environ['IMAGINARIUM_S3_STACK_AWARE'] = '1'
os.environ['IMAGINARIUM_S4_STACK_AWARE'] = '1'

from run_imaginarium_I2Layout import main

if __name__ == "__main__":
    main()
