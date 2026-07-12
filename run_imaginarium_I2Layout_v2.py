#!/usr/bin/env python3
"""
Imaginarium CLI Entry Point — V2 (方案A: 保护支撑物，地面验证不误删)
Imaginarium 命令行入口 —— V2

与 run_imaginarium_I2Layout.py 行为完全一致，唯一区别：
开启环境变量 IMAGINARIUM_FLOOR_VERIFY_V2=1，使 S1 的"地面父物体验证"
使用 verify_floor_parent_with_vlm_v2（方案A），处理逻辑：

  - 被 VLM 判定"非地面"的物体不再被粗暴删除；
  - 规则1：若已挂在非地面父物体上 -> 仅修正 isOnFloor，保留；
  - 规则2(最高优先级)：若它是其他物体的支撑者(被任意 supported 引用)
          -> 绝不删除；尝试几何兜底重新挂到下方父物体，找不到则保留为地面支撑；
  - 规则3：非支撑者且能几何找到下方父物体 -> 重新挂父；
          仅"既不支撑别人、又找不到任何合理父物体"的孤立物体才作为幻觉删除。

这修复了"小箱子底下的大箱子被误删 -> 小箱子悬空/被误判靠墙"的问题。

用法 / Usage:
    python run_imaginarium_I2Layout_v2.py <image_path> [--debug] [--clean]

示例 / Example:
    python run_imaginarium_I2Layout_v2.py demo/custom_scene3.png --clean
"""

import os

# 开启方案A v2 地面验证（必须在导入/运行 pipeline 之前设置）
os.environ['IMAGINARIUM_FLOOR_VERIFY_V2'] = '1'

from run_imaginarium_I2Layout import main

if __name__ == "__main__":
    main()
