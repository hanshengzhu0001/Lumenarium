"""放在支撑面上的物体如何贴合——纯几何决策，不依赖 bpy，因此可单测。

为什么单独一个模块
------------------
决策本身是一行算术，但它决定渲染图里最刺眼的那类错误是否出现，所以必须能被
测试钉死。`S4_blender_layout_and_corr` 在导入时就需要 `bpy`，测试进程里没有
Blender，于是把判断搬到这里，S4 只负责把Blender 量出来的间隙喂进来。

改动前的行为与它错在哪里
------------------------
原本 `process_other_objects` 对两个方向不对称：

    穿模（子物体底面低于父物体顶面）  ->  间隙精确归零
    悬空（子物体底面高于父物体顶面）  ->  只在超过 0.2 m 时才修，
                                          且修完仍然停在距顶面 0.2 m，
                                          0 到 0.2 m 的悬空完全不处理

`tree_sons` 只收录 `SpatialRel == "on"` 的关系，并且在构造处就排除了父物体是墙
的情况，所以走到这里的语义严格是"甲放在乙上面"，挂墙的画框一类不会进来。对这个
语义而言留一道缝就是错的：一只杯子浮在桌面上方两厘米，人一眼就能看出来，而
support 类指标却可能因为间隙很小而几乎不动。两个方向因此统一成同一个式子。

为什么保留一个间隙上限
----------------------
间隙小意味着"本该放在上面却浮着"，贴合一定是对的。间隙很大则更可能是支撑关系本身
判错——例如一盏吊灯被误判为放在某件家具上——这时贴合会把物体拍到一个错误的地方，
比留着悬空更糟。超过上限时保留改动前的行为，即只做原来的 0.2 m 夹断，因为在这种
情形下没有证据支持任何更激进的处理。上限默认取 0.5m，与 S3 已有的
`IMAGINARIUM_LAYOUTVLM_MAX_CONTACT_GAP` 同值：同一个判断不应该有两个常数。

被否决的替代方案
----------------
让刚体模拟去落地：主流程里的落地仿真已经在跑，却仍留下这些悬空，因为这里的间隙是
在仿真之后由 `process_z` 按支撑树重新赋值的；把决定权交回仿真等于要求两套机制对同
一个 z 达成一致，比统一这一行算术复杂得多，且不可复现。
"""

from __future__ import annotations

import os
import math

# 改动前写死在process_other_objects 里的悬空保留量。超过间隙上限时沿用它，
# 使关闭开关后的行为逐行等价于改动前。
LEGACY_RETAINED_FLOAT_M = 0.2

DEFAULT_MAX_SETTLE_GAP_M = 0.5


def rotation_explained_horizontal_motion(
    horizontal_translation_m: float,
    rotation_rad: float,
    horizontal_mesh_radius_m: float,
    *,
    slip_tolerance_m: float = 0.005,
) -> tuple[bool, float]:
    """Certify that XY motion can be explained by rotation about a contact pivot.

    A point at horizontal radius ``r`` moves by at most the rotation chord
    ``2 r sin(theta/2)``.  Motion beyond that chord is independent horizontal
    sliding and is not allowed by the local settle protocol.
    """
    translation = max(0.0, float(horizontal_translation_m))
    theta = min(math.pi, abs(float(rotation_rad)))
    radius = max(0.0, float(horizontal_mesh_radius_m))
    tolerance = max(0.0, float(slip_tolerance_m))
    bound = 2.0 * radius * math.sin(theta / 2.0) + tolerance
    return translation <= bound, bound


def shortest_rotation_angle(rotation_rad: float) -> float:
    """Fold any quaternion-reported angle onto the SO(3) geodesic [0, pi]."""
    angle = abs(float(rotation_rad)) % (2.0 * math.pi)
    return min(angle, 2.0 * math.pi - angle)


def resolve_settle_policy(environ: dict[str, str] | None = None) -> tuple[bool, float]:
    """从环境变量解析沉降策略。

    `IMAGINARIUM_SETTLE_ON_SUPPORT=0` 时完全退回改动前的行为，Fix61 基线因此
    可复现；这是把行为改动放在冻结基线上的前提，不是可选的方便设施。
    """
    env = os.environ if environ is None else environ
    enabled = env.get("IMAGINARIUM_SETTLE_ON_SUPPORT", "1") != "0"
    raw = env.get("IMAGINARIUM_SETTLE_MAX_GAP", str(DEFAULT_MAX_SETTLE_GAP_M))
    try:
        max_gap = float(raw)
    except (TypeError, ValueError):
        max_gap = DEFAULT_MAX_SETTLE_GAP_M
    if not (max_gap > 0.0):
        max_gap = DEFAULT_MAX_SETTLE_GAP_M
    return enabled, max_gap


def settle_after_simulation_enabled(environ: dict[str, str] | None = None) -> bool:
    """仿真之后是否重做一次支撑面对齐。

    默认开启，因为 s2 段单独做对齐已被实测证明无效：对齐结果会被之后的刚体仿真覆盖。
    单独一个开关而不是复用 IMAGINARIUM_SETTLE_ON_SUPPORT，是为了能把"两处都做"、
    "只在仿真前做"（即改动前的行为）、"只在仿真后做"三种情形分别跑出来对比；把它们
    绑在一个开关上就无法分辨到底是哪一处产生了效果。
    """
    env = os.environ if environ is None else environ
    return env.get("IMAGINARIUM_SETTLE_AFTER_SIM", "1") != "0"


def settle_delta_z(
    child_min_z: float,
    parent_max_z: float,
    *,
    enabled: bool = True,
    max_gap: float = DEFAULT_MAX_SETTLE_GAP_M,
) -> tuple[float, str]:
    """返回让子物体底面落到父物体顶面所需的 z 位移，以及采取该位移的理由。

    理由字符串会打进日志，使每一次位移都能在渲染前被逐条核对，而不是只看到一个
    汇总数字。
    """
    gap = float(child_min_z) - float(parent_max_z)

    if gap < 0.0:
        # 穿模：顶到支撑面。改动前就是这样，未做修改。
        return -gap, "lifted_out_of_penetration"

    if gap == 0.0:
        return 0.0, "already_in_contact"

    if enabled and gap <= max_gap:
        return -gap, "settled_onto_its_support"

    if gap > LEGACY_RETAINED_FLOAT_M:
        # 间隙超过上限，支撑关系本身可疑，只做改动前的夹断。
        return -(gap - LEGACY_RETAINED_FLOAT_M), "clamped_gap_left_unsettled"

    return 0.0, "small_gap_left_unsettled"
