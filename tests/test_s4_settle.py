"""贴合到支撑面的决策测试。

每个期望值都是手算的z 算术，并对应 process_other_objects 里改动前的一条分支，
所以关掉开关时的等价性是被钉死的，不是靠阅读代码相信的。
"""

from __future__ import annotations

import math
import unittest

from modules._s4_settle import (
    DEFAULT_MAX_SETTLE_GAP_M,
    LEGACY_RETAINED_FLOAT_M,
    resolve_settle_policy,
    rotation_explained_horizontal_motion,
    shortest_rotation_angle,
    settle_after_simulation_enabled,
    settle_delta_z,
)


class HorizontalMotionCertificateTests(unittest.TestCase):
    def test_rotation_angle_uses_shortest_so3_branch(self):
        self.assertAlmostEqual(
            shortest_rotation_angle(math.radians(323.0)),
            math.radians(37.0),
        )
        self.assertAlmostEqual(
            shortest_rotation_angle(math.radians(358.0)),
            math.radians(2.0),
        )

    def test_pure_drop_allows_only_numerical_xy_slip(self):
        passed, bound = rotation_explained_horizontal_motion(0.004, 0.0, 1.0)
        self.assertTrue(passed)
        self.assertAlmostEqual(bound, 0.005)
        self.assertFalse(
            rotation_explained_horizontal_motion(0.006, 0.0, 1.0)[0]
        )

    def test_tipping_allows_rotation_chord_but_rejects_extra_slide(self):
        passed, bound = rotation_explained_horizontal_motion(
            0.105, math.radians(60.0), 0.1
        )
        self.assertTrue(passed)
        self.assertAlmostEqual(bound, 0.105, places=9)
        self.assertFalse(
            rotation_explained_horizontal_motion(
                0.106, math.radians(60.0), 0.1
            )[0]
        )


class AfterSimulationSwitchTests(unittest.TestCase):
    """三种情形要能分别跑出来，否则无法分辨效果来自哪一处。"""

    def test_it_defaults_on_because_settling_before_the_simulation_is_overwritten(self):
        self.assertTrue(settle_after_simulation_enabled({}))

    def test_it_can_be_turned_off_independently(self):
        self.assertTrue(
            resolve_settle_policy({"IMAGINARIUM_SETTLE_AFTER_SIM": "0"})[0]
        )
        self.assertFalse(
            settle_after_simulation_enabled({"IMAGINARIUM_SETTLE_AFTER_SIM": "0"})
        )


class SettleTests(unittest.TestCase):
    def test_a_small_float_is_closed_completely(self):
        # 杯子浮在 0.75m 高的桌面上方 2cm：改动前完全不处理，因为不到 0.2m。
        delta, reason = settle_delta_z(0.77, 0.75)
        self.assertAlmostEqual(delta, -0.02, places=9)
        self.assertEqual(reason, "settled_onto_its_support")

    def test_a_float_just_under_the_legacy_threshold_is_closed(self):
        delta, reason = settle_delta_z(0.94, 0.75)
        self.assertAlmostEqual(delta, -0.19, places=9)
        self.assertEqual(reason, "settled_onto_its_support")

    def test_a_float_the_legacy_branch_only_clamped_is_now_closed(self):
        # 改动前：0.45的间隙夹到 0.2，仍留 0.2悬空。现在归零。
        delta, reason = settle_delta_z(1.20, 0.75)
        self.assertAlmostEqual(delta, -0.45, places=9)
        self.assertEqual(reason, "settled_onto_its_support")

    def test_penetration_is_lifted_exactly_as_before(self):
        delta, reason = settle_delta_z(0.70, 0.75)
        self.assertAlmostEqual(delta, 0.05, places=9)
        self.assertEqual(reason, "lifted_out_of_penetration")

    def test_contact_needs_no_move(self):
        delta, reason = settle_delta_z(0.75, 0.75)
        self.assertEqual(delta, 0.0)
        self.assertEqual(reason, "already_in_contact")

    def test_a_gap_beyond_the_limit_keeps_the_previous_behaviour(self):
        # 一盏被误判为放在家具上的吊灯：贴合会把它拍到家具顶上，比留着更糟。
        delta, reason = settle_delta_z(2.60, 0.75, max_gap=0.5)
        self.assertAlmostEqual(delta, -(1.85 - LEGACY_RETAINED_FLOAT_M), places=9)
        self.assertEqual(reason, "clamped_gap_left_unsettled")

    def test_the_limit_boundary_is_inclusive(self):
        delta, reason = settle_delta_z(1.25, 0.75, max_gap=0.5)
        self.assertAlmostEqual(delta, -0.5, places=9)
        self.assertEqual(reason, "settled_onto_its_support")


class DisabledEquivalenceTests(unittest.TestCase):
    """关掉开关必须逐条等价于改动前，否则冻结基线不可复现。"""

    def test_disabled_leaves_a_small_float_untouched(self):
        delta, reason = settle_delta_z(0.77, 0.75, enabled=False)
        self.assertEqual(delta, 0.0)
        self.assertEqual(reason, "small_gap_left_unsettled")

    def test_disabled_clamps_a_large_float_to_the_legacy_value(self):
        delta, reason = settle_delta_z(1.20, 0.75, enabled=False)
        self.assertAlmostEqual(delta, -(0.45 - LEGACY_RETAINED_FLOAT_M), places=9)
        self.assertEqual(reason, "clamped_gap_left_unsettled")

    def test_disabled_still_fixes_penetration(self):
        delta, reason = settle_delta_z(0.70, 0.75, enabled=False)
        self.assertAlmostEqual(delta, 0.05, places=9)
        self.assertEqual(reason, "lifted_out_of_penetration")


class PolicyTests(unittest.TestCase):
    def test_the_default_is_on(self):
        enabled, max_gap = resolve_settle_policy({})
        self.assertTrue(enabled)
        self.assertAlmostEqual(max_gap, DEFAULT_MAX_SETTLE_GAP_M)

    def test_the_kill_switch_is_the_string_zero(self):
        enabled, _ = resolve_settle_policy({"IMAGINARIUM_SETTLE_ON_SUPPORT": "0"})
        self.assertFalse(enabled)

    def test_a_custom_limit_is_read(self):
        _, max_gap = resolve_settle_policy({"IMAGINARIUM_SETTLE_MAX_GAP": "0.15"})
        self.assertAlmostEqual(max_gap, 0.15)

    def test_a_malformed_or_nonpositive_limit_falls_back(self):
        for raw in ("", "abc", "0", "-1"):
            with self.subTest(raw=raw):
                _, max_gap = resolve_settle_policy(
                    {"IMAGINARIUM_SETTLE_MAX_GAP": raw}
                )
                self.assertAlmostEqual(max_gap, DEFAULT_MAX_SETTLE_GAP_M)


if __name__ == "__main__":
    unittest.main()
