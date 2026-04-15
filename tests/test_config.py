"""无麦克风、无网络：仅测试指令词匹配逻辑。"""

import unittest

from robot_auditory.config import (
    DEFAULT_MOVE_DISTANCE_M,
    DEFAULT_TURN_ANGLE_DEG,
    extract_angle_deg,
    extract_distance_m,
    extract_speed_scale,
    extract_special_turn_angle_deg,
    match_command,
    match_motion_commands,
    match_move_command,
    normalize_text,
)


class TestMatchCommand(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize_text("  前进， "), "前进")

    def test_forward(self) -> None:
        self.assertEqual(match_command("请向前走一步"), "forward")

    def test_stop(self) -> None:
        self.assertEqual(match_command("马上停下来"), "stop")

    def test_turn_left(self) -> None:
        self.assertEqual(match_command("往左转"), "turn_left")

    def test_unknown(self) -> None:
        self.assertIsNone(match_command("今天天气真好"))

    def test_forward_synonym(self) -> None:
        self.assertEqual(match_command("请直走两步"), "forward")

    def test_backward_synonym(self) -> None:
        self.assertEqual(match_command("请倒车一步"), "backward")

    def test_stop_synonym(self) -> None:
        self.assertEqual(match_command("马上刹车"), "stop")

    def test_status_synonym(self) -> None:
        self.assertEqual(match_command("报告一下当前位置"), "status")


class TestDistanceParse(unittest.TestCase):
    def test_one_step(self) -> None:
        self.assertAlmostEqual(extract_distance_m("前进一步"), DEFAULT_MOVE_DISTANCE_M)

    def test_two_steps_digit(self) -> None:
        self.assertAlmostEqual(extract_distance_m("前进2步"), DEFAULT_MOVE_DISTANCE_M * 2)

    def test_one_meter(self) -> None:
        self.assertAlmostEqual(extract_distance_m("向前走1米"), 1.0)

    def test_two_meters_ch(self) -> None:
        self.assertAlmostEqual(extract_distance_m("后退两米"), 2.0)

    def test_one_meter_m_symbol(self) -> None:
        self.assertAlmostEqual(extract_distance_m("前进1m"), 1.0)

    def test_one_step_is_half_meter(self) -> None:
        self.assertAlmostEqual(extract_distance_m("前进一步"), 0.5)


class TestAngleParse(unittest.TestCase):
    def test_turn_30_degrees(self) -> None:
        self.assertAlmostEqual(extract_angle_deg("右转30度"), 30.0)

    def test_turn_chinese_degrees(self) -> None:
        self.assertAlmostEqual(extract_angle_deg("左转三十度"), 30.0)

    def test_special_turn_around(self) -> None:
        self.assertAlmostEqual(extract_special_turn_angle_deg("掉头"), 180.0)

    def test_special_turn_circle(self) -> None:
        self.assertAlmostEqual(extract_special_turn_angle_deg("右转一圈"), 360.0)


class TestSpeedParse(unittest.TestCase):
    def test_slow_speed(self) -> None:
        self.assertAlmostEqual(extract_speed_scale("慢慢前进"), 0.6)

    def test_fast_speed(self) -> None:
        self.assertAlmostEqual(extract_speed_scale("快速右转"), 1.5)


class TestMatchMoveCommand(unittest.TestCase):
    def test_forward_default(self) -> None:
        cmd = match_move_command("前进")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd[0], "forward")
        self.assertAlmostEqual(cmd[1], DEFAULT_MOVE_DISTANCE_M)

    def test_backward_two_steps(self) -> None:
        cmd = match_move_command("后退两步")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd[0], "backward")
        self.assertAlmostEqual(cmd[1], DEFAULT_MOVE_DISTANCE_M * 2)

    def test_turn_left(self) -> None:
        cmd = match_move_command("左转")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd[0], "turn_left")
        self.assertAlmostEqual(cmd[1], 0.0)

    def test_multi_motion_command(self) -> None:
        cmds = match_motion_commands("前进1m然后左转30度再前进1m")
        self.assertEqual([c.cmd_id for c in cmds], ["forward", "turn_left", "forward"])
        self.assertAlmostEqual(cmds[0].distance_m, 1.0)
        self.assertAlmostEqual(cmds[1].angle_deg, 30.0)
        self.assertAlmostEqual(cmds[2].distance_m, 1.0)

    def test_default_turn_angle(self) -> None:
        cmds = match_motion_commands("右转")
        self.assertEqual(cmds[0].cmd_id, "turn_right")
        self.assertAlmostEqual(cmds[0].angle_deg, DEFAULT_TURN_ANGLE_DEG)

    def test_turn_around_command(self) -> None:
        cmds = match_motion_commands("掉头")
        self.assertEqual(cmds[0].cmd_id, "turn_left")
        self.assertAlmostEqual(cmds[0].angle_deg, 180.0)

    def test_turn_circle_command(self) -> None:
        cmds = match_motion_commands("右转一圈")
        self.assertEqual(cmds[0].cmd_id, "turn_right")
        self.assertAlmostEqual(cmds[0].angle_deg, 360.0)

    def test_multi_motion_with_synonyms(self) -> None:
        cmds = match_motion_commands("直走1米然后右拐再倒车一步")
        self.assertEqual([c.cmd_id for c in cmds], ["forward", "turn_right", "backward"])
        self.assertAlmostEqual(cmds[0].distance_m, 1.0)
        self.assertAlmostEqual(cmds[1].angle_deg, 90.0)
        self.assertAlmostEqual(cmds[2].distance_m, 0.5)

    def test_slow_forward_command(self) -> None:
        cmds = match_motion_commands("慢慢前进两步")
        self.assertEqual(cmds[0].cmd_id, "forward")
        self.assertAlmostEqual(cmds[0].distance_m, 1.0)
        self.assertAlmostEqual(cmds[0].speed_scale, 0.6)

    def test_fast_turn_command(self) -> None:
        cmds = match_motion_commands("快速右转半圈")
        self.assertEqual(cmds[0].cmd_id, "turn_right")
        self.assertAlmostEqual(cmds[0].angle_deg, 180.0)
        self.assertAlmostEqual(cmds[0].speed_scale, 1.5)


if __name__ == "__main__":
    unittest.main()
