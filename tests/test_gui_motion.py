"""无 GUI 运行环境下测试小车运动学。"""

import unittest

from robot_auditory.gui_app import RobotState, _apply_move


class TestGuiMotion(unittest.TestCase):
    def test_initial_forward_faces_positive_x(self) -> None:
        state = RobotState()
        _apply_move(state, "forward", 1.0)
        self.assertAlmostEqual(state.x, 1.0)
        self.assertAlmostEqual(state.y, 0.0)

    def test_left_turn_then_forward_goes_negative_y(self) -> None:
        state = RobotState()
        _apply_move(state, "turn_left", 0.0, step_yaw_deg=90.0)
        _apply_move(state, "forward", 1.0)
        self.assertAlmostEqual(state.x, 0.0, places=6)
        self.assertAlmostEqual(state.y, -1.0, places=6)

    def test_right_turn_30_degrees(self) -> None:
        state = RobotState()
        _apply_move(state, "turn_right", 0.0, step_yaw_deg=30.0)
        self.assertAlmostEqual(state.yaw_deg, 30.0)


if __name__ == "__main__":
    unittest.main()
