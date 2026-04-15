"""播报回声屏蔽测试。"""

import time
import unittest

from robot_auditory import tts


class TestTtsGuard(unittest.TestCase):
    def setUp(self) -> None:
        tts._RECENT_SPOKEN.clear()
        tts._LISTEN_BLOCK_UNTIL = 0.0

    def test_recent_spoken_text_should_be_ignored(self) -> None:
        tts._remember_spoken_text("收到，正在前进。", hold_seconds=5.0)
        ignored, reason = tts.should_ignore_recognized_text("收到正在前进")
        self.assertTrue(ignored)
        self.assertIsNotNone(reason)

    def test_different_text_should_not_be_ignored(self) -> None:
        tts._remember_spoken_text("收到，正在前进。", hold_seconds=5.0)
        ignored, _ = tts.should_ignore_recognized_text("今天天气不错")
        self.assertFalse(ignored)

    def test_listen_block_window(self) -> None:
        tts._set_listen_block(0.2)
        self.assertGreater(tts.remaining_listen_block(), 0.0)
        time.sleep(0.25)
        self.assertLessEqual(tts.remaining_listen_block(), 0.02)

    def test_long_text_should_get_longer_guard(self) -> None:
        short_guard = tts._calc_guard_seconds("好的")
        long_guard = tts._calc_guard_seconds("这是一个很长很长的系统反馈，用来测试动态保护时间是否会自动延长。")
        self.assertGreater(long_guard, short_guard)

    def test_expired_spoken_text_should_not_be_ignored(self) -> None:
        tts._remember_spoken_text("收到，正在前进。", hold_seconds=0.1)
        time.sleep(0.15)
        ignored, _ = tts.should_ignore_recognized_text("收到正在前进")
        self.assertFalse(ignored)


if __name__ == "__main__":
    unittest.main()
