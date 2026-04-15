"""控制逻辑：执行动作、自干扰抑制（播报时不监听）、指令去重。"""

import logging
import threading
import time
from typing import Callable, Optional

from . import config
from .config import match_command
from .tts import speak

logger = logging.getLogger(__name__)

# 动作回调类型：便于接入真实机器人 / 仿真
ActionHandler = Callable[[str], None]


def default_action_handler(cmd_id: str) -> None:
    """默认：在终端打印模拟动作（可替换为串口、ROS 等）。"""
    labels = {
        "forward": "[动作] 机器人前进",
        "backward": "[动作] 机器人后退",
        "stop": "[动作] 机器人停止",
        "turn_left": "[动作] 机器人左转",
        "turn_right": "[动作] 机器人右转",
        "status": "[状态] 运行正常，传感器就绪",
    }
    print(labels.get(cmd_id, f"[动作] 未知指令 {cmd_id}"))


class RobotController:
    """
    - speaking: 为 True 时主循环不得调用麦克风（防止自触发）。
    - 播报结束后再经 COOLDOWN_AFTER_SPEECH 恢复监听。
    - 相同指令在 COMMAND_DEBOUNCE 秒内不重复执行。
    """

    def __init__(
        self,
        on_action: Optional[ActionHandler] = None,
    ) -> None:
        self._on_action = on_action or default_action_handler
        self._speaking = False
        self._lock = threading.Lock()
        self._last_cmd: Optional[str] = None
        self._last_cmd_time: float = 0.0

    @property
    def speaking(self) -> bool:
        with self._lock:
            return self._speaking

    def _set_speaking(self, v: bool) -> None:
        with self._lock:
            self._speaking = v

    def _should_debounce(self, cmd_id: str) -> bool:
        now = time.monotonic()
        if self._last_cmd == cmd_id and (now - self._last_cmd_time) < config.COMMAND_DEBOUNCE:
            logger.info("去重：忽略重复指令 %s", cmd_id)
            return True
        self._last_cmd = cmd_id
        self._last_cmd_time = now
        return False

    def process_recognized_text(self, text: str) -> None:
        """解析文本 -> 执行 -> 语音反馈（全程 speaking 为 True 时不应被调用）。"""
        cmd_id = match_command(text)
        if cmd_id is None:
            logger.info("识别结果无匹配指令: %s", text)

            def _start_u() -> None:
                self._set_speaking(True)

            def _end_u() -> None:
                self._set_speaking(False)

            speak(config.UNKNOWN_FEEDBACK, on_start=_start_u, on_end=_end_u)
            time.sleep(config.COOLDOWN_AFTER_SPEECH)
            return

        if self._should_debounce(cmd_id):
            return

        logger.info("指令: %s (原文: %s)", cmd_id, text)
        self._on_action(cmd_id)

        feedback = config.COMMAND_FEEDBACK.get(cmd_id, config.UNKNOWN_FEEDBACK)

        def _start() -> None:
            self._set_speaking(True)

        def _end() -> None:
            self._set_speaking(False)

        speak(feedback, on_start=_start, on_end=_end)
        time.sleep(config.COOLDOWN_AFTER_SPEECH)
