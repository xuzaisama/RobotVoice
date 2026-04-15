"""
机器人听觉感知系统 — 主程序

闭环：语音输入 → 识别 → 行为执行 → 语音反馈
自干扰抑制：仅在非播报状态下采集麦克风（播报时绝不监听）。
"""

import argparse
import logging
import sys
import time
from typing import Optional

from . import config
from .controller import RobotController, default_action_handler
from .enrollment import ensure_voiceprint_enrolled
from .listener import VoiceListener
from .voiceprint import VoiceprintVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_loop(duration_sec: Optional[float] = None) -> None:
    """持续运行；duration_sec 为 None 则直到 Ctrl+C。"""
    controller = RobotController(on_action=default_action_handler)
    verifier = VoiceprintVerifier() if config.VOICEPRINT_ENABLED else None
    start = time.monotonic()

    logger.info(
        "系统启动。支持指令: %s",
        ", ".join(config.COMMAND_KEYWORDS.keys()),
    )
    logger.info("说话时请清晰说出：前进 / 后退 / 停止 / 左转 / 右转 / 状态")
    logger.info("播报过程中不会采集麦克风，避免自触发。按 Ctrl+C 退出。")
    logger.info("准备启动语音识别监听。")

    with VoiceListener() as listener:
        listener.calibrate_ambient()
        if verifier is not None:
            verifier.warmup()
            ensure_voiceprint_enrolled(listener, verifier)
        while True:
            if duration_sec is not None and (time.monotonic() - start) >= duration_sec:
                logger.info("已达到运行时长，正常退出。")
                break

            if controller.speaking:
                time.sleep(0.05)
                continue

            text = listener.listen_once(voiceprint_verifier=verifier)
            if not text:
                continue

            controller.process_recognized_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="机器人听觉感知系统")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="运行秒数（默认无限，直至 Ctrl+C）。课程要求可设 70 验证≥1分钟稳定运行",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="启动 3D 可视化界面（语音控制小机器人在 3D 场景中行走）",
    )
    args = parser.parse_args()
    try:
        if args.gui:
            from .gui_app import run_gui

            run_gui(duration_sec=args.duration)
        else:
            run_loop(duration_sec=args.duration)
    except KeyboardInterrupt:
        logger.info("用户中断，退出。")
        sys.exit(0)


if __name__ == "__main__":
    main()
