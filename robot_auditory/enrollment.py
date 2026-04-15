"""声纹录入流程。"""

from __future__ import annotations

import logging
import time
from typing import List

from . import config
from .tts import speak
from .voiceprint import VoiceprintVerifier

logger = logging.getLogger(__name__)


def ensure_voiceprint_enrolled(listener, verifier: VoiceprintVerifier) -> None:
    """
    若本地没有声纹档案，则引导用户录入。
    listener 需提供 capture_audio_once()。
    """
    if not config.VOICEPRINT_ENABLED or (verifier.enrolled and not verifier.needs_reenroll):
        return

    if verifier.needs_reenroll:
        logger.info("当前声纹模型需要重新录入。")
        speak("检测到新的声纹模型版本，需要重新录入声纹。")
        time.sleep(config.COOLDOWN_AFTER_SPEECH)

    logger.info("开始录入声纹。")
    speak("接下来请按提示录入声纹，并保持每次由同一位使用者发声。")
    time.sleep(config.COOLDOWN_AFTER_SPEECH)

    prompts = config.VOICEPRINT_ENROLLMENT_PHRASES
    if len(prompts) != config.VOICEPRINT_SAMPLES:
        raise RuntimeError("声纹录入提示语数量与录入次数配置不一致。")

    samples: List[bytes] = []
    for idx, prompt in enumerate(prompts):
        sample_no = idx + 1
        speak(f"请录入第 {sample_no} 次声纹。请说：{prompt}")
        time.sleep(config.COOLDOWN_AFTER_SPEECH)

        for attempt in range(3):
            audio = listener.capture_audio_once(ignore_tts_guard=True)
            if audio is None:
                logger.warning("第 %d 次声纹录入超时，重试 %d/3", sample_no, attempt + 1)
                speak("没有听清，请再说一次。")
                time.sleep(config.COOLDOWN_AFTER_SPEECH)
                continue

            raw = audio.get_raw_data(
                convert_rate=config.VOICEPRINT_SAMPLE_RATE,
                convert_width=2,
            )
            samples.append(raw)
            logger.info("已完成第 %d 次声纹录入，提示语为：%s", sample_no, prompt)
            speak(f"第 {sample_no} 次录入完成。")
            time.sleep(config.COOLDOWN_AFTER_SPEECH)
            break
        else:
            raise RuntimeError(f"第 {sample_no} 次声纹录入失败，请重新启动程序后再试。")

    verifier.enroll_from_raw_samples(samples)
    speak("声纹录入完成。后续只有匹配声纹的语音会被识别。")
    time.sleep(config.COOLDOWN_AFTER_SPEECH)
