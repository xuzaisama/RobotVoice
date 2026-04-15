"""麦克风采集 + Google 语音识别（需联网）+ 可选声纹过滤。"""

import logging
from typing import Callable, Optional

import speech_recognition as sr

from . import config
from . import tts
from .voiceprint import VoiceprintVerifier

logger = logging.getLogger(__name__)


class VoiceListener:
    """封装环境噪声校准与一次拾音识别。"""

    def __init__(self) -> None:
        self._recognizer = sr.Recognizer()
        self._mic: Optional[sr.Microphone] = None
        self._calibrated = False

    def __enter__(self) -> "VoiceListener":
        self._mic = sr.Microphone()
        self._mic.__enter__()
        logger.info("语音识别已启动：麦克风已打开，等待输入。")
        return self

    def __exit__(self, *args) -> None:
        if self._mic is not None:
            self._mic.__exit__(*args)
            self._mic = None
        logger.info("语音识别已停止：麦克风已关闭。")
        self._calibrated = False

    def calibrate_ambient(self) -> None:
        assert self._mic is not None
        if not self._calibrated:
            logger.info("正在进行环境噪声校准…")
            self._recognizer.adjust_for_ambient_noise(
                self._mic, duration=config.AMBIENT_DURATION
            )
            self._calibrated = True

    def capture_audio_once(self, ignore_tts_guard: bool = False) -> Optional[sr.AudioData]:
        """阻塞采集一段语音音频，超时返回 None。"""
        assert self._mic is not None
        remaining_block = tts.remaining_listen_block()
        if (not ignore_tts_guard) and remaining_block > 0:
            logger.info("监听仍处于播报保护期，剩余 %.2f 秒。", remaining_block)
            return None
        if ignore_tts_guard and remaining_block > 0:
            logger.info("当前处于声纹录入模式，绕过播报保护期 %.2f 秒。", remaining_block)
        logger.info("开始监听语音输入…")
        try:
            audio = self._recognizer.listen(
                self._mic,
                timeout=config.LISTEN_TIMEOUT,
                phrase_time_limit=config.PHRASE_TIME_LIMIT,
            )
        except sr.WaitTimeoutError:
            logger.info("本轮监听超时，未检测到语音。")
            return None

        logger.info("已检测到语音，正在处理音频…")
        return audio

    def recognize_audio(self, audio: sr.AudioData) -> Optional[str]:
        """将已采集的音频送入语音识别服务。"""
        try:
            text = self._recognizer.recognize_google(
                audio, language=config.RECOGNITION_LANGUAGE
            )
            logger.info("语音接收完成，识别结果：%s", text)
            return text
        except sr.UnknownValueError:
            logger.warning("语音接收完成，但未能识别出有效内容。")
            return None
        except sr.RequestError as e:
            logger.error("识别服务不可用: %s", e)
            return None

    def listen_once(
        self,
        voiceprint_verifier: Optional[VoiceprintVerifier] = None,
        on_voiceprint_checked: Optional[Callable[[bool, float], None]] = None,
    ) -> Optional[str]:
        """
        阻塞：等待一次语音并返回识别文本；超时或失败返回 None。
        """
        audio = self.capture_audio_once()
        if audio is None:
            return None

        if voiceprint_verifier is not None and voiceprint_verifier.enrolled:
            raw = audio.get_raw_data(
                convert_rate=config.VOICEPRINT_SAMPLE_RATE,
                convert_width=2,
            )
            matched, score = voiceprint_verifier.verify_raw(raw)
            if on_voiceprint_checked is not None:
                on_voiceprint_checked(matched, score)
            if not matched:
                logger.warning("声纹比对未通过，已忽略该段语音。相似度=%.3f", score)
                return None
            logger.info("声纹比对通过，相似度=%.3f", score)

        logger.info("开始语音内容识别…")
        text = self.recognize_audio(audio)
        if not text:
            return None

        should_ignore, reason = tts.should_ignore_recognized_text(text)
        if should_ignore:
            logger.warning("检测到疑似播报回声，已忽略该次识别：%s", reason)
            return None
        return text
