"""语音播报：优先 macOS `say` 音色，其它平台可装 pyttsx3 作后备。"""

from collections import deque
from difflib import SequenceMatcher
import logging
import platform
import subprocess
import sys
import threading
import time
from typing import Callable, Optional

from . import config
from .config import normalize_text

logger = logging.getLogger(__name__)

# macOS 优先尝试“曼波/Mambo”语音包；若当前系统未安装，则自动回退。
# 可在系统「辅助功能/听写」或 `say -v '?'` 查看本机可用音色。
_DARWIN_VOICES = ("Mambo", "Ting-Ting", "Mei-Jia", "Sin-ji")

_GUARD_LOCK = threading.Lock()
_RECENT_SPOKEN: deque[tuple[str, float]] = deque(maxlen=8)
_LISTEN_BLOCK_UNTIL = 0.0


def _estimate_tts_duration(text: str) -> float:
    norm = normalize_text(text)
    if not norm:
        return 0.0
    return len(norm) * config.TTS_ESTIMATED_CHAR_SECONDS


def _calc_guard_seconds(text: str, actual_duration: Optional[float] = None) -> float:
    estimated = _estimate_tts_duration(text)
    base = max(estimated, actual_duration or 0.0)
    guard = base + config.TTS_EXTRA_GUARD_SECONDS
    return max(config.POST_SPEECH_LISTEN_BLOCK, min(config.TTS_MAX_GUARD_SECONDS, guard))


def _remember_spoken_text(text: str, hold_seconds: Optional[float] = None) -> None:
    norm = normalize_text(text)
    if not norm:
        return
    if hold_seconds is None:
        hold_seconds = config.RECENT_SPOKEN_TEXT_WINDOW
    expires_at = time.monotonic() + max(0.0, hold_seconds)
    with _GUARD_LOCK:
        _RECENT_SPOKEN.append((norm, expires_at))


def _set_listen_block(seconds: float) -> None:
    global _LISTEN_BLOCK_UNTIL
    until = time.monotonic() + max(0.0, seconds)
    with _GUARD_LOCK:
        _LISTEN_BLOCK_UNTIL = max(_LISTEN_BLOCK_UNTIL, until)


def remaining_listen_block() -> float:
    with _GUARD_LOCK:
        return max(0.0, _LISTEN_BLOCK_UNTIL - time.monotonic())


def should_ignore_recognized_text(text: str) -> tuple[bool, Optional[str]]:
    """
    若识别结果与最近播报内容高度相似，则判为自触发回声并忽略。
    """
    norm = normalize_text(text)
    if not norm:
        return False, None

    now = time.monotonic()
    with _GUARD_LOCK:
        recent = list(_RECENT_SPOKEN)

    for spoken, expires_at in reversed(recent):
        if now > expires_at:
            continue
        if norm == spoken or norm in spoken or spoken in norm:
            return True, "命中最近播报文本"
        ratio = SequenceMatcher(None, norm, spoken).ratio()
        if ratio >= config.RECENT_SPOKEN_TEXT_SIMILARITY:
            return True, f"与最近播报文本相似度过高({ratio:.2f})"
    return False, None


def _speak_darwin(text: str) -> None:
    last_err: Optional[Exception] = None
    for voice in _DARWIN_VOICES:
        try:
            logger.info("尝试使用 macOS 音色播报：%s", voice)
            subprocess.run(
                ["say", "-v", voice, text],
                check=True,
                timeout=60,
                capture_output=True,
            )
            logger.info("当前播报音色：%s", voice)
            return
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            last_err = e
            continue
    if last_err:
        logger.warning("say 失败，尝试无指定音色: %s", last_err)
    subprocess.run(["say", text], check=False, timeout=60, capture_output=True)


def _speak_pyttsx3(text: str) -> None:
    try:
        import pyttsx3
    except ImportError as e:
        raise RuntimeError("请安装 pyttsx3: pip install pyttsx3") from e
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def speak(
    text: str,
    on_start: Optional[Callable[[], None]] = None,
    on_end: Optional[Callable[[], None]] = None,
) -> None:
    """
    阻塞式播报。在开始前/结束后调用回调，便于与「禁止监听」联动。
    """
    if on_start:
        on_start()
    estimated_guard = _calc_guard_seconds(text)
    _remember_spoken_text(
        text,
        hold_seconds=min(config.RECENT_SPOKEN_TEXT_MAX_WINDOW, estimated_guard + config.RECENT_SPOKEN_TEXT_WINDOW),
    )
    start = time.monotonic()
    try:
        if sys.platform == "darwin":
            _speak_darwin(text)
        else:
            _speak_pyttsx3(text)
    finally:
        actual_duration = max(0.0, time.monotonic() - start)
        guard_seconds = _calc_guard_seconds(text, actual_duration=actual_duration)
        _remember_spoken_text(
            text,
            hold_seconds=min(config.RECENT_SPOKEN_TEXT_MAX_WINDOW, guard_seconds + config.RECENT_SPOKEN_TEXT_WINDOW),
        )
        logger.info(
            "播报完成，实际时长 %.2f 秒，动态保护 %.2f 秒。",
            actual_duration,
            guard_seconds,
        )
        _set_listen_block(guard_seconds)
        if on_end:
            on_end()
