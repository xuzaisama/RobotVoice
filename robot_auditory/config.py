"""系统参数与指令词表（不少于 5 类控制指令）。"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 识别使用 Google Web Speech API 时的语言（需联网）
RECOGNITION_LANGUAGE = "zh-CN"

# 单次监听：环境噪声校准时间（秒）
AMBIENT_DURATION = 0.6

# 单次拾音上限（秒），允许一次说出多段指令
PHRASE_TIME_LIMIT = 10.0

# 无语音时 listen 超时（秒）
LISTEN_TIMEOUT = 5.0

# 播报结束后延迟再恢复监听（秒），抑制扬声器尾音被麦克风拾取
COOLDOWN_AFTER_SPEECH = 0.7

# 更强的播报后监听封锁窗口，专门压制扬声器尾音/回声
POST_SPEECH_LISTEN_BLOCK = 1.8

# 最近播报文本的回声屏蔽窗口与相似度阈值
RECENT_SPOKEN_TEXT_WINDOW = 8.0
RECENT_SPOKEN_TEXT_SIMILARITY = 0.72
RECENT_SPOKEN_TEXT_MAX_WINDOW = 30.0

# 长播报保护：根据文本长度/实际播报时长动态延长保护时间
TTS_ESTIMATED_CHAR_SECONDS = 0.22
TTS_EXTRA_GUARD_SECONDS = 1.5
TTS_MAX_GUARD_SECONDS = 3.0

# 相同指令去重：该时间窗内不重复执行（秒）
COMMAND_DEBOUNCE = 2.5

# 指令类别 -> 关键词列表（命中任一则归为该指令）
COMMAND_KEYWORDS: Dict[str, List[str]] = {
    "forward": [
        "前进",
        "向前",
        "往前走",
        "往前",
        "直走",
        "继续前进",
        "继续往前",
        "往前开",
        "朝前走",
        "向前走",
    ],
    "backward": [
        "后退",
        "向后",
        "往后退",
        "往后",
        "退后",
        "向后退",
        "往后走",
        "倒车",
    ],
    "stop": [
        "停止",
        "停下",
        "停",
        "别动",
        "刹车",
        "暂停",
        "终止",
        "原地别动",
        "不要动",
    ],
    "turn_left": [
        "左转",
        "向左",
        "往左",
        "往左转",
        "向左转",
        "左拐",
        "往左拐",
        "向左拐",
        "朝左转",
        "朝左",
        "掉头",
        "调头",
        "向后转",
        "转身",
        "左转弯",
    ],
    "turn_right": [
        "右转",
        "向右",
        "往右",
        "往右转",
        "向右转",
        "右拐",
        "往右拐",
        "向右拐",
        "朝右转",
        "朝右",
        "右转弯",
    ],
    "status": [
        "状态",
        "报告",
        "你在哪",
        "现在怎样",
        "什么情况",
        "汇报状态",
        "报告状态",
        "当前位置",
        "现在到哪了",
    ],  # 第 6 类：查询（加分演示）
}

# 指令 -> 语音反馈文案
COMMAND_FEEDBACK: Dict[str, str] = {
    "forward": "收到，正在前进。",
    "backward": "收到，正在后退。",
    "stop": "收到，已停止。",
    "turn_left": "收到，正在左转。",
    "turn_right": "收到，正在右转。",
    "status": "系统运行正常，等待您的指令。",
}

# 未知指令时的反馈
UNKNOWN_FEEDBACK = "没有听清有效指令，请再说一遍。"

# 距离单位：一“步”为多少米（也用于默认“前进/后退”的距离）
UNIT_STEP_M = 0.5

# 默认：说“前进/后退”但未指定距离时，按 1 步处理
DEFAULT_MOVE_DISTANCE_M = 1.0 * UNIT_STEP_M

# 默认：说“左转/右转”但未指定角度时，按 90 度处理
DEFAULT_TURN_ANGLE_DEG = 90.0

# 声纹配置：首次运行时录入，后续仅允许匹配到已录入声纹的语音进入指令识别
VOICEPRINT_ENABLED = True
VOICEPRINT_PATH = Path("voiceprint_profile.json")
VOICEPRINT_MODEL_VERSION = 4
VOICEPRINT_SAMPLE_RATE = 16000
VOICEPRINT_MODEL_NAME = "ecapatdnn_voxceleb12"
VOICEPRINT_MODEL_CACHE_DIR = Path(".cache/paddlespeech")
VOICEPRINT_SAMPLES = 5
VOICEPRINT_MIN_VALID_SAMPLES = 4
VOICEPRINT_MATCH_THRESHOLD = 0.40
VOICEPRINT_MIN_CENTROID_COSINE = 0.93
VOICEPRINT_MIN_NEAREST_COSINE = 0.95
VOICEPRINT_MAX_STANDARD_DISTANCE = 2.2
VOICEPRINT_ENROLLMENT_PHRASES = (
    "你好，现在我要开始录入声纹信息",
    "今天天气很好，适合出门散步",
    "我的电话号码是一二三四五六七",
    "你晚上想吃什么呢？",
    "以上就是我的声纹信息",
)

# 千问 / 百炼配置：默认使用根目录中的 API Key 文件，并固定到华北2（北京）地域
QIANWEN_API_KEY_FILE = Path("千问api_key")
QIANWEN_MODEL = "qwen3.6-plus"
QIANWEN_REGION = "华北2（北京）"
QIANWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LOCAL_CITY = os.getenv("ROBOT_LOCAL_CITY", "上海")
LOCAL_TIMEZONE = os.getenv("ROBOT_LOCAL_TIMEZONE", "Asia/Shanghai")
WEATHER_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class MotionCommand:
    """结构化运动指令，供 GUI/控制层执行。"""

    cmd_id: str
    distance_m: float = 0.0
    angle_deg: float = 0.0
    speed_scale: float = 1.0
    text: str = ""


def normalize_text(text: str) -> str:
    """去掉空白与常见标点，便于匹配。"""
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"[\s，。！？、；：""''「」『』]", "", t)
    return t


def match_command(text: str) -> Optional[str]:
    """
    将识别文本映射为指令 id。
    按关键词长度降序匹配，减少短词误命中。
    """
    norm = normalize_text(text)
    if not norm:
        return None

    best: Optional[Tuple[str, int]] = None  # (cmd_id, keyword_len)
    for cmd_id, keywords in COMMAND_KEYWORDS.items():
        for kw in keywords:
            if kw in norm:
                ln = len(kw)
                if best is None or ln > best[1]:
                    best = (cmd_id, ln)
    return best[0] if best else None


def _ch_digit_to_int(ch: str) -> Optional[int]:
    # 只做单字符数字映射（'一'~'九'、'两'）
    mapping = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    return mapping.get(ch)


def _ch_number_to_float(text: str) -> Optional[float]:
    """解析常见中文整数数字，覆盖“一/两/三十/二十五/一百”等课程口令场景。"""
    if not text:
        return None
    if text in ("半",):
        return 0.5

    digit_map = {
        "零": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    unit_map = {"十": 10, "百": 100}

    if len(text) == 1 and text in digit_map:
        return float(digit_map[text])

    total = 0
    current = 0
    seen = False
    for ch in text:
        if ch in digit_map:
            current = digit_map[ch]
            seen = True
        elif ch in unit_map:
            seen = True
            unit = unit_map[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
        else:
            return None

    if not seen:
        return None
    total += current
    return float(total)


def _extract_number_before_unit(text: str, units: str) -> Optional[float]:
    num_re = rf"(?P<num>[0-9]+(?:\.[0-9]+)?)(?P<unit>{units})"
    m_num = re.search(num_re, text)
    if m_num:
        return float(m_num.group("num"))

    ch_re = rf"(?P<ch>[零一二两三四五六七八九十百半]+)(?P<unit>{units})"
    m_ch = re.search(ch_re, text)
    if m_ch:
        return _ch_number_to_float(m_ch.group("ch"))

    return None


def extract_distance_m(text: str) -> Optional[float]:
    """
    从识别文本中提取“距离（米）”。
    支持单位：
    - “步”：如“一步/两步/3步”
    - “米”：如“一米/两米/1米/2米”

    例如：
    - “前进一步” -> 0.5
    - “后退2米” -> 2.0
    """
    norm = normalize_text(text)
    if not norm:
        return None

    meters = _extract_number_before_unit(norm, "米|m")
    if meters is not None:
        return meters

    steps = _extract_number_before_unit(norm, "步")
    if steps is not None:
        return steps * UNIT_STEP_M

    # “一步/一米”这种若被切词成 “一” + “步/米”外形，已可被上面正则命中；
    # 若只说“步”或“米”，没有数量，返回 None 让上层使用默认距离。
    return None


def extract_angle_deg(text: str) -> Optional[float]:
    """从文本中提取转向角度，支持“30度/三十度”。"""
    norm = normalize_text(text)
    if not norm:
        return None
    return _extract_number_before_unit(norm, "度")


def extract_special_turn_angle_deg(text: str) -> Optional[float]:
    """解析“掉头/向后转/转一圈”等特殊转向表达。"""
    norm = normalize_text(text)
    if not norm:
        return None

    if any(token in norm for token in ("掉头", "调头", "向后转", "转身")):
        return 180.0

    if "转一圈" in norm or "转圈" in norm:
        return 360.0

    half_turn = re.search(r"(半圈)", norm)
    if half_turn:
        return 180.0

    quarter_turn = re.search(r"(四分之一圈)", norm)
    if quarter_turn:
        return 90.0

    return None


def extract_speed_scale(text: str) -> float:
    """从文本中提取速度语义，返回速度倍率。"""
    norm = normalize_text(text)
    if not norm:
        return 1.0

    slow_tokens = ("慢慢", "缓慢", "慢一点", "慢点", "低速")
    fast_tokens = ("快速", "快点", "快一点", "赶快", "加速", "高速", "迅速")

    if any(token in norm for token in slow_tokens):
        return 0.6
    if any(token in norm for token in fast_tokens):
        return 1.5
    return 1.0


def _find_command_spans(norm: str) -> List[Tuple[int, int, str]]:
    matches: List[Tuple[int, int, str]] = []
    for cmd_id, keywords in COMMAND_KEYWORDS.items():
        for kw in keywords:
            kw_norm = normalize_text(kw)
            start = 0
            while True:
                idx = norm.find(kw_norm, start)
                if idx < 0:
                    break
                matches.append((idx, idx + len(kw_norm), cmd_id))
                start = idx + 1

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: List[Tuple[int, int, str]] = []
    covered_until = -1
    for start, end, cmd_id in matches:
        if start < covered_until:
            continue
        selected.append((start, end, cmd_id))
        covered_until = end
    return selected


def match_motion_commands(text: str) -> List[MotionCommand]:
    """
    解析一段文本中的多条运动指令。

    例：“前进1m然后左转30度再前进1m” ->
    forward(1.0m), turn_left(30deg), forward(1.0m)
    """
    norm = normalize_text(text)
    if not norm:
        return []

    spans = _find_command_spans(norm)
    commands: List[MotionCommand] = []
    for i, (start, _end, cmd_id) in enumerate(spans):
        next_start = spans[i + 1][0] if i + 1 < len(spans) else len(norm)
        segment_start = max(0, start - 6)
        segment = norm[segment_start:next_start]

        if cmd_id in ("forward", "backward"):
            dist = extract_distance_m(segment)
            if dist is None:
                dist = DEFAULT_MOVE_DISTANCE_M
            commands.append(
                MotionCommand(
                    cmd_id=cmd_id,
                    distance_m=float(dist),
                    speed_scale=extract_speed_scale(segment),
                    text=segment,
                )
            )
        elif cmd_id in ("turn_left", "turn_right"):
            angle = extract_angle_deg(segment)
            if angle is None:
                angle = extract_special_turn_angle_deg(segment)
            if angle is None:
                angle = DEFAULT_TURN_ANGLE_DEG
            commands.append(
                MotionCommand(
                    cmd_id=cmd_id,
                    angle_deg=float(angle),
                    speed_scale=extract_speed_scale(segment),
                    text=segment,
                )
            )
        elif cmd_id in ("stop", "status"):
            commands.append(MotionCommand(cmd_id=cmd_id, text=segment))

    deduped: List[MotionCommand] = []
    for command in commands:
        if deduped and command.cmd_id == deduped[-1].cmd_id and command.cmd_id in ("stop", "status"):
            continue
        deduped.append(command)

    return deduped


def match_move_command(text: str) -> Optional[Tuple[str, float]]:
    """
    匹配“可运动指令”，并返回 (cmd_id, distance_m)。
    - forward/backward：支持距离；未指定则用默认 1 步
    - turn_left/turn_right/stop/status：distance_m 返回 0.0（不直线移动）
    """
    commands = match_motion_commands(text)
    if not commands:
        return None

    command = commands[0]
    return command.cmd_id, command.distance_m
